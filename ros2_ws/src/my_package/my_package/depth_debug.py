#!/usr/bin/env python3
"""
Depth Plane Detection and Analysis Node
This ROS2 node processes depth camera data to detect and analyze planar surfaces.
Features include:
- Robust plane fitting using RANSAC algorithm
- Adaptive noise filtering and outlier removal
- Real-time visualization of detected planes
- Measurement of plane area and orientation
- Rotation axis estimation from plane normal sequences
- CSV logging of frame-by-frame measurements
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.linear_model import RANSACRegressor
import csv

# --- Camera intrinsics (replace with your actual values) ---
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5


class DepthDebugger(Node):
    def __init__(self):
        super().__init__('depth_debugger')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/depth', self.depth_callback, 10)
        self.subscription

        # Store frame results
        self.frame_results = []  # (frame_number, area, normal_angle)
        self.normals = []
        self.frame_count = 0

    # ---------- Utility Functions ----------

    def depth_to_point_cloud(self, depth):
        """Convert depth image to 3D point cloud."""
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        z = depth / 1000.0  # convert mm → meters if needed
        x = (c - cx) * z / fx
        y = (r - cy) * z / fy
        points = np.dstack((x, y, z))
        return points.reshape(-1, 3)

    def filter_point_cloud(self, points):
        """Remove noisy / invalid 3D points using range and statistical filters."""
        # Remove NaN or infinite values
        mask = np.isfinite(points[:, 2])
        points = points[mask]

        # Remove depth points that are too close or too far
        points = points[(points[:, 2] > 0.3) & (points[:, 2] < 3.0)]

        if len(points) == 0:
            return points

        # Statistical filtering: remove outliers based on z-distance clustering
        z_values = points[:, 2]
        mean_z = np.mean(z_values)
        std_z = np.std(z_values)
        mask = np.abs(z_values - mean_z) < 2.0 * std_z
        filtered_points = points[mask]

        return filtered_points

    def fit_plane(self, points):
        """
        Fit a plane to 3D points using RANSAC algorithm with adaptive tolerance.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            tuple: (plane_points, coefficients, intercept) or (None, None, None) if fitting fails
        """
        points = points[np.isfinite(points[:, 2])]
        if len(points) < 10:
            return None, None, None
            
        X = points[:, :2]  # Use x,y coordinates as features
        y = points[:, 2]   # Use z coordinate as target
        
        # Adaptive RANSAC tolerance based on depth variance (2.5-6mm range)
        depth_std = np.std(points[:, 2])
        tol = max(0.0025, min(0.006, 0.0025 + 0.5 * depth_std))
        
        ransac = RANSACRegressor(residual_threshold=tol)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        plane_points = points[inlier_mask]
        return plane_points, ransac.estimator_.coef_, ransac.estimator_.intercept_

    def compute_area(self, plane_points):
        """
        Calculate the area of a plane using convex hull of projected points.
        
        Args:
            plane_points: Nx3 array of plane inlier points
            
        Returns:
            float: Area in square meters
        """
        if len(plane_points) < 3:
            return 0.0
        hull = ConvexHull(plane_points[:, :2])  # Use x,y projection
        return hull.area

    def plane_normal(self, coef):
        """
        Calculate plane normal vector from RANSAC coefficients.
        
        Args:
            coef: RANSAC plane coefficients [a, b] where z = ax + by + c
            
        Returns:
            numpy.ndarray: Normalized plane normal vector
        """
        a, b = coef
        n = np.array([-a, -b, 1.0])
        return n / np.linalg.norm(n)

    def angle_with_camera(self, normal):
        """
        Calculate angle between plane normal and camera optical axis.
        
        Args:
            normal: Plane normal vector
            
        Returns:
            float: Angle in degrees
        """
        camera_dir = np.array([0, 0, 1])  # Camera points along positive z-axis
        cos_theta = np.clip(np.dot(normal, camera_dir), -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return np.degrees(angle)

    def estimate_rotation_axis(self, normals):
        """
        Estimate rotation axis from sequence of plane normal vectors using PCA.
        
        Args:
            normals: List of normal vectors from multiple frames
            
        Returns:
            numpy.ndarray: Estimated rotation axis vector
        """
        normals = np.array(normals)
        cov = np.cov(normals.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmin(eigvals)]  # Eigenvector with smallest eigenvalue
        return axis / np.linalg.norm(axis)

    # ---------- Core Callback ----------

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # --- NEW: Apply median filter to reduce pixel-level noise ---
        depth = cv2.medianBlur(depth, 5)

        # Convert to point cloud
        points = self.depth_to_point_cloud(depth)

        # Filter noisy points
        filtered_points = self.filter_point_cloud(points)
        if len(filtered_points) < 50:
            self.get_logger().warn("Too few valid points after filtering.")
            return

        # Fit plane
        plane_points, coef, intercept = self.fit_plane(filtered_points)
        if plane_points is None:
            self.get_logger().warn("Not enough valid depth points for plane.")
            return

        # --- Adaptive soft filtering: keep plane broad, reject noisy outliers ---
        a, b = coef
        c = intercept
        z_pred = a * plane_points[:, 0] + b * plane_points[:, 1] + c
        z_actual = plane_points[:, 2]
        dist = np.abs(z_actual - z_pred)

        mean_depth = np.mean(plane_points[:, 2])
        adaptive_tol = max(0.005, 0.003 + 0.002 * mean_depth)  # ~5–9 mm
        mask = dist < adaptive_tol
        plane_points = plane_points[mask]

        # --- Local outlier suppression based on neighbor density ---
        if len(plane_points) > 50:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=10).fit(plane_points)
            distances, _ = nbrs.kneighbors(plane_points)
            local_density = np.mean(distances, axis=1)
            dense_mask = local_density < np.percentile(local_density, 95)
            plane_points = plane_points[dense_mask]

        # Compute area and plane normal
        area = self.compute_area(plane_points)
        if len(plane_points) == 0:
            self.get_logger().warn("No plane inliers after filtering.")
            return

        normal = self.plane_normal(coef)

        # Smooth normal across frames
        if len(self.normals) > 0:
            normal = 0.6 * normal + 0.4 * self.normals[-1]
        normal /= np.linalg.norm(normal)

        angle = self.angle_with_camera(normal)

        #  Optional: Stabilize angle display to reduce flicker
        if hasattr(self, 'last_angle'):
            angle = 0.7 * self.last_angle + 0.3 * angle
        self.last_angle = angle

        # --- Relaxed but robust validity checks ---
        # Use the mean_depth we already calculated above
        if mean_depth > 2.0:  # allow up to 2.0m distance
            self.get_logger().warn("Rejected far plane (likely background wall).")
            return

        if area < 0.02:  # smaller threshold to keep full faces
            self.get_logger().warn("Rejected small plane (area too small).")
            return

        if len(plane_points) < 140:  # relaxed inlier requirement
            self.get_logger().warn("Rejected weak plane (too few inliers).")
            return

        if angle > 62:  # allow slightly tilted planes
            self.get_logger().warn(f"Rejected tilted plane (angle {angle:.2f}°).")
            return

        self.normals.append(normal)

        # Save frame result
        self.frame_count += 1
        self.frame_results.append((self.frame_count, area, angle))

        print(f"Frame {self.frame_count}: Visible area: {area:.3f} m², Normal angle: {angle:.2f}°")

        # ---------- Visualization ----------
        vis_image = cv2.convertScaleAbs(depth, alpha=0.05)  # grayscale depth image
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)  # convert to BGR for coloring

        # Project plane points back to pixel coordinates
        plane_pts_3d = plane_points
        u = ((plane_pts_3d[:, 0] * fx) / plane_pts_3d[:, 2]) + cx
        v = ((plane_pts_3d[:, 1] * fy) / plane_pts_3d[:, 2]) + cy
        pixels = np.stack((u, v), axis=-1).astype(np.int32)

        h, w = depth.shape
        pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
                        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)]

        if len(pixels) > 3:
            # Convert projected pixels to a mask to remove sparse points
            mask_img = np.zeros((h, w), np.uint8)
            mask_img[pixels[:, 1], pixels[:, 0]] = 255

            # Smooth + stable contour cleaning
            mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask_img = cv2.erode(mask_img, np.ones((3, 3), np.uint8), iterations=1)
            
            # Get coordinates from cleaned mask
            coords = np.column_stack(np.where(mask_img > 0))
            if len(coords) > 3:
                # Compute hull on cleaned coordinates (note: coords are [row, col] so we swap to [col, row])
                hull = cv2.convexHull(coords[:, [1, 0]])

                # Draw a semi-transparent highlight
                overlay = vis_image.copy()
                cv2.drawContours(overlay, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
                vis_image = cv2.addWeighted(overlay, 0.25, vis_image, 0.75, 0)

                # Add bright boundary outline
                cv2.polylines(vis_image, [hull], True, (0, 255, 0), 2)

        # Add text overlay
        cv2.putText(vis_image, f"Area: {area:.3f} m²", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Angle: {angle:.2f}°", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Depth Plane Visualization", vis_image)
        cv2.waitKey(1)

    # ---------- Saving ----------
    def save_results(self):
        # Save CSV/TXT for frames
        with open("frame_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Visible Area (m²)", "Normal Angle (deg)"])
            writer.writerows(self.frame_results)
        self.get_logger().info("Frame results saved to frame_results.csv")

        # Save rotation axis
        axis = self.estimate_rotation_axis(self.normals)
        with open("rotation_axis.txt", "w") as f:
            f.write(f"{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}")
        self.get_logger().info("Rotation axis saved to rotation_axis.txt")


def main(args=None):
    rclpy.init(args=args)
    node = DepthDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
