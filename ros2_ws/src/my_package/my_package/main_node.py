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
        """Fit a plane using RANSAC."""
        points = points[np.isfinite(points[:, 2])]
        if len(points) < 10:
            return None, None, None
        X = points[:, :2]
        y = points[:, 2]
        ransac = RANSACRegressor(residual_threshold=0.01)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        plane_points = points[inlier_mask]
        return plane_points, ransac.estimator_.coef_, ransac.estimator_.intercept_

    def compute_area(self, plane_points):
        if len(plane_points) < 3:
            return 0.0
        hull = ConvexHull(plane_points[:, :2])
        return hull.area

    def plane_normal(self, coef):
        a, b = coef
        n = np.array([-a, -b, 1.0])
        return n / np.linalg.norm(n)

    def angle_with_camera(self, normal):
        camera_dir = np.array([0, 0, 1])
        cos_theta = np.clip(np.dot(normal, camera_dir), -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return np.degrees(angle)

    def estimate_rotation_axis(self, normals):
        normals = np.array(normals)
        cov = np.cov(normals.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmin(eigvals)]
        return axis / np.linalg.norm(axis)

    # ---------- Core Callback ----------

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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

        # Compute area & normal
        area = self.compute_area(plane_points)
        normal = self.plane_normal(coef)
        angle = self.angle_with_camera(normal)
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
            hull = cv2.convexHull(pixels)

            # Draw a semi-transparent highlight
            overlay = vis_image.copy()
            cv2.drawContours(overlay, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
            vis_image = cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0)

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
