# 10x_Assignemnt

ROS 2 depth-plane detection and analysis project. The main node subscribes to a depth image topic, fits a plane using RANSAC, estimates visible plane area and orientation, and logs results per frame. It also provides a real-time OpenCV visualization and estimates a rotation axis from the sequence of detected plane normals.

## Features
- Depth image to point cloud conversion using camera intrinsics
- Noise filtering and outlier removal for robust plane fitting
- RANSAC-based plane detection with adaptive tolerance
- Plane area estimation via convex hull
- Plane normal and angle-to-camera computation
- Per-frame CSV logging and rotation-axis export
- OpenCV visualization of detected plane region

## Repository layout
```
.
├── ros2_ws/                     # ROS 2 workspace
│   ├── src/my_package/          # Python ROS 2 package
│   ├── frame_results.csv        # Output (generated at runtime)
│   └── rotation_axis.txt        # Output (generated at runtime)
└── New_assesment/
    ├── Perception Assignment.pdf
    └── depth/                   # Sample ROS 2 bag (depth.db3)
```

## Prerequisites
- ROS 2 (Humble or compatible) with `rclpy`, `sensor_msgs`, and `cv_bridge`
- Python 3 packages: `numpy`, `scipy`, `scikit-learn`, `opencv-python`

## Build
```bash
cd /home/runner/work/10x_Assignemnt/10x_Assignemnt/ros2_ws
colcon build
source install/setup.bash
```

## Run
### Start the node
```bash
ros2 run my_package depth_debug
# or
ros2 run my_package main_node
```

### Play the sample bag (optional)
```bash
ros2 bag play /home/runner/work/10x_Assignemnt/10x_Assignemnt/New_assesment/depth/depth.db3
```

The node subscribes to `/depth` and opens a visualization window with plane overlay, visible area, and angle readout.

## Outputs
Generated in `ros2_ws/` when the node exits:
- `frame_results.csv`: per-frame visible area and normal angle
- `rotation_axis.txt`: estimated rotation axis vector

## Configuration notes
Camera intrinsics and thresholds are currently defined in:
`ros2_ws/src/my_package/my_package/depth_debug.py`

Key defaults:
- Depth range filter: 0.3–3.0 m
- Topic: `/depth`

## Tests
```bash
cd /home/runner/work/10x_Assignemnt/10x_Assignemnt/ros2_ws
colcon test
```
