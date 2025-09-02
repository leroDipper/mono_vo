import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load ground truth
with open("office_dataset_aruco/ground_truth_poses.json", "r") as file:
    ground_truth = json.load(file)

# Extract marker info
marker_info = ground_truth['aruco_markers']['markers'][0]  # Marker 0
marker_size_meters = marker_info['size']  # USE ACTUAL SIZE FROM BLENDER
print(f"Using marker size: {marker_size_meters}m")

# Camera intrinsics
camera_intrinsics = ground_truth["camera_intrinsics"]
K = np.array([[camera_intrinsics["fx"], 0, camera_intrinsics["cx"]],
              [0, camera_intrinsics["fy"], camera_intrinsics["cy"]],
              [0, 0, 1]], dtype=np.float32)

distortion = np.zeros(5, dtype=np.float32)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# IMPROVED detector parameters
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.03
parameters.minCornerDistanceRate = 0.05
parameters.minMarkerDistanceRate = 0.05

images_path = "office_dataset_aruco/left"
marker_poses = []
frame_numbers = []

for filename in sorted(os.listdir(images_path)):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(images_path, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None and 0 in ids.flatten():
            idx = np.where(ids.flatten() == 0)[0][0]
            marker_corners = corners[idx]
            
            # CRITICAL FIX: Use correct marker size
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], marker_size_meters, K, distortion
            )
            
            rvec = rvecs[0].flatten()
            tvec = tvecs[0].flatten()
            
            # CRITICAL: Transform from camera frame to world frame
            # tvec is marker position in camera coordinates
            # We need to invert this to get camera position relative to marker
            
            # Get rotation matrix from rvec
            R, _ = cv2.Rodrigues(rvec)
            
            # Camera position in marker frame = -R.T @ tvec
            camera_pos_marker_frame = -R.T @ tvec
            
            # Since marker is at origin in world, camera position = camera_pos_marker_frame
            marker_poses.append(camera_pos_marker_frame)
            
            # Extract frame number for plotting
            frame_num = int(filename.split('_')[1].split('.')[0])
            frame_numbers.append(frame_num)
            
            print(f"Frame {frame_num}: Marker at {tvec}")
            
            # Visualize
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.drawFrameAxes(img, K, distortion, rvec, tvec, 0.1)
            
        cv2.imshow('ArUco Detection', cv2.resize(img, (800, 600)))
        cv2.waitKey(50)

cv2.destroyAllWindows()

if marker_poses:
    poses = np.array(marker_poses)
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY trajectory
    ax1.plot(poses[:, 0], poses[:, 1], 'b-o', markersize=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Marker 0 Trajectory (Top View)')
    ax1.axis('equal')
    ax1.grid(True)
    
    # Distance over time
    distances = np.linalg.norm(poses, axis=1)
    ax2.plot(frame_numbers, distances, 'r-o', markersize=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance to Marker')
    ax2.grid(True)
    
    # XZ trajectory (side view)
    ax3.plot(poses[:, 0], poses[:, 2], 'g-o', markersize=2)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Marker 0 Trajectory (Side View)')
    ax3.grid(True)
    
    # Y position over time
    ax4.plot(frame_numbers, poses[:, 1], 'purple', marker='o', markersize=2)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Y Position vs Frame')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate if trajectory looks circular
    center_x = np.mean(poses[:, 0])
    center_y = np.mean(poses[:, 1])
    radii = np.sqrt((poses[:, 0] - center_x)**2 + (poses[:, 1] - center_y)**2)
    
    print(f"\nDetected {len(marker_poses)} poses")
    print(f"Trajectory center: ({center_x:.3f}, {center_y:.3f})")
    print(f"Mean radius: {np.mean(radii):.3f}m ± {np.std(radii):.3f}m")
    print(f"Distance variation: {np.std(distances):.3f}m")
    
    if np.std(radii) < 0.1:
        print("✓ Circular trajectory detected!")
    else:
        print("✗ Trajectory not circular - check marker size or detection")
else:
    print("No markers detected!")