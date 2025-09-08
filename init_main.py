import numpy as np
import cv2
from modules.feature_extractor import SIFT, ORB, BRISK
from modules.frame_loader import FrameLoader
import matplotlib.pyplot as plt
from modules.pose_estimation import MotionEstimator
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json


if __name__ == "__main__":
    # === Load and Prepare Ground Truth Data ===
    
    with open('office_dataset_aruco/ground_truth_poses.json', 'r') as file:
        ground_truth = json.load(file)

    # --- Paths to Blender dataset images ---
    images_path = "office_dataset_aruco/left"

    # Extract marker info
    marker_info = ground_truth['aruco_markers']['markers'][0]  # Marker 0
    marker_size_meters = marker_info['size']  # USE ACTUAL SIZE FROM BLENDER

    
    images = [images_path]

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

    marker_poses = []
    max_images = 122

    
    # Initialize VO state
    current_pose = np.eye(4)
    trajectory = []
    
    # Previous frame data
    prev_image = None
    prev_kp = None
    prev_desc = None
    prev_points_3d = None
    
    # Scale estimation variables
    base_scale = None  # Will be set from first marker pair
    accumulated_vo_path = []  # Store VO translations for path-based scaling
    marker_indices = []  # Frames where markers were detected

    # Initialize components
    extractor_SIFT = SIFT(n_features=1500)
    load_frames = FrameLoader(images_path=images, max_images=max_images)
    motion_estimator = MotionEstimator(camera_matrix=K, method='essential')
    #triangulator = Triangulator(camera_matrix=K)
    
    print(f"Processing {max_images} frames...")
    
    for idx, image in enumerate(load_frames.get_frames()):
        print(f"Processing frame {idx}")

        # Detect markers for ground truth positioning
        if idx <= 1 or idx % 2 == 0:
        #if idx <= 2 or idx % 15 == 0:
            corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
            if ids is not None and 0 in ids.flatten():
                id_idx = np.where(ids.flatten() == 0)[0][0]
                marker_corners = corners[id_idx]
             
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [marker_corners], marker_size_meters, K, distortion
                )
                
                rvec = rvecs[0].flatten()
                tvec = tvecs[0].flatten()
                
                R_aruco, _ = cv2.Rodrigues(rvec)
                camera_pos_marker_frame = -R_aruco.T @ tvec
                
                marker_poses.append(camera_pos_marker_frame)
                marker_indices.append(idx)
                print(f"Frame {idx}: Marker position: {camera_pos_marker_frame}")

        # Initialize first frame
        if idx == 0:
            if len(marker_poses) > 0:
                current_pose[:3, 3] = marker_poses[0]
            trajectory.append(current_pose.copy())
            
            # Extract features and continue
            kp, desc = extractor_SIFT.detect_and_compute(image)
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            print(f"Frame {idx}: Initialized with {len(kp)} features")
            continue
        
        # Extract features
        kp, desc = extractor_SIFT.detect_and_compute(image)
        
        # Match features between the current and previous frame
        matches = extractor_SIFT.match_features(prev_desc, desc)

        if len(matches) < 50:
            print(f"Frame {idx}: Not enough matches ({len(matches)}), skipping")
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            trajectory.append(current_pose.copy())  # Repeat last pose
            continue

        # Estimate motion
        R, t, mask = motion_estimator.estimate_pose(prev_kp, kp, matches)

        if R is None or t is None:
            print(f"Frame {idx}: Pose estimation failed")
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            trajectory.append(current_pose.copy())  # Repeat last pose
            continue

        # Store raw VO translation for path-based scaling
        accumulated_vo_path.append(t.flatten())
        
        # Determine scale factor
        if base_scale is None and len(marker_poses) >= 2:
            # Calculate initial scale from first marker pair
            marker_distance = np.linalg.norm(marker_poses[1] - marker_poses[0])
            vo_distance = np.linalg.norm(t)  # Single step distance
            base_scale = marker_distance / vo_distance
            print(f"Initial base scale: {base_scale:.6f}")
        
        # Use base scale or default to 1.0
        current_scale = base_scale if base_scale is not None else 1.0
        
        # Apply variable motion scaling based on feature quality
        motion_quality = min(np.sum(mask) / 100.0, 1.5) if mask is not None else 1.0
        
        # Scale translation with quality-based variation
        t_scaled = current_scale * motion_quality * t.flatten()
        
        # Create relative transformation
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t_scaled
        
        # Update pose
        old_pos = current_pose[:3, 3].copy()
        current_pose = current_pose @ T_rel
        step_size = np.linalg.norm(current_pose[:3, 3] - old_pos)
        
        print(f"Frame {idx}: Quality={motion_quality:.3f}, Step size={step_size:.4f}")
        
        # Periodic drift correction using markers
        if len(marker_poses) > len(marker_indices) - len(marker_poses) + 1:
            # New marker detected - apply drift correction
            if len(marker_poses) >= 2:
                marker_pos = marker_poses[-1]
                current_pos = current_pose[:3, 3]
                drift = np.linalg.norm(current_pos - marker_pos)
                
                if drift > 0.1:  # Only correct significant drift
                    # Blend VO position with marker position
                    blend_factor = 0.5  # How much to trust marker vs VO
                    corrected_pos = (1 - blend_factor) * current_pos + blend_factor * marker_pos
                    current_pose[:3, 3] = corrected_pos
                    print(f"Frame {idx}: Drift corrected by {drift:.4f}m")
                    

        trajectory.append(current_pose.copy())

        # Store for next iteration
        prev_image = image
        prev_kp = kp
        prev_desc = desc

    print(f"VO completed! Processed {len(trajectory)} poses")
    
    # Simple visualization
    trajectory_array = np.array([pose[:3, 3] for pose in trajectory])
    marker_only_trajectory = np.array(marker_poses)
    

    # Plot comparison
    plt.figure(figsize=(12, 5))

    # Subplot 1: Full VO trajectory 
    plt.subplot(1, 2, 1)
    plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-o', markersize=2)
    plt.title('Full VO Trajectory (X-Y)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)

    # Subplot 2: Just marker positions (no VO interpolation)
    plt.subplot(1, 2, 2)
    plt.plot(marker_only_trajectory[:, 0], marker_only_trajectory[:, 1], 'r-o', markersize=4)
    plt.title('Marker Positions Only (X-Y)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    poses_data = {
    "x": [p[0] for p in trajectory_array],
    "y": [p[1] for p in trajectory_array], 
    "z": [p[2] for p in trajectory_array], 
    }

    df = pd.DataFrame(poses_data)

    df.to_csv("utils/vo_poses_aruco_every_2nd_frame.csv")

        
    # ground_truth_poses = ground_truth['poses']

    # left_positions = []
    # for ground_truth_pose in ground_truth_poses:
    #     left_trans = ground_truth_pose['left_camera']['translation']
    #     left_positions.append([left_trans[0], left_trans[1], left_trans[2]])

    # left_positions = np.array(left_positions)

    # position_errors = np.linalg.norm(trajectory_array - left_positions[:len(trajectory_array)], axis=1)

    # ate = np.sqrt(np.mean(position_errors**2))

    # print(f"Absolute Trajectory Error (ATE): {ate:.4f} meters")

    # # Error distribution analysis
    # plt.figure(figsize=(15, 10))

    # # Error over time
    # plt.subplot(2, 3, 1)
    # plt.plot(position_errors, 'b-', linewidth=1)
    # plt.scatter(marker_indices, position_errors[marker_indices], color='red', s=30, zorder=5)
    # plt.title('Position Error Over Time')
    # plt.xlabel('Frame')
    # plt.ylabel('Error (m)')
    # plt.grid(True)
    # plt.legend(['VO Error', 'Marker Frames'])

    # # Error histogram
    # plt.subplot(2, 3, 2)
    # plt.hist(position_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Error Distribution')
    # plt.xlabel('Error (m)')
    # plt.ylabel('Frequency')
    # plt.grid(True, alpha=0.3)

    # # Cumulative error
    # plt.subplot(2, 3, 3)
    # plt.plot(np.cumsum(position_errors), 'g-', linewidth=2)
    # plt.title('Cumulative Error')
    # plt.xlabel('Frame')
    # plt.ylabel('Cumulative Error (m)')
    # plt.grid(True)

    # # Error vs distance from start
    # distances_from_start = np.linalg.norm(trajectory_array - trajectory_array[0], axis=1)
    # plt.subplot(2, 3, 4)
    # plt.scatter(distances_from_start, position_errors, alpha=0.6, s=15)
    # plt.title('Error vs Distance from Start')
    # plt.xlabel('Distance from Start (m)')
    # plt.ylabel('Error (m)')
    # plt.grid(True)

    # # X,Y error components
    # plt.subplot(2, 3, 5)
    # error_components = trajectory_array - left_positions[:len(trajectory_array)]
    # plt.scatter(error_components[:, 0], error_components[:, 1], alpha=0.6, s=15)
    # plt.title('X-Y Error Components')
    # plt.xlabel('X Error (m)')
    # plt.ylabel('Y Error (m)')
    # plt.axis('equal')
    # plt.grid(True)

    # # Error statistics
    # plt.subplot(2, 3, 6)
    # stats_text = f"""Error Statistics:
    # Mean: {np.mean(position_errors):.3f}m
    # Median: {np.median(position_errors):.3f}m
    # Std: {np.std(position_errors):.3f}m
    # Max: {np.max(position_errors):.3f}m
    # 95th percentile: {np.percentile(position_errors, 95):.3f}m

    # Marker frames: {len(marker_indices)}
    # Error at markers: {np.mean(position_errors[marker_indices]):.3f}m"""

    # plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    # plt.axis('off')
    # plt.title('Error Statistics')

    # plt.tight_layout()
    # plt.show()



