import numpy as np
from modules.feature_extractor import SIFT, ORB, BRISK
from modules.frame_loader import FrameLoader
from modules.triangulation import Triangulator
import matplotlib.pyplot as plt
from modules.pose_estimation2 import MotionEstimator
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json


if __name__ == "__main__":
    # === Load and Prepare Ground Truth Data ===
    
    with open('maximum_feature_office_dataset/ground_truth_poses.json', 'r') as file:
        ground_truth = json.load(file)

    # --- Paths to Blender dataset images ---
    images_path = "maximum_feature_office_dataset/left"
    
    images = [images_path]

    # Access camera intrinsics
    camera_intrinsics = ground_truth['camera_intrinsics']
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    resolution = tuple(camera_intrinsics['resolution'])

    max_images = 122

    K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]])
    
    # Initialize VO state
    current_pose = np.eye(4)
    trajectory = [current_pose.copy()]
    
    # Previous frame data
    prev_image = None
    prev_kp = None
    prev_desc = None
    prev_points_3d = None

    # Initialize components
    extractor_SIFT = SIFT(n_features=1500)
    load_frames = FrameLoader(images_path=images, max_images=max_images)
    motion_estimator = MotionEstimator(camera_matrix=K, method='essential')
    triangulator = Triangulator(camera_matrix=K)
    
    print(f"Processing {max_images} frames...")
    
    for idx, image in enumerate(load_frames.get_frames()):
        print(f"Processing frame {idx}")
        
        # Extract features
        kp, desc = extractor_SIFT.detect_and_compute(image)
        
        if prev_image is None:
            # First frame - initialize
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            print(f"Frame {idx}: Initialized with {len(kp)} features")
            continue

        # Match features between the current and previous frame
        matches = extractor_SIFT.match_features(prev_desc, desc)

        if len(matches) < 50:
            print(f"Frame {idx}: Not enough matches ({len(matches)}), skipping")
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            trajectory.append(current_pose.copy())  # Repeat last pose
            continue

        print(f"Frame {idx}: Found {len(matches)} matches")

        # Estimate motion
        R, t, mask = motion_estimator.estimate_pose(prev_kp, kp, matches)

        if R is None or t is None:
            print(f"Frame {idx}: Pose estimation failed")
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            trajectory.append(current_pose.copy())  # Repeat last pose
            continue

        # Filter matches with inlier mask
        good_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

        print(f"Frame {idx}: {len(good_matches)} inlier matches")

        # Triangulate points
        I = np.eye(3)
        O = np.zeros((3, 1))
        points_3d = triangulator.triangulate_points(pts1, pts2, I, O, R, t)

        # Check if this is the first triangulation
        if prev_points_3d is None:
            print(f"Frame {idx}: First triangulation, storing reference")
            prev_points_3d = points_3d
            # Don't update pose on first triangulation
            trajectory.append(current_pose.copy())
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            continue

        # FIXED: Use unit scale for pure geometric testing
        scale = 1.0  # No scale estimation - test pure motion geometry
        t_scaled = t.flatten()  # Use raw translation vector

        print(f"Frame {idx}: Fixed scale = {scale:.3f}, Translation = {np.linalg.norm(t_scaled):.3f}")


        # Just test the basic pose update without coordinate tricks
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        current_pose = current_pose @ T_rel

        # # Update pose (camera moves in world)
        # T_rel = np.eye(4)
        # T_rel[:3, :3] = R
        # T_rel[:3, 3] = t_scaled  # Already flattened
        
        # #current_pose = current_pose @ T_rel
        # #current_pose = T_rel @ current_pose
        # T_rel_inv = np.linalg.inv(T_rel)
        # current_pose = current_pose @ T_rel_inv

        trajectory.append(current_pose.copy())
















        # Store for next iteration
        prev_image = image
        prev_kp = kp
        prev_desc = desc
        prev_points_3d = points_3d

    print(f"VO completed! Processed {len(trajectory)} poses")
    
    # Simple visualization
    trajectory_array = np.array([pose[:3, 3] for pose in trajectory])
    
    # Plot both X-Y and X-Z views to see trajectory shape
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-o', label='Estimated Trajectory', markersize=3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('VO Trajectory (X-Y View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.plot(trajectory_array[:, 0], trajectory_array[:, 2], 'r-o', label='Estimated Trajectory', markersize=3)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('VO Trajectory (X-Z View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 3, 3)
    plt.plot(trajectory_array[:, 1], trajectory_array[:, 2], 'g-o', label='Estimated Trajectory', markersize=3)
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('VO Trajectory (Y-Z View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
