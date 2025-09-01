import numpy as np
from modules.feature_extractor import SIFT, ORB, BRISK
from modules.frame_loader import FrameLoader
from modules.triangulation import Triangulator
import matplotlib.pyplot as plt
from modules.pose_estimation import MotionEstimator
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

        # FIX: matches is a list, not a tuple
        if len(matches) < 50:
            print(f"Frame {idx}: Not enough matches ({len(matches)}), skipping")
            continue

        print(f"Frame {idx}: Found {len(matches)} matches")

        # Estimate motion
        R, t, mask = motion_estimator.estimate_pose(prev_kp, kp, matches)

        if R is None or t is None:
            print(f"Frame {idx}: Pose estimation failed")
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

        # Estimate scale
        scale = triangulator.get_scale(prev_points_3d, points_3d, R, t)
        t_scaled = t * scale

        print(f"Frame {idx}: Scale = {scale:.3f}, Translation = {np.linalg.norm(t_scaled):.3f}")

        # Update pose (camera moves in world)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t_scaled.ravel()
        
        current_pose = current_pose @ T_rel
        trajectory.append(current_pose.copy())

        # Store for next iteration
        prev_image = image
        prev_kp = kp
        prev_desc = desc
        prev_points_3d = points_3d

    print(f"VO completed! Processed {len(trajectory)} poses")
    
    # Simple visualization
    trajectory_array = np.array([pose[:3, 3] for pose in trajectory])
    
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_array[:, 0], trajectory_array[:, 2], 'b-o', label='Estimated Trajectory', markersize=4)
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('Monocular VO Trajectory (Top View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Print trajectory positions
    print("\nTrajectory positions:")
    for i, pose in enumerate(trajectory):
        pos = pose[:3, 3]
        print(f"Frame {i}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")