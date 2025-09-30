import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.feature_extractor import SIFT

# --- Timing and metrics storage ---
frame_metrics = []

def record_frame(frame_num, t_extract, t_match, t_essential, t_pose, 
                 n_keypoints, n_matches, n_inliers, translation_mag):
    frame_metrics.append({
        'frame': frame_num,
        'time_extract': t_extract,
        'time_match': t_match,
        'time_essential': t_essential,
        'time_pose': t_pose,
        'time_total': t_extract + t_match + t_essential + t_pose,
        'n_keypoints': n_keypoints,
        'n_matches': n_matches,
        'n_inliers': n_inliers,
        'inlier_ratio': n_inliers / n_matches if n_matches > 0 else 0,
        'translation_mag': translation_mag
    })

# --- Load dataset info ---
with open('office_dataset_aruco/ground_truth_poses.json', 'r') as f:
    ground_truth = json.load(f)

camera_intrinsics = ground_truth["camera_intrinsics"]
K = np.array([[camera_intrinsics["fx"], 0, camera_intrinsics["cx"]],
              [0, camera_intrinsics["fy"], camera_intrinsics["cy"]],
              [0, 0, 1]], dtype=np.float32)

images_path = "office_dataset_aruco/left"
image_files = sorted([f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# --- Initialize feature extractor ---
extractor = SIFT(n_features=2000)

# --- ArUco setup ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
marker_size = ground_truth['aruco_markers']['markers'][0]['size']
dist_coeffs = np.zeros(5)

# --- Initialise pose from first detected marker ---
trajectory = []
current_pose = np.eye(4)
marker_positions = []
scale = 1.0
scale_computed = False

# Initialise with first marker
for idx, filename in enumerate(image_files[:5]):
    img = cv2.imread(os.path.join(images_path, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None and 0 in ids.flatten():
        id_idx = np.where(ids.flatten() == 0)[0][0]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[id_idx]], marker_size, K, dist_coeffs
        )
        R, _ = cv2.Rodrigues(rvecs[0])
        camera_pos = -R.T @ tvecs[0].flatten()
        current_pose[:3, 3] = camera_pos
        current_pose[:3, :3] = R.T
        trajectory.append(current_pose.copy())
        marker_positions.append(camera_pos)
        print(f"Initialised VO from ArUco at frame {idx}, pos={camera_pos}")
        break

# --- Process frames sequentially with PURE VO ---
prev_img = None
prev_kp, prev_desc = None, None

print("\nProcessing frames...")
for idx, filename in enumerate(image_files[:122]):
    img = cv2.imread(os.path.join(images_path, filename), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Time feature extraction
    t0 = time.perf_counter()
    kp, desc = extractor.detect_and_compute(img)
    t_extract = time.perf_counter() - t0

    if prev_img is None:
        prev_img, prev_kp, prev_desc = img, kp, desc
        record_frame(idx, t_extract, 0, 0, 0, len(kp), 0, 0, 0)
        continue

    # Time feature matching
    t0 = time.perf_counter()
    matches = extractor.match_features(prev_desc, desc)
    t_match = time.perf_counter() - t0

    if len(matches) < 100:
        trajectory.append(current_pose.copy())
        record_frame(idx, t_extract, t_match, 0, 0, len(kp), len(matches), 0, 0)
        continue

    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

    # Time essential matrix estimation
    t0 = time.perf_counter()
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    t_essential = time.perf_counter() - t0

    if E is None:
        trajectory.append(current_pose.copy())
        record_frame(idx, t_extract, t_match, t_essential, 0, len(kp), len(matches), 0, 0)
        continue

    n_inliers = np.sum(mask)

    # Time pose recovery
    t0 = time.perf_counter()
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    t_pose = time.perf_counter() - t0

    # Only use markers for scale computation from first two frames
    if not scale_computed and len(marker_positions) < 2:
        img_color = cv2.imread(os.path.join(images_path, filename))
        gray_current = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_current, aruco_dict, parameters=parameters)
        
        if ids is not None and 0 in ids.flatten():
            id_idx = np.where(ids.flatten() == 0)[0][0]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[id_idx]], marker_size, K, dist_coeffs
            )
            R_marker, _ = cv2.Rodrigues(rvecs[0])
            camera_pos_current = -R_marker.T @ tvecs[0].flatten()
            marker_positions.append(camera_pos_current)
            
            if len(marker_positions) == 2:
                real_dist = np.linalg.norm(marker_positions[1] - marker_positions[0])
                vo_dist = np.linalg.norm(t.flatten())
                if vo_dist > 1e-6:
                    scale = real_dist / vo_dist
                    scale_computed = True
                    print(f"Scale computed from first two markers: {scale:.3f}")

    R = R.T
    t = -t

    # Apply VO transformation with computed scale
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
    T_rel[:3, 3] = t.flatten() * scale

    current_pose = current_pose @ T_rel
    trajectory.append(current_pose.copy())

    translation_mag = np.linalg.norm(t.flatten() * scale)
    record_frame(idx, t_extract, t_match, t_essential, t_pose, 
                 len(kp), len(matches), n_inliers, translation_mag)

    prev_img, prev_kp, prev_desc = img, kp, desc

# --- Save metrics ---
df_metrics = pd.DataFrame(frame_metrics)
df_metrics.to_csv('results/laptop/performance_metrics.csv', index=False)

# --- Convert trajectory to numpy ---
traj = np.array([pose[:3, 3] for pose in trajectory])

# # --- Save trajectory ---
# df_traj = pd.DataFrame({
#     "frame": list(range(len(traj))),
#     "x": traj[:, 0],
#     "y": traj[:, 1],
#     "z": traj[:, 2],
# })
# df_traj.to_csv('results/pure_vo_2.csv', index=False)

# --- Print summary statistics ---
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

avg_total = df_metrics['time_total'].mean()
avg_extract = df_metrics['time_extract'].mean()
avg_match = df_metrics['time_match'].mean()
avg_essential = df_metrics['time_essential'].mean()
avg_pose = df_metrics['time_pose'].mean()

print(f"\nTiming (average per frame):")
print(f"  Feature extraction: {avg_extract*1000:.1f} ms ({avg_extract/avg_total*100:.1f}%)")
print(f"  Feature matching:   {avg_match*1000:.1f} ms ({avg_match/avg_total*100:.1f}%)")
print(f"  Essential matrix:   {avg_essential*1000:.1f} ms ({avg_essential/avg_total*100:.1f}%)")
print(f"  Pose recovery:      {avg_pose*1000:.1f} ms ({avg_pose/avg_total*100:.1f}%)")
print(f"  TOTAL:              {avg_total*1000:.1f} ms")
print(f"\nThroughput: {1/avg_total:.2f} fps")

print(f"\nFeature statistics:")
print(f"  Avg keypoints:   {df_metrics['n_keypoints'].mean():.0f}")
print(f"  Avg matches:     {df_metrics['n_matches'].mean():.0f}")
print(f"  Avg inliers:     {df_metrics['n_inliers'].mean():.0f}")
print(f"  Avg inlier ratio: {df_metrics['inlier_ratio'].mean():.2%}")

print(f"\nTrajectory:")
print(f"  Total frames:    {len(traj)}")
print(f"  Path length:     {np.sum(df_metrics['translation_mag']):.2f} m")
print(f"  Final position:  [{traj[-1, 0]:.2f}, {traj[-1, 1]:.2f}, {traj[-1, 2]:.2f}]")
print(f"  Scale factor:    {scale:.3f}")
print("="*60)

# --- Plot results ---
if len(traj) > 0:
    plt.figure(figsize=(15, 10))

    # XY trajectory
    plt.subplot(2, 3, 1)
    plt.plot(traj[:, 0], traj[:, 1], 'b-o', markersize=2, linewidth=1)
    plt.scatter(traj[0, 0], traj[0, 1], c='red', s=50, label="Start")
    plt.title("Pure VO Trajectory (X-Y)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis("equal"); plt.grid(True)
    plt.legend()

    # Distance from origin
    plt.subplot(2, 3, 2)
    distances = np.linalg.norm(traj - traj[0], axis=1)
    plt.plot(distances, 'r-', linewidth=2)
    plt.title("Distance from Start")
    plt.xlabel("Frame"); plt.ylabel("Distance (m)"); plt.grid(True)

    # Z trajectory (depth)
    plt.subplot(2, 3, 3)
    plt.plot(traj[:, 2], 'g-', linewidth=2)
    plt.title("Z (Depth) Trajectory")
    plt.xlabel("Frame"); plt.ylabel("Z (m)"); plt.grid(True)

    # Timing breakdown
    plt.subplot(2, 3, 4)
    plt.plot(df_metrics['time_extract']*1000, label='Extract', alpha=0.7)
    plt.plot(df_metrics['time_match']*1000, label='Match', alpha=0.7)
    plt.plot(df_metrics['time_essential']*1000, label='Essential', alpha=0.7)
    plt.plot(df_metrics['time_total']*1000, 'k-', label='Total', linewidth=2)
    plt.title("Processing Time per Frame")
    plt.xlabel("Frame"); plt.ylabel("Time (ms)"); plt.legend(); plt.grid(True)

    # Inlier ratio
    plt.subplot(2, 3, 5)
    plt.plot(df_metrics['inlier_ratio']*100, 'g-')
    plt.title("Inlier Ratio")
    plt.xlabel("Frame"); plt.ylabel("Inlier %"); plt.grid(True)
    plt.ylim([0, 100])

    # Number of matches
    plt.subplot(2, 3, 6)
    plt.plot(df_metrics['n_matches'], 'b-', label='Matches')
    plt.plot(df_metrics['n_inliers'], 'g-', label='Inliers')
    plt.title("Feature Matches")
    plt.xlabel("Frame"); plt.ylabel("Count"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/laptop/performance_analysis.png', dpi=150)
    plt.show()