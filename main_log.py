import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.feature_extractor import SIFT

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

from modules.diagnostics import VODiagnostics

diagnostics = VODiagnostics(K)

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

for idx, filename in enumerate(image_files[:122]):
    img = cv2.imread(os.path.join(images_path, filename), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    kp, desc = extractor.detect_and_compute(img)
    if prev_img is None:
        prev_img, prev_kp, prev_desc = img, kp, desc
        continue

    matches = extractor.match_features(prev_desc, desc)
    if len(matches) < 100:
        trajectory.append(current_pose.copy())
        continue

    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        trajectory.append(current_pose.copy())
        continue

    # Check essential matrix quality
    if idx < 25:
    #if idx > 1 and idx < 122:
        inliers = np.sum(mask)
        print(f"Frame {idx}: Essential matrix - {inliers}/{len(matches)} inliers ({inliers/len(matches)*100:.1f}%)")

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    diagnostics.analyze_matches(idx, prev_kp, kp, matches, mask_pose, R, t)


    # # DEBUG: Print translation vectors
    # if idx < 25:
    #     print(f"Frame {idx}: VO translation = {t.flatten()}")

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
                # Compute scale from first two marker observations
                real_dist = np.linalg.norm(marker_positions[1] - marker_positions[0])
                vo_dist = np.linalg.norm(t.flatten())
                if vo_dist > 1e-6:
                    scale = real_dist / vo_dist
                    scale_computed = True
                    print(f"Scale computed from first two markers: {scale:.3f}")
                    print(f"Real distance: {real_dist:.3f}, VO distance: {vo_dist:.3f}")
                    print(f"STOPPING marker detection at frame {idx} - pure VO from here on")

     # Option A: Invert the rotation
    R = R.T
    t = -t

    # Apply VO transformation with computed scale
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
    T_rel[:3, 3] = t.flatten() * scale

    current_pose = current_pose @ T_rel
    trajectory.append(current_pose.copy())

    prev_img, prev_kp, prev_desc = img, kp, desc

# --- Convert trajectory to numpy ---
traj = np.array([pose[:3, 3] for pose in trajectory])

# --- Plot results ---
if len(traj) > 0:
    plt.figure(figsize=(15, 5))

    # XY trajectory
    plt.subplot(1, 3, 1)
    plt.plot(traj[:, 0], traj[:, 1], 'b-o', markersize=2, linewidth=1)
    plt.scatter(traj[0, 0], traj[0, 1], c='red', s=50, label="Start")
    # if len(marker_positions) > 1:
    #     marker_pos = np.array(marker_positions)
    #     plt.scatter(marker_pos[:, 0], marker_pos[:, 1], c='green', s=50, marker='s', label="Scale Markers")
    plt.title("Pure VO Trajectory (X-Y)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis("equal"); plt.grid(True)
    plt.legend()

    # Distance from origin
    plt.subplot(1, 3, 2)
    distances = np.linalg.norm(traj - traj[0], axis=1)
    plt.plot(distances, 'r-', linewidth=2)
    plt.title("Distance from Start")
    plt.xlabel("Frame"); plt.ylabel("Distance (m)"); plt.grid(True)

    # Z trajectory (depth)
    plt.subplot(1, 3, 3)
    plt.plot(traj[:, 2], 'g-', linewidth=2)
    plt.title("Z (Depth) Trajectory")
    plt.xlabel("Frame"); plt.ylabel("Z (m)"); plt.grid(True)

    plt.tight_layout(); plt.show()
else:
    print("No trajectory generated")

print(f"\nFinal scale factor used: {scale:.3f}")
print(f"Total trajectory length: {len(traj)} frames")
if len(traj) > 10:
    print(f"Final position: [{traj[-1, 0]:.2f}, {traj[-1, 1]:.2f}, {traj[-1, 2]:.2f}]")


# # Create dataframe
# pure_vo = {
#     "Frame number":list(range(len(traj))),
#     "x": traj[:, 0],
#     "y": traj[:, 1],
#     "z": traj[:, 2],
# }

# df = pd.DataFrame(pure_vo)

# df.to_csv('results/pure_vo_2.csv')

diag_df = pd.DataFrame(diagnostics.get_logs())
diag_df.to_csv("results/vo_diagnostics.csv", index=False)

