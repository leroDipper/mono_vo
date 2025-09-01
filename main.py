import numpy as np
from modules.feature_extractor import SIFT, ORB, BRISK
from modules.triangulation import triangulate, triangulate_unfiltered
from modules.frame_loader import FrameLoader
from modules.triangulation import Triangulator
import matplotlib.pyplot as plt
from modules.pose_estimation import MotionEstimator
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json


if __name__ == "__main__":
    # === Load and Prepare Ground Truth Data ===
    
    with open('office_stereo_dataset/ground_truth_poses.json', 'r') as file:
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

    max_images = 10

    K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]])
    
    
    # Previous frame data
    prev_image = None
    prev_kp = None
    prev_desc = None
    prev_points_3d = None



    extractor_SIFT = SIFT(n_features=1500)
    load_frames = FrameLoader(images_path=images, max_images=max_images)
    motion_estimator = MotionEstimator(camera_matrix=K, method='essential')
    triangulator = Triangulator(camera_matrix=K)
    
    

    for idx, (image) in enumerate(load_frames.get_frames()):

        kp, desc = extractor_SIFT.detect_and_compute(image)

        if prev_image is None:
            prev_image = image
            prev_kp = kp
            prev_desc = desc
            continue

        # Match features between the current and previous frame
        matches = extractor_SIFT.match_features(prev_desc, desc)

        if len(matches[0]) < 50:
            print(f"Not enough matches!!")


        #estimate motion
        R, t, mask = motion_estimator.estimate_pose(prev_kp, kp, matches)

        if R is None or t is None:
            print("Pose estimation failed.")
            continue

       
        # Filter matches with inlier mask
        good_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])


        I = np.eye(3)
        O = np.zeros((3, 1))

        points_3d = triangulator.triangulate_points(pts1, pts2, I, O, R, t)

        scale = triangulator.get_scale(prev_points_3d, points_3d, R, t)

        t_scaled = t * scale

        # Triangulate points




        

        














