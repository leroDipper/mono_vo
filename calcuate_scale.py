import json
import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os

# ----------------------------
# Step 1: Load cameras.sfm
# ----------------------------
def load_cameras_sfm(sfm_path):
    with open(sfm_path, "r") as f:
        sfm = json.load(f)

    intrinsics = {}
    for cam in sfm["intrinsics"]:
        intrinsics[cam["intrinsicId"]] = cam

    cameras = {}
    for view in sfm["views"]:
        pose_id = view["poseId"]
        intr = intrinsics[view["intrinsicId"]]
        cameras[view["path"].split("/")[-1]] = {
            "poseId": pose_id,
            "intrinsics": intr,
        }

    poses = {}
    for pose in sfm["poses"]:
        # pose: rotation matrix Rcw, center C (camera center in world coords)
        # Convert to numpy arrays with float64 dtype
        R = np.array(pose["pose"]["transform"]["rotation"], dtype=np.float64).reshape(3, 3)
        C = np.array(pose["pose"]["transform"]["center"], dtype=np.float64)
        t = -R @ C
        poses[pose["poseId"]] = (R, t)

    return cameras, poses


# ----------------------------
# Step 2: Build projection matrix
# ----------------------------
def build_projection_matrix(intr, R, t):
    # Convert Meshroom focal length from mm to pixels
    focal_mm = float(intr["focalLength"])
    sensor_w = float(intr["sensorWidth"])
    sensor_h = float(intr["sensorHeight"])
    width = float(intr["width"])
    height = float(intr["height"])
    
    fx = (focal_mm / sensor_w) * width
    fy = (focal_mm / sensor_h) * height
    cx = float(intr["principalPoint"][0])
    cy = float(intr["principalPoint"][1])
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    Rt = np.hstack((R, t.reshape(-1, 1)))
    return K @ Rt


# ----------------------------
# Step 3: Triangulation
# ----------------------------
def triangulate_point(pt1, P1, pt2, P2):
    # Ensure points are float arrays
    pt1 = np.array(pt1, dtype=np.float64)
    pt2 = np.array(pt2, dtype=np.float64)
    
    pts4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
    pts3d = pts4d[:3] / pts4d[3]
    return pts3d.ravel()


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    sfm_path = "StructureFromMotion/output/cameras.sfm"

    cameras, poses = load_cameras_sfm(sfm_path)

    # ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    image_paths = sorted(glob.glob("office_dataset_aruco/left/*.png"))

    detections = {}
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(corners) > 0:
            detections[os.path.basename(path)] = corners[0][0]  # first marker only

    if len(detections) < 2:
        print(f"Error: Need at least 2 images with ArUco detections. Found: {len(detections)}")
        print(f"Images with detections: {list(detections.keys())}")
        return

    # Pick two images that see the marker
    img1, img2 = list(detections.keys())[:2]
    pts1 = detections[img1]
    pts2 = detections[img2]

    print(f"Using images: {img1} and {img2}")

    # Use first corner (top-left of ArUco)
    pt1 = pts1[0]
    pt2 = pts2[0]

    # Get projection matrices
    cam1 = cameras[img1]
    cam2 = cameras[img2]

    R1, t1 = poses[cam1["poseId"]]
    R2, t2 = poses[cam2["poseId"]]

    P1 = build_projection_matrix(cam1["intrinsics"], R1, t1)
    P2 = build_projection_matrix(cam2["intrinsics"], R2, t2)

    # Triangulate one edge (corner 0 and corner 1)
    tri_corners = []
    for i in range(2):  # corners 0 and 1
        pt1 = pts1[i]
        pt2 = pts2[i]
        X = triangulate_point(pt1, P1, pt2, P2)
        tri_corners.append(X)

    tri_corners = np.array(tri_corners)

    # Compute scale factor
    dist = np.linalg.norm(tri_corners[0] - tri_corners[1])
    marker_size = 0.30  # Blender marker size in meters
    scale = marker_size / dist

    print(f"Measured distance: {dist:.4f}")
    print(f"Scale factor: {scale:.4f}")


if __name__ == "__main__":
    main()