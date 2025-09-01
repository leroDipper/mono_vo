import cv2
import numpy as np

class Triangulator:
    def __init__(self, camera_matrix):
        self.K = camera_matrix
        
    def triangulate_points(self, pts1, pts2, R1, t1, R2, t2):
        # Projection matrices
        P1 = self.K @ np.hstack([R1, t1.reshape(-1, 1)])
        P2 = self.K @ np.hstack([R2, t2.reshape(-1, 1)])
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous
        
        return points_3d.T
    
    def get_scale(self, points_3d_prev, points_3d_curr, R, t):
        # Simple scale estimation from triangulated points
        if points_3d_prev is None or len(points_3d_prev) < 10:
            return 1.0
            
        # Transform previous points to current frame
        points_3d_prev_transformed = (R @ points_3d_prev.T).T + t.T
        
        # Compute scale as median ratio of distances
        dists_curr = np.linalg.norm(points_3d_curr, axis=1)
        dists_prev = np.linalg.norm(points_3d_prev_transformed, axis=1)
        
        valid = (dists_curr > 0) & (dists_prev > 0)
        if np.sum(valid) < 5:
            return 1.0
            
        scale_ratios = dists_prev[valid] / dists_curr[valid]
        return np.median(scale_ratios)