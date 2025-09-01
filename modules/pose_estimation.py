import cv2
import numpy as np
from modules.feature_extractor import SIFT, ORB, BRISK

class MotionEstimator:
    def __init__(self, camera_matrix, method='essential'):
        self.K = camera_matrix
        self.method = method
        
    def estimate_pose(self, kp1, kp2, matches):
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        if len(pts1) < 8:
            return None, None, None
            
        if self.method == 'essential':
            return self._estimate_with_essential(pts1, pts2)
        elif self.method == 'fundamental':
            return self._estimate_with_fundamental(pts1, pts2)
    
    def _estimate_with_essential(self, pts1, pts2):
        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        if E is None:
            return None, None, None
            
        # Recover pose from Essential matrix
        points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        
        return R, t, mask & mask_pose.ravel().astype(bool)
    
    def _estimate_with_fundamental(self, pts1, pts2):
        # For uncalibrated case (though you have Blender intrinsics)
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, 
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            confidence=0.99
        )
        
        # Convert F to E if camera matrix is known
        E = self.K.T @ F @ self.K
        points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        
        return R, t, mask & mask_pose.ravel().astype(bool)