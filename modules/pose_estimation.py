import cv2
import numpy as np

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
        E, mask_essential = cv2.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        if E is None:
            return None, None, None
        
        # Convert mask to proper boolean array
        mask_essential = mask_essential.ravel().astype(bool)
        
        # Filter points with essential matrix inliers
        pts1_filtered = pts1[mask_essential]
        pts2_filtered = pts2[mask_essential]
        
        # Recover pose from Essential matrix
        points, R, t, mask_pose = cv2.recoverPose(E, pts1_filtered, pts2_filtered, self.K)
        
        # Create final mask that maps back to original matches
        final_mask = np.zeros(len(pts1), dtype=bool)
        essential_indices = np.where(mask_essential)[0]
        
        # mask_pose corresponds to the filtered points
        mask_pose = mask_pose.ravel().astype(bool)
        final_indices = essential_indices[mask_pose]
        final_mask[final_indices] = True
        
        return R, t, final_mask
    
    def _estimate_with_fundamental(self, pts1, pts2):
        # For uncalibrated case
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, 
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            confidence=0.99
        )
        
        if F is None:
            return None, None, None
        
        # Convert F to E if camera matrix is known
        E = self.K.T @ F @ self.K
        mask = mask.ravel().astype(bool)
        
        # Filter points
        pts1_filtered = pts1[mask]
        pts2_filtered = pts2[mask]
        
        points, R, t, mask_pose = cv2.recoverPose(E, pts1_filtered, pts2_filtered, self.K)
        
        # Create final mask
        final_mask = np.zeros(len(pts1), dtype=bool)
        fundamental_indices = np.where(mask)[0]
        mask_pose = mask_pose.ravel().astype(bool)
        final_indices = fundamental_indices[mask_pose]
        final_mask[final_indices] = True
        
        return R, t, final_mask