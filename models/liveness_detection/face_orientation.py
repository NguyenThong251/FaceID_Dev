import torch
import numpy as np

class FaceOrientationDetector():
    """This class detects the orientation of a face in an image."""
    def __init__(self):
        self.frontal_range = [25, 60]
        # Pre-calculate some values
        self.min_range = self.frontal_range[0]
        self.max_range = self.frontal_range[1]
    
    def calculate_angle(self, v1: (list, tuple, torch.Tensor, np.ndarray), v2: (list, tuple, torch.Tensor, np.ndarray)):
        '''Calculate the angle between 2 vectors v1 and v2'''
        # Convert to numpy array efficiently
        v1 = v1.numpy() if isinstance(v1, torch.Tensor) else np.asarray(v1)
        v2 = v2.numpy() if isinstance(v2, torch.Tensor) else np.asarray(v2)
            
        # Optimize calculation
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cosine = dot_product / norms
        
        # Ensure cosine is within valid range to avoid NaN
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.round(np.degrees(np.arccos(cosine)))

    def detect(self, landmarks):
        '''
        Detects the orientation of a face based on landmarks.
        Parameters:
            landmarks: A list/array of 5 points representing the positions on the face 
            [left eye, right eye, nose, left mouth, right mouth].
        Returns:
            str: Face orientation ('front', 'left', 'right', 'up', 'down')
        '''
        try:
            # Ensure landmarks is a numpy array
            if isinstance(landmarks, torch.Tensor):
                landmarks = landmarks.cpu().numpy()
            elif not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)

            # Convert landmarks to numpy arrays once
            left_eye = np.asarray(landmarks[0])
            right_eye = np.asarray(landmarks[1])
            nose = np.asarray(landmarks[2])
            
            # Calculate vectors
            left2right_eye = right_eye - left_eye
            lefteye2nose = nose - left_eye
            right2left_eye = left_eye - right_eye
            righteye2nose = nose - right_eye
            
            # Calculate angles
            left_angle = self.calculate_angle(left2right_eye, lefteye2nose)
            right_angle = self.calculate_angle(right2left_eye, righteye2nose)
            
            # Optimize orientation checks
            is_front = (self.min_range <= left_angle <= self.max_range and 
                       self.min_range <= right_angle <= self.max_range)
            if is_front:
                return 'front'
                
            if left_angle > self.max_range and right_angle > self.max_range:
                return 'down'
                
            if left_angle < self.min_range and right_angle < self.min_range:
                return 'up'
                
            return 'right' if left_angle < right_angle else 'left'
        except Exception as e:
            print(f"Error in face orientation detection: {str(e)}")
            return 'front'  # Default to front in case of error