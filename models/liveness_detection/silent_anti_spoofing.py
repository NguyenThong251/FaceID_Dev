import os
import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from functools import lru_cache

from models.anti_spoofs_control.anti_spoof_predict import AntiSpoofPredict
from models.anti_spoofs_control.generate_patches import CropImage
from models.anti_spoofs_control.utility import parse_model_name
from config.settings import SCORE_ANTI_SPOOFING_THRESHOLD

class SilentAntiSpoofing:
    def __init__(self, device_id: int = 0, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "anti_spoofs/anti_spoof_models")
        
        self.model_dir = model_dir
        self.model = AntiSpoofPredict()
        self.image_cropper = CropImage()
        
        # Constants for thresholds
        self.real_threshold = 0.85
        self.fake_threshold = 0.65
        self.final_confidence_threshold = SCORE_ANTI_SPOOFING_THRESHOLD
        
        # Cache model paths and parameters
        self.model_paths = []
        self.model_params = {}
        for model_name in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, model_name)
            h_input, w_input, _, scale = parse_model_name(model_name)
            self.model_paths.append((model_name, model_path))
            self.model_params[model_name] = {
                "h_input": h_input,
                "w_input": w_input,
                "scale": scale
            }

    @lru_cache(maxsize=32)
    def _preprocess_image(self, image: np.ndarray, target_ratio: float = 0.75) -> np.ndarray:
        """Cache preprocessed images for better performance"""
        height, width = image.shape[:2]
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.1:
            return image
            
        target_height = 80
        target_width = int(target_height * target_ratio)
        
        if current_ratio > target_ratio:
            interim_width = int(target_height * current_ratio)
            image = cv2.resize(image, (interim_width, target_height))
            start_x = (interim_width - target_width) // 2
            image = image[:, start_x:start_x+target_width]
        else:
            image = cv2.resize(image, (target_width, int(target_width / current_ratio)))
            start_y = (image.shape[0] - target_height) // 2
            image = image[start_y:start_y+target_height]
            
        return cv2.resize(image, (target_width, target_height))

    def detect(self, image: np.ndarray) -> Tuple[bool, float]:
        try:
            # Convert RGB to BGR once
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get face bbox
            image_bbox = self.model.get_bbox(image)
            if image_bbox is None:
                return False, 1.0

            # Initialize prediction array
            prediction = np.zeros((1, 3))
            
            # Process with cached model paths and parameters
            for model_name, model_path in self.model_paths:
                params = self.model_params[model_name]
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": params["scale"],
                    "out_w": params["w_input"],
                    "out_h": params["h_input"],
                    "crop": params["scale"] is not None
                }
                
                # Crop and predict
                img = self.image_cropper.crop(**param)
                prediction += self.model.predict(img, model_path)

            # Calculate scores efficiently
            softmax_scores = prediction[0] / prediction[0].sum()
            real_score = float(softmax_scores[1])
            
            if np.argmax(prediction) == 1:  # Real prediction
                fake_scores = (softmax_scores[0], softmax_scores[2])
                max_fake_score = max(fake_scores)
                
                # Check conditions efficiently
                is_real = (real_score > self.real_threshold and
                          real_score > max_fake_score * 1.5 and
                          max_fake_score < self.fake_threshold and
                          real_score > self.final_confidence_threshold)
                return is_real, real_score
            
            return False, float(max(softmax_scores[0], softmax_scores[2]))

        except Exception:
            return False, 1.0

    def detect_with_visualization(self, image: np.ndarray, save_path: Optional[str] = None) -> Tuple[bool, float, np.ndarray]:
        try:
            # Convert RGB to BGR once
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get detection result
            is_real, confidence = self.detect(image)
            
            # Get bbox for visualization
            image_bbox = self.model.get_bbox(image_bgr)
            if image_bbox is not None:
                # Set visualization parameters
                color = (255, 0, 0) if is_real else (0, 0, 255)
                result_text = f"{'Real' if is_real else 'Fake'} Face ({confidence*100:.1f}%)"
                font_scale = 0.5 * image_bgr.shape[0] / 1024
                
                # Draw rectangle and text
                cv2.rectangle(
                    image_bgr,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    color, 2)
                cv2.putText(
                    image_bgr,
                    result_text,
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, color)

            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, image_bgr)
                
            # Convert back to RGB for return
            return is_real, confidence, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        except Exception:
            return False, 1.0, image
