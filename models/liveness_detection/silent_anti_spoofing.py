# import os
# import cv2
# import numpy as np
# import torch
# from typing import Tuple, Optional

# from models.anti_spoofs_control.anti_spoof_predict import AntiSpoofPredict
# from models.anti_spoofs_control.generate_patches import CropImage
# from models.anti_spoofs_control.utility import parse_model_name

# class SilentAntiSpoofing:
#     def __init__(self, device_id: int = 0, model_dir: str = None):
#         """
#         Initialize the Silent Anti Spoofing detector
#         Args:
#             device_id: GPU device id (default 0)
#             model_dir: Directory containing the model files
#         """
#         if model_dir is None:
#             model_dir = os.path.join(os.path.dirname(__file__), "anti_spoofs/anti_spoof_models")
        
#         self.model_dir = model_dir
#         self.model = AntiSpoofPredict(device_id)
#         self.image_cropper = CropImage()
        
#         # Thêm các ngưỡng kiểm tra
#         self.real_threshold = 0.85  # Ngưỡng cơ bản để xác định real face
#         self.fake_threshold = 0.65  # Ngưỡng để xác định fake face
#         self.final_confidence_threshold = 0.99  # Ngưỡng cuối cùng để xác nhận real face

#     def check_image(self, image: np.ndarray) -> bool:
#         """Check if image has correct aspect ratio (3:4)"""
#         height, width = image.shape[:2]
#         return abs(width/height - 3/4) < 0.1  # Allow some tolerance in aspect ratio

#     def preprocess_image(self, image: np.ndarray) -> np.ndarray:
#         """
#         Preprocess image to standard size and aspect ratio
#         Args:
#             image: RGB numpy array image
#         Returns:
#             Preprocessed image
#         """
#         height, width = image.shape[:2]
        
#         # First, resize to standard height while maintaining aspect ratio
#         target_height = 80  # Standard height
#         target_width = int(target_height * 3/4)  # Width for 3:4 ratio
        
#         # Resize with padding to maintain aspect ratio
#         current_ratio = width / height
#         target_ratio = 3/4
        
#         if current_ratio > target_ratio:
#             # Image is too wide, resize by height and crop width
#             interim_width = int(target_height * current_ratio)
#             interim_height = target_height
#             image = cv2.resize(image, (interim_width, interim_height))
            
#             # Crop center to achieve target ratio
#             start_x = (interim_width - target_width) // 2
#             image = image[:, start_x:start_x+target_width]
#         else:
#             # Image is too tall, resize by width and crop height
#             interim_width = target_width
#             interim_height = int(target_width / current_ratio)
#             image = cv2.resize(image, (interim_width, interim_height))
            
#             # Crop center to achieve target ratio
#             start_y = (interim_height - target_height) // 2
#             image = image[start_y:start_y+target_height, :]
        
#         # Final resize to ensure exact dimensions
#         image = cv2.resize(image, (target_width, target_height))
        
#         return image

#     def detect(self, image: np.ndarray) -> Tuple[bool, float]:
#         """
#         Detect if an image is real or fake
#         Args:
#             image: RGB numpy array image
#         Returns:
#             Tuple of (is_real, confidence_score)
#             is_real: True if real face (label == 1), False if fake
#             confidence_score: Score between 0 and 1
#                 For fake detection (is_real=False): Higher score means more confident it's fake
#                 For real detection (is_real=True): Higher score means more confident it's real
#         """
#         try:
#             # Convert RGB to BGR for OpenCV
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # Check image ratio
#             if not self.check_image(image):
#                 # Resize to 3:4 ratio if needed
#                 target_width = int(image.shape[0] * 3/4)
#                 image = cv2.resize(image, (target_width, image.shape[0]))

#             # Get face bbox
#             image_bbox = self.model.get_bbox(image)
#             if image_bbox is None:
#                 return False, 1.0

#             prediction = np.zeros((1, 3))
#             model_count = 0
            
#             # Process with each model in directory
#             for model_name in os.listdir(self.model_dir):
#                 h_input, w_input, model_type, scale = parse_model_name(model_name)
#                 param = {
#                     "org_img": image,
#                     "bbox": image_bbox,
#                     "scale": scale,
#                     "out_w": w_input,
#                     "out_h": h_input,
#                     "crop": True,
#                 }
#                 if scale is None:
#                     param["crop"] = False
                    
#                 img = self.image_cropper.crop(**param)
#                 model_prediction = self.model.predict(img, os.path.join(self.model_dir, model_name))
#                 prediction += model_prediction
#                 model_count += 1

#             # Get final prediction
#             label = np.argmax(prediction)
            
#             # Calculate confidence scores
#             softmax_scores = prediction[0] / prediction[0].sum()
            
#             # Kiểm tra chặt chẽ cho real face
#             is_real = False
#             if label == 1:  # Nếu model dự đoán là real
#                 real_score = float(softmax_scores[1])
#                 fake_photo_score = float(softmax_scores[0])
#                 fake_video_score = float(softmax_scores[2])
                
#                 # Kiểm tra các điều kiện cơ bản trước
#                 basic_check_passed = (
#                     real_score > self.real_threshold and  # Score real phải cao
#                     real_score > fake_photo_score * 1.5 and  # Score real phải cao hơn fake_photo nhiều
#                     real_score > fake_video_score * 1.5 and  # Score real phải cao hơn fake_video nhiều
#                     fake_photo_score < self.fake_threshold and  # Score fake_photo phải thấp
#                     fake_video_score < self.fake_threshold  # Score fake_video phải thấp
#                 )
                
#                 # Nếu pass điều kiện cơ bản, kiểm tra ngưỡng cuối cùng
#                 if basic_check_passed:
#                     is_real = real_score > self.final_confidence_threshold
                    
#                 confidence_score = real_score
                
#                 # Print debug info
#                 # print(f"\nLiveness Detection Results:")
#                 # print(f"Basic check passed: {basic_check_passed}")
#                 # print(f"Real score: {real_score:.4f}")
#                 # print(f"Final threshold check: {real_score > self.final_confidence_threshold}")
#                 # print(f"Final result: {'REAL' if is_real else 'FAKE'}")
                
#             else:
#                 # Nếu là fake, lấy max score của fake classes
#                 confidence_score = float(max(softmax_scores[0], softmax_scores[2]))
            
#             return is_real, confidence_score

#         except Exception as e:
#             return False, 1.0

#     def detect_with_visualization(self, image: np.ndarray, save_path: Optional[str] = None) -> Tuple[bool, float, np.ndarray]:
#         """
#         Detect with visualization of results
#         Args:
#             image: RGB numpy array image
#             save_path: Optional path to save visualization
#         Returns:
#             Tuple of (is_real, confidence_score, visualized_image)
#             confidence_score: Higher score means more confident in the prediction
#                 (whether it's real or fake)
#         """
#         try:
#             # Convert RGB to BGR
#             image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # Get detection result
#             is_real, confidence = self.detect(image)
            
#             # Draw result
#             image_bbox = self.model.get_bbox(image_bgr)
#             if image_bbox is not None:
#                 color = (255, 0, 0) if is_real else (0, 0, 255)
#                 result_text = f"Real Face ({confidence*100:.1f}%)" if is_real else f"Fake Face ({confidence*100:.1f}%)"
                
#                 cv2.rectangle(
#                     image_bgr,
#                     (image_bbox[0], image_bbox[1]),
#                     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
#                     color, 2)
#                 cv2.putText(
#                     image_bgr,
#                     result_text,
#                     (image_bbox[0], image_bbox[1] - 5),
#                     cv2.FONT_HERSHEY_COMPLEX, 0.5*image_bgr.shape[0]/1024, color)

#             # Save if path provided
#             if save_path:
#                 cv2.imwrite(save_path, image_bgr)
                
#             # Convert back to RGB for return
#             image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#             return is_real, confidence, image_rgb

#         except Exception as e:
#             return False, 1.0, image
import os
import cv2
import numpy as np
import torch
from typing import Tuple, Optional

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
        self.real_threshold = 0.85
        self.fake_threshold = 0.65
        self.final_confidence_threshold = SCORE_ANTI_SPOOFING_THRESHOLD
        
        # Cache model paths
        self.model_paths = [(model_name, os.path.join(self.model_dir, model_name)) 
                           for model_name in os.listdir(self.model_dir)]

    def preprocess_image(self, image: np.ndarray, target_ratio: float = 0.75) -> np.ndarray:
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
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get face bbox
            image_bbox = self.model.get_bbox(image)
            if image_bbox is None:
                return False, 1.0

            prediction = np.zeros((1, 3))
            
            # Process with cached model paths
            for model_name, model_path in self.model_paths:
                h_input, w_input, _, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": scale is not None
                }
                
                img = self.image_cropper.crop(**param)
                prediction += self.model.predict(img, model_path)

            # Calculate scores
            softmax_scores = prediction[0] / prediction[0].sum()
            real_score = float(softmax_scores[1])
            
            if np.argmax(prediction) == 1:  # Real prediction
                fake_scores = (softmax_scores[0], softmax_scores[2])
                is_real = (real_score > self.real_threshold and
                          real_score > max(fake_scores) * 1.5 and
                          max(fake_scores) < self.fake_threshold and
                          real_score > self.final_confidence_threshold)
                return is_real, real_score
            
            return False, float(max(softmax_scores[0], softmax_scores[2]))

        except Exception:
            return False, 1.0

    def detect_with_visualization(self, image: np.ndarray, save_path: Optional[str] = None) -> Tuple[bool, float, np.ndarray]:
        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            is_real, confidence = self.detect(image)
            
            image_bbox = self.model.get_bbox(image_bgr)
            if image_bbox is not None:
                color = (255, 0, 0) if is_real else (0, 0, 255)
                result_text = f"{'Real' if is_real else 'Fake'} Face ({confidence*100:.1f}%)"
                
                cv2.rectangle(
                    image_bgr,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    color, 2)
                cv2.putText(
                    image_bgr,
                    result_text,
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5*image_bgr.shape[0]/1024, color)

            if save_path:
                cv2.imwrite(save_path, image_bgr)
                
            return is_real, confidence, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        except Exception:
            return False, 1.0, image 

