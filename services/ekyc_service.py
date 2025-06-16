import torch
import numpy as np
import json
from typing import List, Dict

from .redis_service import redis_service
from .database_service import db_service
from .detect_mask_service import mask_detection_service
from models.liveness_detection.silent_anti_spoofing import SilentAntiSpoofing
from utils.image_utils import batch_preprocess_images, base64_to_rgb_image
from models import (
    MTCNN,
    VGGFace2,
    FaceOrientationDetector
)
from config.settings import (
    FACE_MATCH_THRESHOLD,
    VALID_CHALLENGES
)
from config.model_config import device

class eKYC_Service:
    def __init__(self):
        # Initialize models with caching
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize and cache models"""
        self.device = device
        self.mtcnn = MTCNN(
            device=self.device,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            select_largest=True
        )
        self.verification_model = VGGFace2.load_model(device=self.device)
        self.anti_spoofing = SilentAntiSpoofing(device_id=0 if torch.cuda.is_available() else -1)
        self.orientation_detector = FaceOrientationDetector()

    def check_liveness(self, frame, challenge, user_id):
        """Optimized liveness check with mask detection"""
        try:
            if not user_id:
                return {'success': False, 'error': {'message': 'USER_ID_REQUIRED'}}

            # Convert base64 to image if needed
            if isinstance(frame, str):
                try:
                    frame = base64_to_rgb_image(frame, resize=True)
                except Exception:
                    return {'success': False, 'error': {'message': 'INVALID_IMAGE'}}

            # Check for mask
            mask_result = mask_detection_service.detect_mask(frame)
            if not mask_result['success']:
                return {'success': False, 'error': {'message': 'MASK_CHECK_FAILED'}}
            
            if mask_result['has_mask']:
                return {'success': False, 'error': {'message': 'MASK_DETECTED'}}

            # Anti-spoofing check
            is_real, spoof_score = self.anti_spoofing.detect(frame)
            if not is_real:
                return {'success': False, 'error': {'message': 'SPOOF_DETECTED'}}

            # Challenge validation
            if challenge and challenge not in VALID_CHALLENGES:
                return {'success': False, 'error': {'message': 'INVALID_CHALLENGE'}}

            # Face detection
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
            if boxes is None or len(boxes) == 0:
                return {'success': False, 'error': {'message': 'NO_FACE_DETECTED'}}

            # For verification mode (no challenge)
            if not challenge:
                return self._handle_verification(frame, boxes, user_id)

            # For challenge mode
            orientation = self.orientation_detector.detect(landmarks[0])
            is_challenge_passed = orientation == challenge
            
            if not is_challenge_passed:
                return {
                    'success': False,
                    'error': {
                        'message': 'CHALLENGE_FAILED',
                        'detected_orientation': str(orientation),
                        'expected_orientation': str(challenge)
                    }
                }
            
            return {'success': True, 'result': {'message': 'OK'}}

        except Exception as e:
            return {'success': False, 'error': {'message': 'SYSTEM_ERROR'}}

    def verify_registration_faces(self, frames):
        """Optimized face verification with mask detection"""
        try:
            if len(frames) != 3:
                return {'success': False, 'error': {'message': 'INVALID_IMAGE_COUNT'}}

            features = []
            for frame in frames:
                # Check for mask
                mask_result = mask_detection_service.detect_mask(frame)
                if not mask_result['success'] or mask_result['has_mask']:
                    return {'success': False, 'error': {'message': 'MASK_DETECTED'}}

                # Anti-spoofing check
                is_real, _ = self.anti_spoofing.detect(frame)
                if not is_real:
                    return {'success': False, 'error': {'message': 'SPOOF_DETECTED'}}

                # Extract features
                feature = self.extract_face_features(frame)
                if feature is None:
                    return {'success': False, 'error': {'message': 'FEATURE_EXTRACTION_FAILED'}}
                    
                features.append(feature / (np.linalg.norm(feature) + 1e-10))

            # Compare all pairs
            for i in range(2):
                similarity = np.dot(features[i], features[i+1])
                if similarity < FACE_MATCH_THRESHOLD:
                    return {
                        'success': False,
                        'error': {'message': 'FACE_MISMATCH'}
                    }

            return {'success': True, 'result': {'message': 'OK'}}

        except Exception as e:
            return {'success': False, 'error': {'message': 'SYSTEM_ERROR'}}

    def extract_face_features(self, frame):
        """Optimized feature extraction"""
        try:
            # Detect faces
            boxes, probs, _ = self.mtcnn.detect(frame, landmarks=True)
            if boxes is None or len(boxes) == 0:
                return None

            # Select best face
            box = boxes[np.argmax(probs)] if len(boxes) > 1 else boxes[0]
            
            # Preprocess face
            frame_tensor = torch.from_numpy(
                batch_preprocess_images([frame])[0]
            ).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = self.verification_model(frame_tensor)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                
            return features.cpu().numpy()[0]
            
        except Exception as e:
            return None

    def _handle_verification(self, frame: np.ndarray, boxes: np.ndarray, user_id: str) -> Dict:
        face_features = self._get_stored_features(user_id)
        if not face_features: 
            return {'success': False, 'error': {'message': 'USER_NOT_FOUND'}}
        
        face_frame_feature = self.extract_face_features(frame)
        if face_frame_feature is None: 
            return {'success': False, 'error': {'message': 'FEATURE_EXTRACTION_FAILED'}}
        
        verification_result = self.match_face_features(face_frame_feature, face_features)
        
        if verification_result.get('verified', False):
            return {
                'success': True,
                'result': {
                    'message': 'OK'
                }
            }
        
        return {
            'success': False,
            'error': {
                'message': 'FACE_NOT_MATCH'
            }
        }

    def _get_stored_features(self, user_id: str) -> List[np.ndarray]:
        try:
            data = redis_service.get_cached_features(user_id)
            if not data:
                data = db_service.get_stored_features(user_id)
                if data:
                    redis_service.cache_face_features(user_id, data)
            if data:
                features = json.loads(data)
                return [np.array(f) for f in features] if isinstance(features, list) else []
            return []
            
        except Exception as e:
            raise RuntimeError(f"Failed to get stored features: {str(e)}")

    def match_face_features(self, face_frame_feature: np.ndarray, face_features: List[np.ndarray]) -> Dict:
        try:
            if not face_features or face_frame_feature is None: 
                return {"verified": False, "error": "INVALID_FEATURES"}
            
            face_frame_feature = face_frame_feature / (np.linalg.norm(face_frame_feature) + 1e-10)
            normalized_features = [f / (np.linalg.norm(f) + 1e-10) for f in face_features if np.linalg.norm(f) > 1e-10]
            
            if not normalized_features: 
                return {"verified": False, "error": "INVALID_STORED_FEATURES"}
            
            similarities = np.dot(np.vstack(normalized_features), face_frame_feature)
            max_sim = np.max(similarities)
            
            return {
                "verified": max_sim > FACE_MATCH_THRESHOLD,
                "confidence": max_sim
            }
            
        except Exception:
            return {"verified": False, "error": "MATCHING_ERROR"}

    def search_face(self, frame: np.ndarray, top_k: int = 5) -> Dict:
        try:
            # Check for mask
            mask_result = mask_detection_service.detect_mask(frame)
            if not mask_result['success'] or mask_result['has_mask']:
                return {'success': False, 'error': {'message': 'MASK_DETECTED'}}

            # Anti-spoofing check
            is_real, _ = self.anti_spoofing.detect(frame)
            if not is_real:
                return {'success': False, 'error': {'message': 'SPOOF_DETECTED'}}
            
            # Extract features
            face_feature = self.extract_face_features(frame)
            if face_feature is None:
                return {'success': False, 'error': {'message': 'FEATURE_EXTRACTION_FAILED'}}
            
            # Search similar faces
            similar_faces = db_service.search_similar_faces(face_feature, k=top_k)
            if not similar_faces:
                return {'success': False, 'error': {'message': 'NO_MATCH_FOUND'}}
            
            return {'success': True, 'results': [{'user_id': user_id} for user_id, _ in similar_faces]}
            
        except Exception:
            return {'success': False, 'error': {'message': 'SYSTEM_ERROR'}}

ekyc_service = eKYC_Service() 
