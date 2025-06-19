import redis
import json
import numpy as np
from typing import Optional, Dict, List, Tuple
from flask import jsonify
import os
from utils.image_utils import save_image

class RedisService:
    # Redis Configuration
    REDIS_CONFIG = {
        'host': '127.0.0.1',
        'port': 6379,
        'db': 0,
        'decode_responses': True
    }
    REDIS_TEMP_EXPIRE = 300  # 5 minutes in seconds
    
    def __init__(self):
        self.client = redis.Redis(**self.REDIS_CONFIG)
        
    def get_error_response(self, error_code: str, details: str = None) -> Dict:
        response = {
            'success': False,
            'error': {
                'message': error_code
            }
        }
        if details:
            response['error']['details'] = details
        return response
        
    def store_temp_image(self, user_id: str, challenge: str, image_frame: str) -> None:
        """Store temporary face image during registration process"""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            temp_data = self.client.get(redis_key)
            temp_images = json.loads(temp_data) if temp_data else {}
            temp_images[challenge] = image_frame
            self.client.setex(redis_key, self.REDIS_TEMP_EXPIRE, json.dumps(temp_images))
            return True
        except Exception as e:
            return False

    def store_temp_gender(self, user_id: str, gender: str) -> None:
        """Store temporary gender information during registration"""
        try:
            redis_key = f"ERP:TempGenderInfo:{user_id}"
            self.client.setex(redis_key, self.REDIS_TEMP_EXPIRE, gender)
            return True
        except Exception as e:
            return False

    def get_temp_gender(self, user_id: str) -> None:
        """Get temporary gender information for a user"""
        try:
            redis_key = f"ERP:TempGenderInfo:{user_id}"
            gender = self.client.get(redis_key)
            if gender is None:
                return None
            return gender
        except Exception as e:
            return None

    def delete_temp_gender(self, user_id: str) -> None:
        """Delete temporary gender information after registration"""
        try:
            redis_key = f"ERP:TempGenderInfo:{user_id}"
            self.client.delete(redis_key)
            return True
        except Exception as e:
            return False

    def get_temp_images(self, user_id: str) -> Dict:
        """Get all temporary face images for a user"""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            temp_data = self.client.get(redis_key)
            if temp_data is None:
                return None
            return json.loads(temp_data)
        except Exception as e:
            return None

    def delete_temp_images(self, user_id: str) -> None:
        """Delete temporary face images after registration"""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            self.client.delete(redis_key)
            return True
        except Exception as e:
            return False
    def cache_face_features(self, user_id: str, features: str) -> None:
        """Cache face features for faster verification"""
        try:
            redis_key = f"ERP:FaceFeatures:{user_id}"
            self.client.set(redis_key, features) 
            return True
        except Exception as e:
            return False

    def get_cached_features(self, user_id: str) ->  Optional[str]:
        """Get cached face features"""
        try:
            redis_key = f"ERP:FaceFeatures:{user_id}"
            features = self.client.get(redis_key)
            if features is None:
                return None
            return features
        except Exception as e:
            return None

    def delete_cached_features(self, user_id: str) -> None:
        try:
            redis_key = f"ERP:FaceFeatures:{user_id}"
            self.client.delete(redis_key)
            return True
        except Exception as e:
            return False
    def delete_cached_userInfo(self, user_id: str) -> None:
        try:
            redis_key = f"ERP:Users:{user_id}"
            self.client.delete(redis_key)
            return True
        except Exception as e:
            return False
    # v2 vector face
    def get_all_face_features(self) -> List[Tuple[str, np.ndarray]]:
        """Lấy tất cả face features từ Redis"""
        try:
            # Lấy tất cả keys có pattern ERP:FaceFeatures:*
            pattern = "ERP:FaceFeatures:*"
            keys = self.client.keys(pattern)
            
            if not keys:
                return []

            face_data = []
            for key in keys:
                user_id = key.split(':')[-1]  # Lấy user_id từ key
                features = self.client.get(key)
                if features:
                    # Chuyển đổi features string thành numpy array
                    feature_array = np.array(json.loads(features)[0])
                    face_data.append((user_id, feature_array))
            
            return face_data
        except Exception as e:
            return []

    def get_all_users(self) -> list:
        """Lấy danh sách userId đã đăng ký (dựa vào Redis key ảnh)."""
        try:
            pattern = "ERP:FaceFeatures:*"
            keys = self.client.keys(pattern)
            return [key.split(":")[-1] for key in keys]
        except Exception:
            return []

    def get_user_images(self, user_id: str) -> list:
        """Lấy danh sách challenge đã lưu ảnh cho userId."""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            temp_data = self.client.get(redis_key)
            if temp_data:
                temp_images = json.loads(temp_data)
                return list(temp_images.keys())
            # Nếu không có trong Redis, kiểm tra file storage
            faces_dir = "storage/faces"
            if not os.path.exists(faces_dir):
                return []
            files = os.listdir(faces_dir)
            challenges = []
            for f in files:
                if f.startswith(f"{user_id}_") and f.endswith(".jpg"):
                    challenge = f[len(user_id)+1:-4]
                    challenges.append(challenge)
            return challenges
        except Exception:
            return []

    def get_image_path(self, user_id: str, challenge: str) -> str:
        """Trả về path ảnh dựa vào userId và challenge."""
        faces_dir = "storage/faces"
        image_path = os.path.join(faces_dir, f"{user_id}_{challenge}.jpg")
        if os.path.exists(image_path):
            return image_path
        return None

# Create singleton instance
redis_service = RedisService() 

# Trong process_image hoặc sau khi có frame:
image_path = f"storage/faces/{user_id}_{challenge}.jpg"
save_image(frame, image_path)
# Lưu image_path vào DB/Redis nếu cần 