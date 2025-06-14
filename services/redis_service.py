import redis
import json
import numpy as np
from typing import Optional, Dict, List, Tuple
from flask import jsonify

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
        
    def store_temp_image(self, user_id: str, challenge: str, image_frame: str) -> Dict:
        """Store temporary face image during registration process"""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            temp_data = self.client.get(redis_key)
            temp_images = json.loads(temp_data) if temp_data else {}
            temp_images[challenge] = image_frame
            self.client.setex(redis_key, self.REDIS_TEMP_EXPIRE, json.dumps(temp_images))
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_STORE_ERROR"}}

    def store_temp_gender(self, user_id: str, gender: str) -> Dict:
        """Store temporary gender information during registration"""
        try:
            redis_key = f"ERP:TempGenderInfo:{user_id}"
            self.client.setex(redis_key, self.REDIS_TEMP_EXPIRE, gender)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_STORE_ERROR"}}

    def get_temp_gender(self, user_id: str) -> Dict:
        """Get temporary gender information for a user"""
        try:
            redis_key = f"ERP:TempGenderInfo:{user_id}"
            gender = self.client.get(redis_key)
            if gender is None:
                return {"success": False, "error": {"message": "GENDER_NOT_FOUND"}}
            return {"success": True, "gender": gender}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_GET_ERROR"}}

    def delete_temp_gender(self, user_id: str) -> Dict:
        """Delete temporary gender information after registration"""
        try:
            redis_key = f"ERP:TempGenderInfo:{user_id}"
            self.client.delete(redis_key)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_DELETE_ERROR"}}

    def get_temp_images(self, user_id: str) -> Dict:
        """Get all temporary face images for a user"""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            temp_data = self.client.get(redis_key)
            if temp_data is None:
                return {"success": False, "error": {"message": "IMAGES_NOT_FOUND"}}
            return {"success": True, "images": json.loads(temp_data)}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_GET_ERROR"}}

    def delete_temp_images(self, user_id: str) -> Dict:
        """Delete temporary face images after registration"""
        try:
            redis_key = f"ERP:TempFaceInfo:{user_id}"
            self.client.delete(redis_key)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_DELETE_ERROR"}}

    def cache_face_features(self, user_id: str, features: str) -> Dict:
        """Cache face features for faster verification"""
        try:
            redis_key = f"ERP:FaceFeatures:{user_id}"
            self.client.setex(redis_key, 3600, features)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_CACHE_ERROR"}}

    def get_cached_features(self, user_id: str) -> Dict:
        """Get cached face features"""
        try:
            redis_key = f"ERP:FaceFeatures:{user_id}"
            features = self.client.get(redis_key)
            if features is None:
                return {"success": False, "error": {"message": "FEATURES_NOT_FOUND"}}
            return {"success": True, "features": features}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_GET_ERROR"}}

    def delete_cached_features(self, user_id: str) -> Dict:
        try:
            redis_key = f"ERP:FaceFeatures:{user_id}"
            self.client.delete(redis_key)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_DELETE_ERROR"}}

    # v2 vector face
    def get_all_face_features(self) -> Dict:
        """Lấy tất cả face features từ Redis"""
        try:
            # Lấy tất cả keys có pattern ERP:FaceFeatures:*
            pattern = "ERP:FaceFeatures:*"
            keys = self.client.keys(pattern)
            
            if not keys:
                return {"success": True, "features": []}

            face_data = []
            for key in keys:
                user_id = key.split(':')[-1]  # Lấy user_id từ key
                features = self.client.get(key)
                if features:
                    # Chuyển đổi features string thành numpy array
                    feature_array = np.array(json.loads(features)[0])
                    face_data.append((user_id, feature_array))
            
            return {"success": True, "features": face_data}
        except Exception as e:
            return {"success": False, "error": {"message": "REDIS_GET_ERROR"}}

# Create singleton instance
redis_service = RedisService() 