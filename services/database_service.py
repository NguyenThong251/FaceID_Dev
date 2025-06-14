from mysql.connector import pooling
from typing import Optional, List, Tuple, Dict
import json
from .redis_service import redis_service
import faiss
import numpy as np


class DatabaseService:
    # MySQL Configuration
    DB_CONFIG = {
        'host': '127.0.0.1',
        'user': 'vtiger',
        'password': '',
        'database': 'vtiger',
        'pool_name': 'mypool',
        'pool_size': 5
    }

    def __init__(self):
        self.connection_pool = pooling.MySQLConnectionPool(**self.DB_CONFIG)

    def get_connection(self):
        return self.connection_pool.get_connection()

    def check_user_exists(self, user_id: str) -> Dict:
        """Check if user already has registered face features"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM vtiger_timekeeping_face WHERE owner = %s LIMIT 1", 
                        (user_id,)
                    )
                    exists = cursor.fetchone() is not None
                    return {"success": True, "exists": exists}
        except Exception as e:
            return {"success": False, "error": {"message": "DB_CHECK_ERROR"}}

    def save_face_features(self, user_id: str, images: str, features: str, gender: str = None) -> Dict:
        """Save user's face features, images and gender information to database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO vtiger_timekeeping_face 
                        (owner, images, features, created_at)
                        VALUES (%s, %s, %s, NOW())
                        """,
                        (user_id, images, features)
                    )
                    conn.commit()
                    redis_service.cache_face_features(user_id, features)
                    return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "DB_SAVE_ERROR"}}

    def get_stored_features(self, user_id: str) -> Dict:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT features 
                        FROM vtiger_timekeeping_face 
                        WHERE owner = %s 
                        ORDER BY created_at DESC 
                        LIMIT 1
                        """,
                        (user_id,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return {"success": False, "error": {"message": "FEATURES_NOT_FOUND"}}
                    return {"success": True, "features": row[0]}
        except Exception as e:
            return {"success": False, "error": {"message": "DB_GET_ERROR"}}

    def delete_faceid_by_user_id(self, user_id: str) -> Dict:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM vtiger_timekeeping_face WHERE owner = %s",
                        (user_id,)
                    )
                    conn.commit()
                    return {"success": True}
        except Exception as e:
            return {"success": False, "error": {"message": "DB_DELETE_ERROR"}}


    
# v2 vector face
    def get_all_face_features(self) -> Dict:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT owner, features 
                        FROM vtiger_timekeeping_face 
                        ORDER BY created_at DESC
                        """
                    )
                    results = cursor.fetchall()
                    face_data = []
                    for user_id, features in results:
                        redis_service.cache_face_features(user_id, features)
                        feature_array = np.array(json.loads(features)[0]) 
                        face_data.append((user_id, feature_array))
                    return {"success": True, "features": face_data}
        except Exception as e:
            return {"success": False, "error": {"message": "DB_GET_ERROR"}}

    def build_face_index(self) -> Dict:
        try:
            face_data_result = self.get_all_face_features()
            if not face_data_result['success']:
                return face_data_result

            face_data = face_data_result['features']
            if not face_data:
                return {"success": True, "index": None, "user_ids": []}

            user_ids = [data[0] for data in face_data]
            features = np.array([data[1] for data in face_data])
            dimension = features.shape[1]
            index = faiss.IndexFlatL2(dimension) 
            index.add(features.astype('float32'))
            
            return {"success": True, "index": index, "user_ids": user_ids}
        except Exception as e:
            return {"success": False, "error": {"message": "INDEX_BUILD_ERROR"}}

    def search_similar_faces(self, query_feature: np.ndarray, k: int = 5) -> Dict:
        try:
            face_data = []
            redis_features = redis_service.get_all_face_features()
            
            if redis_features['success']:
                face_data = redis_features['features']
            else:  
                face_data_result = self.get_all_face_features()
                if not face_data_result['success']:
                    return face_data_result
                face_data = face_data_result['features']

            if not face_data:
                return {"success": True, "results": []}

            user_ids = [data[0] for data in face_data]
            features = np.array([data[1] for data in face_data])
            dimension = features.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(features.astype('float32'))
            query_feature = query_feature.reshape(1, -1).astype('float32')
            distances, indices = index.search(query_feature, k)
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(user_ids):
                    similarity = 1 / (1 + distance)  # Chuyển đổi khoảng cách thành độ tương đồng
                    results.append((user_ids[idx], float(similarity)))
            
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": {"message": "FACE_SEARCH_ERROR"}}
db_service = DatabaseService() 