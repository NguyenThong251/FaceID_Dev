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
        'user': 'vtigernew',
        'password': 'jt5pbdnW772iMhrj',
        'database': 'vtigernew',
        'pool_name': 'mypool',
        'pool_size': 5
    }

    def __init__(self):
        self.connection_pool = pooling.MySQLConnectionPool(**self.DB_CONFIG)

    def get_connection(self):
        return self.connection_pool.get_connection()

    def check_user_exists(self, user_id: str) -> bool:
        """Check if user already has registered face features"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM vtiger_timekeeping_face WHERE owner = %s LIMIT 1", 
                        (user_id,)
                    )
                    return cursor.fetchone() is not None
        except Exception as e:
            return False

    def save_face_features(self, user_id: str, images: str, features: str, gender: str = None) -> bool:
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
                    return True
        except Exception as e:
            return False

    def get_stored_features(self, user_id: str) ->  Optional[str]:
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
                        return None
                    return row[0]
        except Exception as e:
            return None

    def delete_faceid_by_user_id(self, user_id: str) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM vtiger_timekeeping_face WHERE owner = %s",
                        (user_id,)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            return False


    
# v2 vector face
    def get_all_face_features(self) -> List[Tuple[str, np.ndarray]]:
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
                    return face_data
        except Exception as e:
            return []

    def build_face_index(self) -> Tuple[faiss.Index, List[str]]:
        try:
            face_data_result = self.get_all_face_features()
            if not face_data_result:
                return face_data_result

            face_data = face_data_result['features']
            if not face_data:
                return []

            user_ids = [data[0] for data in face_data]
            features = np.array([data[1] for data in face_data])
            dimension = features.shape[1]
            index = faiss.IndexFlatL2(dimension) 
            index.add(features.astype('float32'))
            
            return {"index": index, "user_ids": user_ids}
        except Exception as e:
            return []

    def search_similar_faces(self, query_feature: np.ndarray, k: int = 5) ->  List[Tuple[str, float]]:
        try:
            face_data = []
            redis_features = redis_service.get_all_face_features()
            
            if redis_features:
                face_data = redis_features
            else:  
                face_data_result = self.get_all_face_features()
                if not face_data_result:
                    return face_data_result
                face_data = face_data_result

            if not face_data:
                return []

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
            
            return results
        except Exception as e:
            return []
    def is_admin(self, user_id: str) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT is_admin FROM vtiger_users WHERE id = %s", (user_id,))
                    result = cursor.fetchone()
                    return result[0] == 'on' if result else False
        except Exception as e:
            return False

    def get_all_users_face(self) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(
                        """
                        SELECT owner, created_at 
                        FROM vtiger_timekeeping_face 
                        ORDER BY created_at DESC
                        """
                    )
                    results = cursor.fetchall()
                    users = []
                    for row in results:
                        user = {
                            'userId': row['owner'],
                            'timecreate': row['created_at'].isoformat() if row['created_at'] else None
                        }
                        users.append(user)
                    return users
        except Exception as e:
            print(f"Error in get_all_users_face: {e}")
            return []

db_service = DatabaseService() 