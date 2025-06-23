# report dev start
import json
import os
from typing import Optional, List, Dict
import hashlib
import secrets

class StorageService:
    def __init__(self):
        self.faces_dir = "storage/faces"
        self._ensure_faces_directory()
    
    def _ensure_faces_directory(self):
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
    
    def get_user_images(self, user_id: str) -> List[str]:
        try:
            if not os.path.exists(self.faces_dir):
                return []
            
            files = os.listdir(self.faces_dir)
            challenges = []
            for f in files:
                if f.startswith(f"{user_id}_") and f.endswith(".jpg"):
                    # Tách challenge từ tên file (có thể có hash)
                    parts = f[len(user_id)+1:-4].split('_')  # Bỏ .jpg và tách theo _
                    if len(parts) >= 1:
                        challenge = parts[0]
                        if challenge not in challenges:
                            challenges.append(challenge)
            return challenges
        except Exception:
            return []

    def get_image_path(self, user_id: str, challenge: str) -> Optional[str]:
        try:
            if not os.path.exists(self.faces_dir):
                return None
            
            files = os.listdir(self.faces_dir)
            for f in files:
                if f.startswith(f"{user_id}_{challenge}_") and f.endswith(".jpg"):
                    return os.path.join(self.faces_dir, f)
            return None
        except Exception:
            return None

    def get_image_filename(self, user_id: str, challenge: str) -> Optional[str]:
        try:
            if not os.path.exists(self.faces_dir):
                return None
            
            files = os.listdir(self.faces_dir)
            for f in files:
                if f.startswith(f"{user_id}_{challenge}_") and f.endswith(".jpg"):
                    return f
            return None
        except Exception:
            return None

    def save_image(self, user_id: str, challenge: str, image_bytes: bytes, filename: str = None) -> Optional[str]:
        random_salt = secrets.token_hex(8)
        hash_input = f"{user_id}_{challenge}_{random_salt}".encode()
        file_hash = hashlib.sha256(hash_input).hexdigest()[:12] 
        try:
            self._ensure_faces_directory()
            if not filename:
                filename = f"{user_id}_{challenge}_{file_hash}.jpg"
            
            image_path = os.path.join(self.faces_dir, filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            return image_path
        except Exception:
            return None

    def delete_user_images(self, user_id: str) -> bool:
        try:
            if not os.path.exists(self.faces_dir):
                return True
            
            files = os.listdir(self.faces_dir)
            deleted_count = 0
            
            for f in files:
                if f.startswith(f"{user_id}_") and f.endswith(".jpg"):
                    file_path = os.path.join(self.faces_dir, f)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except:
                        pass
            
            return True
        except Exception:
            return False

    def get_storage_info(self) -> Dict:
        try:
            if not os.path.exists(self.faces_dir):
                return {"total_files": 0, "total_size": 0, "directory": self.faces_dir}
            
            files = os.listdir(self.faces_dir)
            total_size = 0
            
            for f in files:
                if f.endswith(".jpg"):
                    file_path = os.path.join(self.faces_dir, f)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
            
            return {
                "total_files": len([f for f in files if f.endswith(".jpg")]),
                "total_size": total_size,
                "directory": self.faces_dir
            }
        except Exception:
            return {"total_files": 0, "total_size": 0, "directory": self.faces_dir}

    def get_all_users_from_storage(self) -> List[str]:
        try:
            if not os.path.exists(self.faces_dir):
                return []
            
            files = os.listdir(self.faces_dir)
            user_ids = set()
            
            for f in files:
                if f.endswith(".jpg"):
                    # Tách user_id từ tên file
                    parts = f.split('_')
                    if len(parts) >= 1:
                        user_id = parts[0]
                        user_ids.add(user_id)
            
            return list(user_ids)
        except Exception:
            return []

storage_service = StorageService()
# report dev end