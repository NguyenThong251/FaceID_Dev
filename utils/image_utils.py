import base64
import cv2 as cv
import numpy as np
from typing import Tuple
import hmac
import hashlib
from config.settings import API_SECRET_KEY

def base64_to_rgb_image(base64_string: str, max_size: int = 224) -> np.ndarray:
    try:
        # Remove prefix without string operations if possible
        start = base64_string.find('base64,')
        if start != -1:
            base64_string = base64_string[start + 7:]
        
        # Decode base64 and convert to image in one step
        img_array = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        if img is None: raise ValueError("Invalid image data")

        # Calculate resize scale once
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > max_size:
            scale = max_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            # Use INTER_LINEAR for faster resizing
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    except Exception as e:
        raise RuntimeError(f"Image decoding failed: {str(e)}")

def extract_face_box(box: np.ndarray, image: np.ndarray, margin: float = 0.2) -> Tuple[int, int, int, int]:
    # Calculate margin once using numpy operations
    margin_pixels = int(margin * (box[2] - box[0]))
    
    # Convert box coordinates to integers and apply margin
    x1 = max(0, int(box[0]) - margin_pixels)
    y1 = max(0, int(box[1]) - margin_pixels)
    x2 = min(image.shape[1], int(box[2]) + margin_pixels)
    y2 = min(image.shape[0], int(box[3]) + margin_pixels)
    
    return x1, y1, x2, y2

def encode_image_key(user_id: str, challenge: str) -> str:
    """Tạo key bảo mật cho ảnh dựa trên user_id, challenge và secret key."""
    raw = f"{user_id}:{challenge}".encode()
    secret = API_SECRET_KEY.encode()
    return hmac.new(secret, raw, hashlib.sha256).hexdigest() + f":{user_id}:{challenge}"

def decode_image_key(key: str) -> tuple:
    """Giải mã key ảnh, trả về (user_id, challenge) nếu hợp lệ, ngược lại trả về None."""
    try:
        hash_part, user_id, challenge = key.split(":", 2)
        expected = encode_image_key(user_id, challenge)
        if expected.split(":")[0] == hash_part:
            return user_id, challenge
        return None
    except Exception:
        return None

