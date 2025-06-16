import base64
import cv2 as cv
import numpy as np
from typing import Tuple

def base64_to_rgb_image(base64_string: str, max_size: int = 224) -> np.ndarray:
    try:
        # Remove prefix
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,', 1)[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        if img is None: raise ValueError("Invalid image data")

        # Resize if needed
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    except Exception as e:
        raise RuntimeError(f"Image decoding failed: {str(e)}")

def extract_face_box(box: np.ndarray, image: np.ndarray, margin: float = 0.2) -> Tuple[int, int, int, int]:
    """Extract face box coordinates with margin"""
    margin_pixels = int(margin * (box[2] - box[0]))
    x1, y1, x2, y2 = [int(b) for b in box]
    x1 = max(0, x1 - margin_pixels)
    y1 = max(0, y1 - margin_pixels)
    x2 = min(image.shape[1], x2 + margin_pixels)
    y2 = min(image.shape[0], y2 + margin_pixels)
    return x1, y1, x2, y2 

