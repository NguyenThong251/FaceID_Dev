import google.generativeai as genai
import os
from typing import Dict, Any, Optional, List, Tuple
import base64
import io
from PIL import Image
import numpy as np
import fitz
import re
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from time import time

class ImageProcessor:
    def __init__(self):
        self.max_size = (640, 640)
        self.jpeg_quality = 75
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._cache_max_size = 100
        
    def _get_cache_key(self, image_data: bytes) -> str:
        return str(hash(image_data))
    def get_from_cache(self, image_data: bytes) -> Optional[Tuple[Image.Image, Dict]]:
        with self._cache_lock:
            key = self._get_cache_key(image_data)
            if key in self._cache:
                return self._cache[key]
        return None
    def add_to_cache(self, image_data: bytes, result: Tuple[Image.Image, Dict]):
        with self._cache_lock:
            key = self._get_cache_key(image_data)
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest item
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
    def optimize_image(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """Optimize image and prepare blob in one pass"""
        # Convert to grayscale
        image = image.convert('L')
        # Resize if needed
        if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        # Save optimized image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=self.jpeg_quality, optimize=True)
        img_byte_arr.seek(0)
        # Prepare blob data
        blob_data = {
            "data": img_byte_arr.getvalue(),
            "mime_type": "image/jpeg"
        }
        return image, blob_data
class GeminiOCRService:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        if self._initialized:
            return
        self.api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyADg5hXap-XhTRbiOL-y7GnUGjRm8tHfNc')
        genai.configure(api_key=self.api_key)
        self._model = None
        self._model_lock = threading.Lock()
        self.image_processor = ImageProcessor()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._initialized = True
    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash-002",
                        generation_config={
                            "temperature": 0.1,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 1024,
                        }
                    )
        return self._model
    @lru_cache(maxsize=100)
    def clean_base64(self, base64_string: str) -> str:
        if ';base64,' in base64_string:
            base64_string = base64_string.split(';base64,')[1]
        elif ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64_string.strip()
    def process_pdf(self, file_bytes: bytes) -> Image.Image:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        if len(pdf_document) == 0:
            raise ValueError("PDF has no pages")
        page = pdf_document[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        img_data = pix.tobytes("jpeg")
        image = Image.open(io.BytesIO(img_data))
        pdf_document.close()
        return image
    def base64_to_images(self, base64_string: str, is_pdf: bool = False) -> List[Image.Image]:
        try:
            if not base64_string:
                raise ValueError("Empty base64 string received")
            clean_data = self.clean_base64(base64_string)
            file_bytes = base64.b64decode(clean_data)
            cached_result = self.image_processor.get_from_cache(file_bytes)
            if cached_result:
                return [cached_result[0]]
            if is_pdf:
                image = self.process_pdf(file_bytes)
            else:
                image = Image.open(io.BytesIO(file_bytes))
            optimized_image, blob_data = self.image_processor.optimize_image(image)
            self.image_processor.add_to_cache(file_bytes, (optimized_image, blob_data))
            return [optimized_image]
        except Exception as e:
            raise ValueError(f"Failed to process file: {str(e)}")
    def process_ocr(self, file_base64: str, prompt: Optional[str] = None, is_pdf: bool = False) -> Dict[str, Any]:
        start_time = time()
        try:
            images = self.base64_to_images(file_base64, is_pdf)
            if not images:
                return { "success": False, "error": { "message": "No images were extracted from the file", "type": "ProcessingError"} }
            if not prompt:
                prompt = "Extract text. Format: key: value pairs."
            # Get cached blob data
            image = images[0]
            cached_result = self.image_processor.get_from_cache(image.tobytes())
            blob_data = cached_result[1] if cached_result else self.image_processor.optimize_image(image)[1]
            response = self.model.generate_content([prompt, blob_data])
            processing_time = time() - start_time
            return { "success": True, "result": {   "text": response.text,  "raw_response": response.text,  "page_count": 1,"processing_time": f"{processing_time:.2f}s"}}
        except Exception as e:
            return {"success": False,"error": {  "message": str(e), "type": type(e).__name__  } }
gemini_ocr_service = GeminiOCRService() 



