# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from pathlib import Path

class MaskDetectionService:
	def __init__(self):
		# Get the base path for models
		base_path = Path(__file__).parent.parent / 'models'
		
		# Load face detector model
		prototxt_path = str(base_path / 'face_detector' / 'deploy.prototxt')
		weights_path = str(base_path / 'face_detector' / 'res10_300x300_ssd_iter_140000.caffemodel')
		self.face_net = cv2.dnn.readNet(prototxt_path, weights_path)
		
		# Load mask detector model
		mask_model_path = str(base_path / 'mask_detector.model')
		self.mask_net = load_model(mask_model_path)

	def detect_mask(self, frame):
		"""
		Detect masks in a single frame
		Returns: Dictionary containing detection results
		"""
		# Initialize result dictionary
		result = {
			'success': False,
			'has_mask': False,
			'confidence': 0.0,
			'face_detected': False,
			'error': None
		}

		try:
			# Get frame dimensions and create blob
			h, w = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
				(104.0, 177.0, 123.0))

			# Detect faces
			self.face_net.setInput(blob)
			detections = self.face_net.forward()

			# Process if faces detected
			if detections.shape[2] > 0:
				# Get the detection with highest confidence
				max_confidence_idx = np.argmax(detections[0, 0, :, 2])
				confidence = detections[0, 0, max_confidence_idx, 2]

				if confidence > 0.5:
					result['face_detected'] = True
					
					# Get face coordinates
					box = detections[0, 0, max_confidence_idx, 3:7] * np.array([w, h, w, h])
					startX, startY, endX, endY = box.astype("int")
					startX, startY = max(0, startX), max(0, startY)
					endX, endY = min(w - 1, endX), min(h - 1, endY)

					# Extract and preprocess face ROI
					face = frame[startY:endY, startX:endX]
					if face.size > 0:  # Check if face region is valid
						face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224))
						face = preprocess_input(img_to_array(face))
						mask, without_mask = self.mask_net.predict(face[np.newaxis], verbose=0)[0]
						
						result.update({
							'success': True,
							'has_mask': mask > without_mask,
							'confidence': mask,
							'face_box': (startX, startY, endX, endY)
						})
					else:
						result['error'] = "Invalid face region"
				else:
					result['error'] = "Low confidence"
			else:
				result['error'] = "No faces"

			return result

		except Exception as e:
			result['error'] = str(e)
			return result

# Create singleton instance
mask_detection_service = MaskDetectionService()