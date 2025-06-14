# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import cvlib as cv
# import os

# class GenderDetectionService:
#     def __init__(self):
#         model_path = os.path.join(os.path.dirname(__file__), '../models/gender_detection.model')
#         self.model = load_model(model_path)
#         self.classes = ['man', 'woman']

#     def detect_gender(self, image):
#         """
#         Detect gender from a given image
#         :param image: numpy array of the image (BGR format)
#         :return: tuple (gender, confidence)
#         """
#         try:
#             # detect face
#             face, confidence = cv.detect_face(image)
            
#             if not face:
#                 return None, None

#             # use the first detected face
#             f = face[0]
#             (startX, startY) = f[0], f[1]
#             (endX, endY) = f[2], f[3]

#             # crop the detected face region
#             face_crop = np.copy(image[startY:endY, startX:endX])

#             if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#                 return None, None

#             # preprocessing
#             face_crop = cv2.resize(face_crop, (96, 96))
#             face_crop = face_crop.astype("float") / 255.0
#             face_crop = img_to_array(face_crop)
#             face_crop = np.expand_dims(face_crop, axis=0)

#             # predict gender
#             conf = self.model.predict(face_crop)[0]
#             idx = np.argmax(conf)
            
#             return self.classes[idx], float(conf[idx] * 100)
#         except Exception as e:
#             print(f"Error in gender detection: {str(e)}")
#             return None, None

# # Create singleton instance
# gender_detection_service = GenderDetectionService() 


# from tensorflow.keras.utils import img_to_array
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import cvlib as cv
# import os
# import tensorflow as tf

# class GenderDetectionService:
#     def __init__(self):
#         try:
#             model_path = os.path.join(os.path.dirname(__file__), '../models/gender_detection.model')
#             self.model = load_model(model_path, compile=False)
#             self.classes = ['man', 'woman']
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             raise

#     def detect_gender(self, image):
#         """
#         Detect gender from a given image
#         :param image: numpy array of the image (BGR format)
#         :return: tuple (gender, confidence)
#         """
#         try:
#             # detect face
#             face, confidence = cv.detect_face(image)
            
#             if not face:
#                 return None, None

#             # use the first detected face
#             f = face[0]
#             (startX, startY) = f[0], f[1]
#             (endX, endY) = f[2], f[3]

#             # crop the detected face region
#             face_crop = np.copy(image[startY:endY, startX:endX])

#             if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#                 return None, None

#             # preprocessing với error handling
#             try:
#                 face_crop = cv2.resize(face_crop, (96, 96))
#                 face_crop = face_crop.astype("float32") / 255.0
#                 face_crop = img_to_array(face_crop)
#                 face_crop = np.expand_dims(face_crop, axis=0)

#                 # predict gender với error handling
#                 with tf.device('/CPU:0'):
#                     conf = self.model.predict(face_crop, verbose=0)[0]
#                 idx = np.argmax(conf)
                
#                 return self.classes[idx], float(conf[idx] * 100)
#             except Exception as e:
#                 print(f"Error in preprocessing or prediction: {str(e)}")
#                 return None, None
                
#         except Exception as e:
#             print(f"Error in gender detection: {str(e)}")
#             return None, None

# # Create singleton instance
# gender_detection_service = GenderDetectionService() 