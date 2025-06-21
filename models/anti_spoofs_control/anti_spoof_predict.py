# import os
# import cv2
# import math
# import torch
# import numpy as np
# import torch.nn.functional as F


# from models.anti_spoofs_control.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
# from models.anti_spoofs_control.data_io import transform as trans
# from models.anti_spoofs_control.utility import get_kernel, parse_model_name

# MODEL_MAPPING = {
#     'MiniFASNetV1': MiniFASNetV1,
#     'MiniFASNetV2': MiniFASNetV2,
#     'MiniFASNetV1SE':MiniFASNetV1SE,
#     'MiniFASNetV2SE':MiniFASNetV2SE
# }


# class Detection:
#     def __init__(self):
#         project_root = "/www/wwwroot/eKYC"
#         caffemodel = os.path.join(project_root, "models/liveness_detection/anti_spoofs/detection_model/Widerface-RetinaFace.caffemodel")
#         deploy = os.path.join(project_root, "models/liveness_detection/anti_spoofs/detection_model/deploy.prototxt")
#         self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
#         self.detector_confidence = 0.6

#     def get_bbox(self, img):
#         height, width = img.shape[0], img.shape[1]
#         aspect_ratio = width / height
#         if img.shape[1] * img.shape[0] >= 192 * 192:
#             img = cv2.resize(img,
#                              (int(192 * math.sqrt(aspect_ratio)),
#                               int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

#         blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
#         self.detector.setInput(blob, 'data')
#         out = self.detector.forward('detection_out').squeeze()
#         max_conf_index = np.argmax(out[:, 2])
#         left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
#                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
#         bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
#         return bbox


# class AntiSpoofPredict(Detection):
#     def __init__(self, device_id):
#         super(AntiSpoofPredict, self).__init__()
#         self.device = torch.device("cuda:{}".format(device_id)
#                                   if torch.cuda.is_available() else "cpu")

#     def _load_model(self, model_path):
#         # define model
#         model_name = os.path.basename(model_path)
#         h_input, w_input, model_type, _ = parse_model_name(model_name)
#         self.kernel_size = get_kernel(h_input, w_input,)
#         self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

#         # load model weight
#         state_dict = torch.load(model_path, map_location=self.device)
#         keys = iter(state_dict)
#         first_layer_name = keys.__next__()
#         if first_layer_name.find('module.') >= 0:
#             from collections import OrderedDict
#             new_state_dict = OrderedDict()
#             for key, value in state_dict.items():
#                 name_key = key[7:]
#                 new_state_dict[name_key] = value
#             self.model.load_state_dict(new_state_dict)
#         else:
#             self.model.load_state_dict(state_dict)
#         return None

#     def predict(self, img, model_path):
#         test_transform = trans.Compose([
#             trans.ToTensor(),
#         ])
#         img = test_transform(img)
#         img = img.unsqueeze(0).to(self.device)
#         self._load_model(model_path)
#         self.model.eval()
#         with torch.no_grad():
#             result = self.model.forward(img)
#             result = F.softmax(result).cpu().numpy()
#         return result


import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
import gc

from models.anti_spoofs_control.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from models.anti_spoofs_control.data_io import transform as trans
from models.anti_spoofs_control.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        caffemodel = os.path.join(project_root, "models/liveness_detection/anti_spoofs/detection_model/Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(project_root, "models/liveness_detection/anti_spoofs/detection_model/deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self,device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                  if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size)

        # load model weight
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        # Resize ảnh nếu quá lớn
        max_size = 512
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        
        # Process ảnh
        try:
            img = test_transform(img)
            img = img.unsqueeze(0)
            
            # Load model nếu chưa có
            if self.model is None:
                self._load_model(model_path)
            
            self.model.eval()
            with torch.no_grad():
                result = self.model.forward(img)
                # Fix: Add dim=1 parameter to softmax to specify the dimension
                result = F.softmax(result, dim=1).numpy()
            
            # Cleanup
            del img
            gc.collect()
            
            return result
            
        except Exception as e:
            gc.collect()
            raise e