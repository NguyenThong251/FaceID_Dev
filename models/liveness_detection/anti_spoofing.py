import torch
import torchvision
from PIL import Image
import numpy as np
import os
from .aenet import AENet
import cv2

class AntiSpoofingDetector:
    def __init__(self, model_path=None):
        self.num_class = 2
        self.net = AENet(num_classes=self.num_class)
        
        # Load model weights
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'face_recognition', 'ckpt_iter.pth.tar')
            
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'face_recognition', 'ckpt_iter.pth.tar'),
                os.path.join('models', 'face_recognition', 'ckpt_iter.pth.tar'),
                os.path.join('..', 'face_recognition', 'ckpt_iter.pth.tar')
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                raise FileNotFoundError(f"Model weights not found at {model_path} or any alternative locations")
            
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self._load_pretrained(checkpoint['state_dict'])
        
        self.new_width = self.new_height = 224
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.new_width, self.new_height)),
            torchvision.transforms.ToTensor(),
        ])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.net.eval()

    def _load_pretrained(self, state_dict):
        own_state = self.net.state_dict()
        for name, param in state_dict.items():
            realname = name.replace('module.','')
            if realname in own_state:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[realname].copy_(param)
                except:
                    print(f'Error loading parameter {realname}')

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, (self.new_width, self.new_height))
            image = Image.fromarray(image)
        return self.transform(image)

    def detect(self, image):
        
        try:
            processed_data = self.preprocess_image(image)
            processed_data = processed_data.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.net(processed_data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                prob = float(probabilities[0][1].item())  # Probability of being real
                
                # Optimize threshold check
                # is_real = prob <= 0.008  # Threshold adjusted for better performance
                is_real = prob <= 0.0005  # Threshold adjusted for better performance

            
            return is_real, prob
            
        except Exception as e:
            return False, 0.0