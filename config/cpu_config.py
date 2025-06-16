import os
import warnings
import torch
import tensorflow as tf
from functools import lru_cache
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
device = torch.device('cpu')
torch.set_default_tensor_type(torch.FloatTensor)
@lru_cache(maxsize=32)
def load_model_weights(model, weights_path):
    try:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        return False

