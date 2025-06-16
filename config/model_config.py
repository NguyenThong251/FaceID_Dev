import os
import warnings
import torch
import tensorflow as tf

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Disable PyTorch warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Force CPU usage for PyTorch
torch.set_num_threads(4)  # Adjust based on your CPU cores
torch.set_num_interop_threads(4)

# Configure PyTorch to use CPU
device = torch.device('cpu')
torch.set_default_tensor_type(torch.FloatTensor)

# Configure model loading to use CPU
def load_model_weights(model, weights_path):
    """Load model weights safely with CPU device"""
    try:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        return False 