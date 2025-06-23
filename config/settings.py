API_SECRET_KEY = "ERPA2Ah44CuOfgFCoiqYEVCDtqPLfIYGsQw"
# report dev start
BASE_URL = "https://face.anhouse.asia/erp-api-ekyc/api"
# report dev end

# Service Configuration - Reduced for better stability under high load
MAX_WORKERS = 3  # Reduced from 3 to 2 for better thread safety

# Face Detection Configuration
IMAGE_SIZE = 224
VALID_CHALLENGES = ['front', 'right', 'left']
FACE_MATCH_THRESHOLD = 0.92
# FACE_MATCH_THRESHOLD = 0.84
# Configure batch processing
SCORE_ANTI_SPOOFING_THRESHOLD = 0.99

BATCH_SIZE = 4
NUM_WORKERS = 2
