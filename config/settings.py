API_SECRET_KEY = "ERPA2Ah44CuOfgFCoiqYEVCDtqPLfIYGsQw"

# Service Configuration
MAX_WORKERS = 3  # Maximum number of parallel workers

# Face Detection Configuration
IMAGE_SIZE = 224
VALID_CHALLENGES = ['front', 'right', 'left']
FACE_MATCH_THRESHOLD = 0.92
# FACE_MATCH_THRESHOLD = 0.84
# Configure batch processing
SCORE_ANTI_SPOOFING_THRESHOLD = 0.99

BATCH_SIZE = 4
NUM_WORKERS = 2
