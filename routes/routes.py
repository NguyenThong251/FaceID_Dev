from flask import Blueprint, request, jsonify
import json

from .middleware import require_auth
from services.database_service import db_service
from services.redis_service import redis_service
from utils.image_utils import base64_to_rgb_image
from services.ekyc_service import eKYC_Service
from config.settings import VALID_CHALLENGES
from services.gemini_ocr_service import gemini_ocr_service
from utils.permissions import is_user_admin, get_module_info
erp_face_bp = Blueprint('erp-api-ekyc', __name__)
ekyc_service = eKYC_Service()

def get_error_response(error_code, details=None):
    """
    Generate a standardized error response
    """
    response = {
        'success': False,
        'error': {
            'message': error_code
        }
    }
    
    if details:
        response['error']['details'] = details
        
    return response

@erp_face_bp.route('/', methods=['POST'])
@require_auth
def handle_request():
    try:
        data = request.get_json()
        if not data or '_operation' not in data:
            return jsonify(get_error_response('INVALID_OPERATION')), 200

        operation = data.pop('_operation', None)

        # Route to appropriate handler based on operation
        if operation == 'CheckLiveness':
            return handle_check_liveness(data)
        elif operation == 'RegisterFace':
            return handle_register_face(data)
        elif operation == 'OCR':
            return handle_ocr(data)
        elif operation == 'SearchFace':
            return handle_search_face(data)
        elif operation == 'ClearTempImagesId':
            return handle_clear_temp_images_id(data)
        elif operation == 'ClearFaceId':
            return handle_clear_face_id(data)
        else:
            return jsonify(get_error_response('INVALID_OPERATION', f"Unknown operation: {operation}")), 200

    except Exception as e:
        return jsonify(get_error_response('INVALID_REQUEST', str(e))), 200

def handle_check_liveness(data):
    if not (data and data.get('frame') and data.get('userId')):
        return jsonify(get_error_response('MISSING_FIELDS')), 200
    
    user_id = data['userId']
    challenge = data.get('challenge', '')
    
    if challenge and db_service.check_user_exists(user_id):
        return jsonify(get_error_response('FACE_ALREADY_REGISTERED')), 200
    
    frame = base64_to_rgb_image(data['frame'], max_size=224)
    if frame is None:
        return jsonify(get_error_response('INVALID_IMAGE')), 200
        
    result = ekyc_service.check_liveness(frame, challenge, user_id)
    
    if result['success'] and challenge:
        redis_service.store_temp_image(user_id, challenge, data['frame'])    
    
    return jsonify(result), 200

def handle_register_face(data):
    user_id = data.get('userId')
    if not user_id:
        return jsonify(get_error_response('USER_ID_REQUIRED')), 200
    
    temp_images = redis_service.get_temp_images(user_id)
    if not temp_images:
        return jsonify(get_error_response('NO_TEMP_IMAGES')), 200
    
    missing_challenges = [c for c in VALID_CHALLENGES if c not in temp_images]
    if missing_challenges:
        return jsonify(get_error_response('LIVENESS_CHALLENGES_INCOMPLETE', {
            'missing_challenges': missing_challenges
        })), 200
    
    images = [temp_images[c] for c in VALID_CHALLENGES]
    frames = []
    features = []
    
    # Convert base64 images to frames
    for img_data in images:
        frame = base64_to_rgb_image(img_data)
        if frame is None:
            return jsonify(get_error_response('INVALID_IMAGE')), 200
        frames.append(frame)
    
    # Verify all faces are of the same person
    verification_result = ekyc_service.verify_registration_faces(frames)
    if not verification_result['success']:
        return jsonify(get_error_response('FACE_VERIFICATION_FAILED', verification_result['error'])), 200
    
    # Extract features after verification passes
    for frame in frames:
        feature = ekyc_service.extract_face_features(frame)
        if feature is not None:
            features.append(feature.tolist())
    
    if not features:
        return jsonify(get_error_response('FEATURE_EXTRACTION_FAILED')), 200
    
    db_service.save_face_features(user_id, json.dumps(images), json.dumps(features))
    redis_service.delete_temp_images(user_id)
    
    return jsonify({
        "success": True,
        "result": {
            "message": "Face registration completed successfully",
            "verification_details": verification_result['result']
        }
    }), 200

def handle_ocr(data):
    if not data:
        return jsonify(get_error_response('INVALID_REQUEST')), 200
    
    file_data = data.get('file') or data.get('image')
    if not file_data:
        return jsonify(get_error_response('FILE_REQUIRED')), 200
    
    estimated_size = len(file_data) * 3/4
    if estimated_size > 10 * 1024 * 1024:
        return jsonify(get_error_response('FILE_SIZE_EXCEEDED')), 200
    
    is_pdf = data.get('is_pdf', data.get('isPdf', False))
    prompt = data.get('prompt')
    result = gemini_ocr_service.process_ocr(file_data, prompt, is_pdf)
    
    return jsonify(result), 200

def handle_search_face(data):
    if not data or 'frame' not in data:
        return jsonify(get_error_response('MISSING_FIELDS')), 200
    
    frame = base64_to_rgb_image(data['frame'], max_size=224)
    if frame is None:
        return jsonify(get_error_response('INVALID_IMAGE')), 200
    
    top_k = int(data.get('top_k', 1))
    result = ekyc_service.search_face(frame, top_k=top_k)
    return jsonify(result), 200

def handle_clear_temp_images_id(data):
    if not data or 'user_id' not in data:
        return jsonify(get_error_response('USER_ID_REQUIRED')), 200
    user_id = data['user_id']
    redis_service.delete_temp_images(user_id)
    return jsonify({"success": True}), 200




# debug
def handle_clear_face_id(data):
    if not data or 'user_id' not in data:
        return jsonify(get_error_response('USER_ID_REQUIRED')), 200
        
    user_id = data['user_id']
    user_role = data.get('user_role')
    
    # Kiểm tra quyền admin
    module_info =  get_module_info(redis_service.client, user_role, "Timekeeping")
    if not module_info:
        return jsonify(get_error_response('MODULE_NOT_FOUND')), 200
        
    is_admin = is_user_admin(module_info)
    if not is_admin and user_id != data.get('current_user_id'):
        return jsonify(get_error_response('PERMISSION_DENIED')), 200
    
    # Xóa face ID
    result = redis_service.delete_cached_features(user_id)
    if not result['success']:
        return jsonify(result), 200
        
    result = db_service.delete_faceid_by_user_id(user_id)
    if not result['success']:
        return jsonify(result), 200
        
    return jsonify({"success": True}), 200
