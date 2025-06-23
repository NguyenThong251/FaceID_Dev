from flask import jsonify, request
import json
import asyncio
from services.database_service import db_service
from services.redis_service import redis_service
from services.storage_service import storage_service
# report dev start
from utils.image_utils import base64_to_rgb_image, encode_image_key
from config.settings import VALID_CHALLENGES, BASE_URL
# report dev end
from services.ekyc_service import eKYC_Service
from services.gemini_ocr_service import gemini_ocr_service
import cv2
ekyc_service = eKYC_Service()

def get_error_response(error_code, details=None):
    response = {
        'success': False,
        'error': {
            'message': error_code
        }
    }
    if details:
        response['error']['details'] = details
    return response

async def handle_check_liveness(data):
    if not (data and data.get('frame') and data.get('userId')):
        return jsonify(get_error_response('MISSING_FIELDS')), 200
    
    user_id = data['userId']
    challenge = data.get('challenge', '')
    
    # Check user existence asynchronously
    if challenge:
        user_exists = await asyncio.to_thread(db_service.check_user_exists, user_id)
        if user_exists:
            return jsonify(get_error_response('FACE_ALREADY_REGISTERED')), 200
    
    # Process image asynchronously
    frame = await asyncio.to_thread(base64_to_rgb_image, data['frame'], 224)
    if frame is None:
        return jsonify(get_error_response('INVALID_IMAGE')), 200
        
    # Check liveness asynchronously
    result = await asyncio.to_thread(ekyc_service.check_liveness, frame, challenge, user_id)
    
    if result['success'] and challenge:
        # Store image asynchronously
        await asyncio.to_thread(redis_service.store_temp_image, user_id, challenge, data['frame'])    
    
    return jsonify(result), 200



async def handle_register_face(data):
    user_id = data.get('userId')
    if not user_id:
        return jsonify(get_error_response('USER_ID_REQUIRED')), 200
    temp_images = await asyncio.to_thread(redis_service.get_temp_images, user_id)
    if not temp_images:
        return jsonify(get_error_response('NO_TEMP_IMAGES')), 200
    missing_challenges = [c for c in VALID_CHALLENGES if c not in temp_images]
    if missing_challenges:
        return jsonify(get_error_response('LIVENESS_CHALLENGES_INCOMPLETE', {
            'missing_challenges': missing_challenges
        })), 200
    images = [temp_images[c] for c in VALID_CHALLENGES]
    async def process_image(img_data, challenge):
        frame = await asyncio.to_thread(base64_to_rgb_image, img_data)
        if frame is None:
            return None, None
        feature = await asyncio.to_thread(ekyc_service.extract_face_features, frame)
        # report dev start
        # Lưu ảnh vào storage/faces với hash random để tăng bảo mật
        image_bytes = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes()
        image_path = await asyncio.to_thread(storage_service.save_image, user_id, challenge, image_bytes)
        # report dev end
        return frame, feature.tolist() if feature is not None else None
    results = await asyncio.gather(*[process_image(img, c) for img, c in zip(images, VALID_CHALLENGES)])
    frames = []
    features = []
    for frame, feature in results:
        if frame is None:
            return jsonify(get_error_response('INVALID_IMAGE')), 200
        frames.append(frame)
        if feature is not None:
            features.append(feature)
    if not features:
        return jsonify(get_error_response('FEATURE_EXTRACTION_FAILED')), 200
    verification_result = await asyncio.to_thread(
        ekyc_service.verify_registration_faces, frames
    )
    if not verification_result['success']:
        return jsonify(get_error_response('FACE_VERIFICATION_FAILED', verification_result['error'])), 200
    await asyncio.to_thread(
        db_service.save_face_features,
        user_id,
        json.dumps(images),
        json.dumps(features)
    )
    await asyncio.to_thread(redis_service.delete_temp_images, user_id)
    await asyncio.to_thread(redis_service.delete_cached_userInfo, user_id)
    return jsonify({
        "success": True,
        "result": {
            "message": "Face registration completed successfully",
            "verification_details": verification_result['result']
        }
    }), 200

async def handle_ocr(data):
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
    
    # Process OCR asynchronously
    result = await asyncio.to_thread(
        gemini_ocr_service.process_ocr,
        file_data,
        prompt,
        is_pdf
    )
    
    return jsonify(result), 200

async def handle_search_face(data):
    if not data or 'frame' not in data:
        return jsonify(get_error_response('MISSING_FIELDS')), 200
    
    # Process image asynchronously
    frame = await asyncio.to_thread(base64_to_rgb_image, data['frame'], 224)
    if frame is None:
        return jsonify(get_error_response('INVALID_IMAGE')), 200
    
    top_k = int(data.get('top_k', 1))
    result = await asyncio.to_thread(ekyc_service.search_face, frame, top_k=top_k)
    return jsonify(result), 200

async def handle_delete_temp_images_id(data):
    if not data or 'userId' not in data:
        return jsonify(get_error_response('USER_ID_REQUIRED')), 200
    user_id = data['userId']
    await asyncio.to_thread(redis_service.delete_temp_images, user_id)
    return jsonify({"success": True}), 200



# report dev start
async def handle_delete_face_id(data):
    # user_id = data.get('userId')
    # admin_id = data.get('adminId')

    # if not user_id or not admin_id:
    #     return jsonify(get_error_response('MISSING_FIELDS', 'userId and adminId are required')), 200
        
    # is_user_exist = await asyncio.to_thread(db_service.check_user_exists, user_id)
    # is_admin = await asyncio.to_thread(db_service.is_admin, admin_id)
    
    # if not is_user_exist:
    #     return jsonify(get_error_response('USER_NOT_FOUND')), 200
    # if not is_admin:
    #     return jsonify(get_error_response('PERMISSION_DENIED')), 200


    if not data or 'userId' not in data:
        return jsonify(get_error_response('USER_ID_REQUIRED')), 200

    user_id = data['userId']
    user_admin_id = request.user_id
    is_user, is_admin = await asyncio.gather(
        asyncio.to_thread(db_service.check_user_exists, user_id),
        asyncio.to_thread(db_service.is_admin, user_admin_id)
    )
    # fix crash dev start
    if not is_user:
        return jsonify(get_error_response('USER_NOT_FOUND')), 200
    if not is_admin:
        return jsonify(get_error_response('PERMISSION_DENIED')), 200
    # fix crash dev end
    try:
        db_success, redis_success, storage_success = await asyncio.gather(
            asyncio.to_thread(db_service.delete_faceid_by_user_id, user_id),
            asyncio.to_thread(redis_service.delete_cached_features, user_id),
            asyncio.to_thread(storage_service.delete_user_images, user_id)
        )
        
        if not db_success:
            return jsonify(get_error_response('DB_DELETE_ERROR')), 200
        if not redis_success:
            return jsonify(get_error_response('REDIS_DELETE_ERROR')), 200
        if not storage_success:
            return jsonify(get_error_response('STORAGE_DELETE_ERROR')), 200

        return jsonify({"success": True, "message": "Face ID cleared successfully"}), 200
        
    except Exception as e:
        return jsonify(get_error_response('INVALID_REQUEST', str(e))), 200
# report dev end




# report dev start
# utility function
def create_pagination(data, page=1, limit=20):
    total_hits = len(data)
    total_pages = (total_hits + limit - 1) // limit
    next_page = page + 1 if page < total_pages else False
    start = (page - 1) * limit
    end = start + limit
    return {
        "hits": data[start:end],
        "totalHits": total_hits,
        "totalPages": total_pages,
        "nextPage": next_page,
        "timeResponse": 0
    }



def report_users_face(data):
    page = int(data.get('page', 1))
    limit = int(data.get('limit', 10))
    users = db_service.get_all_users_face()
    result = []
    for user in users:
        user_id = user['userId']
        challenges = storage_service.get_user_images(user_id)
        
        image_urls = []
        for challenge in challenges:
            filename = storage_service.get_image_filename(user_id, challenge)
            if filename:
                key = encode_image_key(filename)
                url = f"{BASE_URL}/image/face?key={key}"
                image_urls.append(url)
        
        user_result = {
            'userId': user_id,
            'timecreate': user.get('timecreate'),
            'images': image_urls
        }
        result.append(user_result)
    
    return create_pagination(result, page, limit)
# report dev end