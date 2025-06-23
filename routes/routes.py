# report dev start
from flask import Blueprint, request, jsonify, send_file
# report dev end
from .middleware import require_auth
from . import route_handlers
import asyncio
# report dev start
from services.storage_service import storage_service
from utils.image_utils import decode_image_key
import os
# report dev end    

erp_face_bp = Blueprint('erp-api-ekyc', __name__)

@erp_face_bp.route('/', methods=['POST'], strict_slashes=False)
@erp_face_bp.route('', methods=['POST'], strict_slashes=False)
@require_auth
def handle_request():
    try:
        data = request.get_json()
        if not data or '_operation' not in data:
            return jsonify({'success': False,'error': {'message': 'INVALID_OPERATION'}}, 200)
        operation = data.pop('_operation', None)
        handler = route_handlers.get(operation)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                return asyncio.run(handler(data))
            return handler(data)
            
        return jsonify({
            'success': False,
            'error': {
                'message': 'INVALID_OPERATION',
                'details': f"Unknown operation: {operation}"
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'message': 'INVALID_REQUEST',
                'details': str(e)
            }
        }), 200

# report dev start
@erp_face_bp.route('/image/face', methods=['GET'])
def report_image():
    key = request.args.get('key')
    if not key:
        return jsonify({"success": False, "error": "UNAUTHORIZED"}), 200
    filename = decode_image_key(key)
    if not filename:
        return jsonify({"success": False, "error": "INVALID_KEY"}), 200
    image_path = os.path.join(storage_service.faces_dir, filename)
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": "IMAGE_NOT_FOUND"}), 200
    return send_file(image_path, mimetype='image/jpeg')
# report dev end