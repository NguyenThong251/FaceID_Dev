from flask import Blueprint, request, jsonify
from .middleware import require_auth
from . import route_handlers
import asyncio

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
            # Handle both sync and async handlers
            if asyncio.iscoroutinefunction(handler):
                # Run async handler in event loop
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
