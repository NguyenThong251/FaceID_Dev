from modules.ekyc_module import (
    handle_check_liveness,
    handle_register_face,
    handle_ocr,
    handle_search_face,
    handle_delete_temp_images_id,
    handle_delete_face_id
)

# Define route handlers mapping
route_handlers = {
    'CheckLiveness': handle_check_liveness,
    'RegisterFace': handle_register_face,
    'OCR': handle_ocr,
    'SearchFace': handle_search_face,
    'DeleteTempImagesId': handle_delete_temp_images_id,
    'DeleteFaceId': handle_delete_face_id
} 