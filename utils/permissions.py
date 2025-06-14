
import json
from typing import Dict, Any, Optional, Union, List

def check_record_edit_permission(record: Dict[str, Any], permission: Dict[str, Any], user_id: str) -> bool:

    if not permission:
        return False
        
    owners_write = permission.get('owners_write', [])
    child_users = permission.get('child_users', [])
    sharing_access = permission.get('sharingAccess')
    
    is_public_share_with_edit = sharing_access in [1, 2]
    is_owner = str(record.get('smownerid')) == str(user_id)
    is_multiowner_write = any(str(id) == str(record.get('smownerid')) for id in record.get('multiowner_w', []))
    is_shared_write = any(str(id) == str(record.get('smownerid')) for id in owners_write)
    is_child = any(str(id) == str(record.get('smownerid')) for id in child_users)
    
    is_updateable = permission.get('updateable', False)
    check_editable = sharing_access is None or is_public_share_with_edit or \
                    is_owner or is_multiowner_write or is_shared_write or is_child
    
    return is_updateable and check_editable

def create_permission_filter(permission: Dict[str, Any], user_id: str) -> str:
    if not permission:
        return ""
        
    owners = permission.get('owners', [])
    sharing_access = permission.get('sharingAccess')
    
    is_public_share = sharing_access in [0, 1, 2]
    
    if not is_public_share and sharing_access:
        filter_permission = f"(smownerid IN [{owners}, {user_id}] OR multiowner_r IN [{user_id}] OR multiowner_w IN [{user_id}])"
        
        if sharing_access == 8:
            filter_permission = f"{filter_permission} AND (smownerid = '{user_id}')"
            
        return filter_permission
    
    return ""

def validate_module_permissions(module_info: Dict[str, Any], is_related_list: bool = False) -> Dict[str, Any]:

    if not module_info or 'permission' not in module_info:
        return {"isValid": False, "error": "Permission denied"}
        
    permission = module_info['permission']
    
    has_view_permission = permission.get('listViewable', False) or \
                         (permission.get('detailAndSelect', False) and is_related_list)
                         
    if not has_view_permission:
        return {"isValid": False, "error": "Permission denied"}
        
    return {"isValid": True, "permission": permission}

def is_user_admin(module_info: Dict[str, Any]) -> bool:
    if not module_info or 'permission' not in module_info:
        return False
        
    permission = module_info['permission']
    
    return all([
        permission.get('listViewable', False),
        permission.get('detailView', False),
        permission.get('updateable', False),
        permission.get('createable', False),
        permission.get('deleteable', False)
    ])

def check_gps_permission(gps_record: Dict[str, Any], user_id: str, module_info: Dict[str, Any]) -> bool:
    if module_info and is_user_admin(module_info):
        return True
        
    employees_allow = gps_record.get('employees_allow', '').split(',') if gps_record.get('employees_allow') else []
    return str(user_id) in employees_allow and gps_record.get('status') == '1'

async def get_module_info(redis_client, user_role: str, module_name: str) -> Optional[Dict[str, Any]]:
    if not redis_client:
        return None
        
    redis_module_info = await redis_client.hget(f"ERP:ModuleInfo:{user_role}", module_name)
    if redis_module_info:
        return json.loads(redis_module_info)
    return None
