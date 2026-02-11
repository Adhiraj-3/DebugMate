
"""
Utility functions for log processing and analysis
"""

import re
from core.config import SERVICE_CONFIGS

# ================== HELPER FUNCTIONS ==================
def clean_log_message(log_message, keep_fields=None, rpc_method=None):
    """
    Clean and compress log messages by removing unimportant content.
    Uses config-based field dropping rules for specific RPC methods.
    
    Args:
        log_message: Raw log message string
        keep_fields: List of important fields to preserve (from config)
        rpc_method: RPC method name to apply method-specific dropping rules
    
    Returns:
        Cleaned log message with only essential information
    """
    if not log_message or not isinstance(log_message, str):
        return log_message
    
    # Load droppable fields from config based on RPC method
    droppable_fields = []
    if rpc_method and SERVICE_CONFIGS:
        for service_name, config in SERVICE_CONFIGS.items():
            for service_group_name, service_group in config.get('rpc_services', {}).items():
                for method in service_group.get('methods', []):
                    if rpc_method in method.get('name', '') or rpc_method in method.get('full_path', ''):
                        # Get droppable fields
                        droppable_config = method.get('request_params_droppable', {})
                        always_drop = droppable_config.get('always_drop', [])
                        for drop_item in always_drop:
                            field = drop_item.get('field', '')
                            # Convert nested field notation to regex pattern
                            # e.g., "fact_data.cart_details.cart_items" -> pattern for matching
                            field_pattern = field.replace('.', r'[.:_]').replace('[', r'\[').replace(']', r'\]')
                            droppable_fields.append(field_pattern)
                        
                        # Get response droppable fields
                        response_droppable = droppable_config.get('response_droppable', [])
                        for drop_item in response_droppable:
                            field = drop_item.get('field', '')
                            field_pattern = field.replace('.', r'[.:_]').replace('[', r'\[').replace(']', r'\]')
                            droppable_fields.append(field_pattern)
                        
                        # Get essential fields to keep
                        if not keep_fields:
                            keep_fields = method.get('essential_for_debugging', [])
                        break
    
    # Default important fields from config if not found
    if keep_fields is None:
        keep_fields = [
            "request-id", "user-id", "l-method-name", "l-start-time",
            "l-end-time", "l-latency", "service-order-id", "error",
            "subscription_id", "order_id", "campaign_id", "user_id",
            # Location fields for whitelisting analysis
            "city_id", "country_id", "p_city_id",
            "user_details.location_details.city_id",
            "user_details.location_details.country_id",
            "business_details.location_details.city_id",
            "business_details.location_details.country_id",
            "business_details.location_details.p_city_id",
            "location_details.city_id", "location_details.country_id",
            # Service and plan type for context
            "service_type", "plan_type", "plan_types"
        ]
    
    # Common stopwords to remove (non-technical words)
    stopwords = {
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can',
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under',
        'it', 'its', 'itself', 'they', 'them', 'their', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'we'
    }
    
    # Remove excessive backslashes and escape sequences
    cleaned = log_message.replace('\\n', ' ').replace('\\t', ' ').replace('\\', '')
    
    # Remove extra quotes
    cleaned = cleaned.replace('""', '"').replace("''", "'")
    
    # Keep only ASCII printable characters
    cleaned = ''.join(char if ord(char) < 128 else ' ' for char in cleaned)
    
    # IMPORTANT: Extract and preserve important fields FIRST (before dropping)
    preserved_parts = []
    
    # Check for important field patterns in the original cleaned message
    for field in keep_fields:
        # Match patterns like "field_name: value" or "field_name=value"
        # Also match nested fields like "location_details.city_id"
        field_variants = [
            field,
            field.replace('_', '.'),  # Handle field_name vs field.name
            field.replace('.', '_'),  # Handle field.name vs field_name
            field.split('.')[-1] if '.' in field else field  # Get last part of nested field
        ]
        
        for variant in field_variants:
            pattern = rf'({variant}[\s:=]+[^\s,;|]+)'
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            preserved_parts.extend(matches)
    
    # NOW drop fields based on config (after preserving essentials)
    if droppable_fields:
        for field_pattern in droppable_fields:
            # Remove entire field and its value (handles various formats)
            # Matches: "field": {...}, field:{...}, field=..., etc.
            patterns = [
                rf'{field_pattern}\s*[:=]\s*\{{[^}}]*\}}',  # JSON object
                rf'{field_pattern}\s*[:=]\s*\[[^\]]*\]',     # Array
                rf'{field_pattern}\s*[:=]\s*"[^"]*"',        # String value
                rf'{field_pattern}\s*[:=]\s*[^\s,;|]+',      # Simple value
            ]
            for pattern in patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Extract error messages (always important)
    error_patterns = [
        r'(error[:\s]+[^|]+)',
        r'(failed[:\s]+[^|]+)',
        r'(exception[:\s]+[^|]+)',
        r'(unable to[^|]+)'
    ]
    for pattern in error_patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        preserved_parts.extend(matches)
    
    # If we have preserved parts, use those; otherwise clean the whole message
    if preserved_parts:
        cleaned = ' | '.join(set(preserved_parts))
    
    # Remove stopwords from remaining text
    words = cleaned.split()
    filtered_words = [
        word for word in words
        if word.lower() not in stopwords and len(word) > 2
    ]
    cleaned = ' '.join(filtered_words)
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Limit length to avoid extremely long logs
    if len(cleaned) > 500:
        cleaned = cleaned[:500] + '...'
    
    return cleaned
