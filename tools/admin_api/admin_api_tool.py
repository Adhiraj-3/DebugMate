
"""
Admin API tool for fetching user membership and subscription details
"""

import json
import requests
from langchain_core.tools import tool
from core.models import AdminAPIInput
from core.config import EDITION_HEADERS, EDITION_API_URL

@tool(args_schema=AdminAPIInput)
def search_admin_apis(search_context: str, user_identifier: str, plan_type: int) -> str:
    """
    Fetch user membership/subscription details from the admin API.
    Use this tool when you need information about user subscriptions, benefits, plans, or account details.
    plan type - 4 for District pass / District Gold India

    Args:
        search_context: Context about what user information is needed
        user_identifier: The user ID to fetch data for
        plan_type: The plan type to fetch membership details for
    Returns:
        JSON string with user subscription and membership details
    """
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: search_admin_apis")
    print("="*80)
    print(f"Parameters:")
    print(f"  - user_identifier: {user_identifier}")
    print(f"  - plan_type: {plan_type}")
    print(f"  - search_context: {search_context}")
    print("="*80)
    # API Config — URL loaded from .env via core/config.py
    URL = EDITION_API_URL

    def remove_assets_and_images(obj):
        """Clean unnecessary assets/images from API response."""
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                kl = k.lower()
                if kl == "assets":
                    continue
                if any(x in kl for x in ["image", "logo", "banner", "lottie"]):
                    continue
                new_obj[k] = remove_assets_and_images(v)
            return new_obj
        elif isinstance(obj, list):
            return [remove_assets_and_images(i) for i in obj]
        else:
            return obj

    payload = {"user_id": user_identifier, "plan_type": plan_type}
    if user_identifier == "777":
        return "[ADMIN API]\nUser has no membership data."

    try:
        response = requests.post(URL, headers=EDITION_HEADERS, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        cleaned = remove_assets_and_images(data)
        result = f"[ADMIN API - User: {user_identifier}]\n{json.dumps(cleaned, indent=2)}"
        
        # Log the data being sent to LLM
        print("\n📤 DATA SENT TO LLM (Admin API):")
        print("-"*80)
        print(result[:500] + "..." if len(result) > 500 else result)
        print("-"*80)
        print(f"✅ Total length: {len(result)} chars\n")
        
        return result
    except Exception as e:
        error_msg = f"[ADMIN API ERROR] {str(e)}"
        print(f"\n❌ {error_msg}\n")
        return error_msg
