
"""
Configuration module for SRE Agent
All secrets, URLs, headers, and tokens are loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""

import json
import os
from dotenv import load_dotenv

# Load .env if not already loaded (safe to call multiple times)
load_dotenv()

# ================== CONFIG ==================
PERSIST_DIR = "./code_vector_db"

# ================== SERVICE CONFIG ==================
# Load service configuration for better debugging context
def load_service_config(service_name):
    """Load service configuration from JSON file"""
    config_path = f"configs/{service_name}.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Invalid JSON in config file: {e}")
        return None

# Load available service configs
SERVICE_CONFIGS = {}
try:
    district_config = load_service_config("district_membership_service")
    if district_config:
        SERVICE_CONFIGS["district-membership-service"] = district_config
        print(f"✅ Loaded config for: district-membership-service")
except Exception as e:
    print(f"⚠️  Error loading service configs: {e}")

# ================== AI GATEWAY CONFIG ==================
AI_GATEWAY_CONFIG = {
    "base_url": os.environ.get("AI_GATEWAY_BASE_URL", ""),
    "project_name": os.environ.get("AI_GATEWAY_PROJECT_NAME", ""),
    "project_auth_key": os.environ.get("AI_GATEWAY_PROJECT_AUTH_KEY", ""),
    "model_name": os.environ.get("AI_GATEWAY_MODEL_NAME", "gpt-5"),
    "temperature": int(os.environ.get("AI_GATEWAY_TEMPERATURE", "0")),
    "max_completion_tokens": int(os.environ.get("AI_GATEWAY_MAX_COMPLETION_TOKENS", "4096")),
    "timeout_ms": int(os.environ.get("AI_GATEWAY_TIMEOUT_MS", "180000")),
}

# ================== OPENAI EMBEDDINGS CONFIG ==================
# Direct OpenAI API key used ONLY for creating embeddings (not routed through AI Gateway)
OPENAI_EMBEDDINGS_API_KEY = os.environ.get("OPENAI_EMBEDDINGS_API_KEY", "")

# ================== APPROVAL GATE CONFIG ==================
# Comma-separated list of Slack user IDs who can approve queries by reacting
# Example: "U12345678,U87654321"
WHITELISTED_APPROVER_IDS = [
    uid.strip()
    for uid in os.environ.get("WHITELISTED_APPROVER_IDS", "").split(",")
    if uid.strip()
]
# Emoji that approvers must react with to approve a query (without colons)
APPROVAL_EMOJI = os.environ.get("APPROVAL_EMOJI", "white_check_mark")

# ================== AWS CONFIG ==================
AWS_CONFIG = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_session_token": os.environ.get("AWS_SESSION_TOKEN", ""),
    "region_name": os.environ.get("AWS_REGION", "ap-south-1"),
}

# ================== LOGSTORE CONFIG ==================
LOGSTORE_URL = os.environ.get("LOGSTORE_URL", "")

LOGSTORE_ORIGIN = os.environ.get("LOGSTORE_ORIGIN", "")
LOGSTORE_DASHBOARD_URL = os.environ.get("LOGSTORE_DASHBOARD_URL", "")

DEFAULT_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "content-type": "application/json",
    "origin": LOGSTORE_ORIGIN,
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "Authorization": os.environ.get("LOGSTORE_AUTH_TOKEN", ""),
    "cookie": os.environ.get("LOGSTORE_COOKIE", ""),
}

# ================== EDITION DASHBOARD CONFIG ==================
EDITION_API_URL = os.environ.get("EDITION_API_URL", "")
EDITION_ORIGIN = os.environ.get("EDITION_ORIGIN", "")

EDITION_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": EDITION_ORIGIN,
    "Referer": f"{EDITION_ORIGIN}/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Cookie": os.environ.get("EDITION_COOKIE", ""),
}


def get_logtype_from_config(service_name="district-membership-service"):
    """Get logtype from service config, fallback to default.
    Logstore API requires underscores in logtype (hyphens are interpreted as minus operators).
    """
    if service_name in SERVICE_CONFIGS:
        raw_name = SERVICE_CONFIGS[service_name].get("service_name", "district_membership_service")
        # Convert hyphens to underscores — logstore treats hyphens as subtraction
        return raw_name.replace("-", "_")
    return "district_membership_service"
