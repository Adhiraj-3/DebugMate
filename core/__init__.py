
"""
Core package - shared modules for the SRE AI Agent.

Re-exports commonly used symbols for convenience:
    from core import AI_GATEWAY_CONFIG, AIGatewayLLM, ...
"""

from core.config import (
    AI_GATEWAY_CONFIG,
    AWS_CONFIG,
    LOGSTORE_URL,
    DEFAULT_HEADERS,
    EDITION_HEADERS,
    SERVICE_CONFIGS,
    OPENAI_EMBEDDINGS_API_KEY,
    WHITELISTED_APPROVER_IDS,
    APPROVAL_EMOJI,
    get_logtype_from_config,
)
from core.ai_gateway_llm import AIGatewayLLM, AIGatewayEmbeddings
from core.models import (
    LogstoreInput,
    CodeSearchInput,
    AdminAPIInput,
    AWSInput,
    DecimalEncoder,
)
from core.prompt import build_system_prompt
from core.utils import sanitize_pii
