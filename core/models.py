
"""
Pydantic models and input schemas for SRE Agent tools
"""

from typing import List
from pydantic import BaseModel, Field
from decimal import Decimal
import json

# ================== TOOL INPUT SCHEMAS ==================
class CodeSearchInput(BaseModel):
    """Input schema for code search tool"""
    search_context: str = Field(..., description="Natural language question about the codebase")
    keywords: List[str] = Field(
        default=[],
        description="Important code symbols, function names, or keywords to help with search"
    )

class LogstoreInput(BaseModel):
    """Input schema for logstore search tool"""
    identifier: str = Field(..., description="Identifier to filter logs (user id, order id, trace id, etc)")
    rpc_method: str = Field(..., description="RPC method name to search for")
    hours_ago: int = Field(default=0, description="How many hours ago the 24h window should end (0 = now)")
    get_error_logs_only: bool = Field(default=False, description="Whether to filter for error logs only")
    environment: str = Field(default="prod", description="Environment to filter logs ('prod', 'stag')")

class AdminAPIInput(BaseModel):
    """Input schema for admin API tool"""
    search_context: str = Field(..., description="Context about what user information is needed")
    user_identifier: str = Field(..., description="User ID to fetch membership details for")
    plan_type: int = Field(..., description="Plan type to fetch membership details for")

class AWSInput(BaseModel):
    """Input schema for AWS DynamoDB tool"""
    partition_key: str = Field(..., description="DynamoDB partition key (e.g., 'USER:123' or 'PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD')")
    table_name: str = Field(..., description="DynamoDB table name")

# ================== HELPER CLASSES ==================
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)
