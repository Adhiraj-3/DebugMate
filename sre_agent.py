

# ==================== COMPLETE CODE - RUN ALL AT ONCE ====================

# ================== INSTALL DEPENDENCIES (Run this first in Colab) ==================
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'boto3',
        'langgraph',
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✅ {package}")
        except Exception as e:
            print(f"⚠️ Error installing {package}: {e}")
    print("✅ All packages installed!\n")

# Uncomment the line below if running for the first time
# install_packages()

import os
from dotenv import load_dotenv
load_dotenv()

import time
import json
import uuid
import requests
from decimal import Decimal
from typing import List, Annotated
from typing_extensions import TypedDict
from collections import defaultdict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# from langchain_openai import OpenAIEmbeddings  # Commented out - not using embeddings
# from langchain_chroma import Chroma  # Commented out - not needed for now
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict

# Import custom AI Gateway LLM (embeddings use OpenAI directly)
from ai_gateway_llm import AIGatewayLLM

# LangGraph imports for agent and memory
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# AWS Imports
import boto3
from boto3.dynamodb.conditions import Key

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
# For local/dev testing, use dummy project
# For production, get credentials from #generative-ai-oncall
AI_GATEWAY_CONFIG = {
    "base_url": os.environ.get("AI_GATEWAY_BASE_URL", ""),
    "project_name": os.environ.get("AI_GATEWAY_PROJECT_NAME", ""),
    "project_auth_key": os.environ.get("AI_GATEWAY_PROJECT_AUTH_KEY", ""),
    "model_name": os.environ.get("AI_GATEWAY_MODEL_NAME", "gpt-5"),
    "temperature": int(os.environ.get("AI_GATEWAY_TEMPERATURE", "0")),
    "max_completion_tokens": int(os.environ.get("AI_GATEWAY_MAX_COMPLETION_TOKENS", "4096")),
    "timeout_ms": int(os.environ.get("AI_GATEWAY_TIMEOUT_MS", "180000")),
}

# Configure LLM using AI Gateway
llm = AIGatewayLLM(**AI_GATEWAY_CONFIG)

# ================== AWS SETUP ==================
awsSession = boto3.Session(
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    aws_session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
    region_name=os.environ.get("AWS_REGION", "ap-south-1")
)
dynamodb = awsSession.resource("dynamodb")

# ================== LOGSTORE CONFIG ==================
LOGSTORE_URL = os.environ.get("LOGSTORE_URL", "")
LOGSTORE_DASHBOARD_URL = os.environ.get("LOGSTORE_DASHBOARD_URL", "")

DEFAULT_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "content-type": "application/json",
    "origin": os.environ.get("LOGSTORE_ORIGIN", ""),
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

# ================== CHECK IF VECTOR DB EXISTS ==================
# NOTE: Vector DB and code search are currently disabled
# Uncomment this section if you need code search functionality

retriever = None  # Disabled for now

"""
# Original vector DB loading code (commented out)
if os.path.exists(PERSIST_DIR):
    print("📂 Loading existing vector database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    print("✅ Vector database loaded!")
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )
    print("✅ Retriever ready!")
else:
    print("⚠️  Vector database not found at:", PERSIST_DIR)
    print("⚠️  Code search will not be available.")
    retriever = None
"""

# COMMENTED OUT: Vector DB creation code (already have the DB)
"""
else:
    print("🔨 Creating vector database (this will take a while)...")

    base_path = "/Users/aditya.bansal@zomato.com/go/src/github.com/Eternal-District/district-membership-service"

    exts = ["go", "proto", "yaml", "yml", "json", "md", "toml"]
    all_docs = []

    for ext in exts:
        loader = DirectoryLoader(
            base_path,
            glob=f"**/*.{ext}",
            exclude=[
                "**/build/**",
                "**/cache/**",
                "**/vendor/**",
                "**/.git/**",
                "**/node_modules/**",
            ],
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            silent_errors=True,
        )
        docs = loader.load()
        print(f"{ext}: {len(docs)}")
        all_docs.extend(docs)

    print("TOTAL:", len(all_docs))

    for d in all_docs:
        path = d.metadata["source"]
        d.metadata["filename"] = os.path.basename(path)
        d.metadata["language"] = path.split(".")[-1]

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.GO,
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(all_docs)
    print("Chunks:", len(chunks))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"✅ Vector store created with {len(chunks)} chunks")
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )
    print("✅ Retriever ready!")
"""

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
    
    import re
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

def get_logtype_from_config(service_name="district-membership-service"):
    """Get logtype from service config, fallback to default"""
    if service_name in SERVICE_CONFIGS:
        # Extract service name without hyphen for logtype
        return SERVICE_CONFIGS[service_name].get("service_name", "district_membership_service")
    return "district_membership_service"

def check_semantic_similarity_llm(log1, log2, llm):
    """
    Use LLM to determine if two logs are semantically similar.
    Returns a similarity score between 0 and 1.
    """
    from langchain_core.messages import HumanMessage
    
    # Truncate logs for efficiency
    log1_truncated = log1[:800] if len(log1) > 800 else log1
    log2_truncated = log2[:800] if len(log2) > 800 else log2
    
    prompt = f"""Compare these two log messages and determine if they represent the same type of operation/event.

Log 1: {log1_truncated}

Log 2: {log2_truncated}

Are these logs semantically similar (same operation, same flow, similar context)?
Respond with ONLY a number between 0.0 (completely different) and 1.0 (identical meaning).

Score:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()
        
        # Extract number from response
        import re
        match = re.search(r'0?\.\d+|1\.0|1', score_text)
        if match:
            score = float(match.group())
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
        return 0.0
    except Exception as e:
        print(f"⚠️  Error checking similarity: {e}")
        return 0.0

def cluster_similar_logs(logs, llm, similarity_threshold=0.75):
    """
    Cluster similar logs together based on semantic similarity.
    Uses LLM to determine semantic similarity via AI Gateway.
    Returns list of clusters where each cluster is a list of similar logs.
    """
    if not logs:
        return []
    
    print(f"🔗 Clustering {len(logs)} logs based on semantic similarity (threshold: {similarity_threshold})...")
    print(f"   Using LLM-based semantic comparison via AI Gateway")
    
    # For efficiency with large log sets, use a smarter approach:
    # 1. First pass: group by string similarity for candidate pairs
    # 2. Second pass: verify semantic similarity with LLM
    
    clusters = []
    used = set()
    comparison_count = 0
    
    for i, log in enumerate(logs):
        if i in used:
            continue
        
        # Start new cluster with current log
        cluster = [log]
        used.add(i)
        
        # Find semantically similar logs
        # Only compare with a reasonable number of candidates to avoid too many LLM calls
        for j in range(i+1, min(i+50, len(logs))):  # Limit comparison window
            if j in used:
                continue
            
            # Quick string similarity check first (as pre-filter)
            import difflib
            string_sim = difflib.SequenceMatcher(None, log[:200].lower(), logs[j][:200].lower()).ratio()
            
            # If string similarity is very low, skip LLM check
            if string_sim < 0.3:
                continue
            
            # Use LLM for semantic similarity check
            comparison_count += 1
            semantic_similarity = check_semantic_similarity_llm(log, logs[j], llm)
            
            if semantic_similarity >= similarity_threshold:
                cluster.append(logs[j])
                used.add(j)
        
        clusters.append(cluster)
        
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(logs)} logs ({comparison_count} LLM comparisons)...")
    
    print(f"✅ Clustering complete: {len(clusters)} clusters from {len(logs)} logs")
    print(f"   Total LLM comparisons: {comparison_count}")
    
    return clusters

def summarize_log_cluster(cluster, llm, cluster_id=None):
    """Summarize a cluster of similar logs using LLM"""
    if len(cluster) == 1:
        return cluster[0]
    
    # Create a prompt for summarization
    logs_text = "\n---\n".join([f"Log {i+1}: {log[:500]}" for i, log in enumerate(cluster[:10])])
    
    prompt = f"""Summarize these {len(cluster)} similar log messages into a concise summary that captures:
1. The common pattern/operation being performed
2. Key identifiers (user_id, order_id, etc.)
3. Status/outcome if present
4. Any errors or important details

Logs:
{logs_text}

Summary (max 200 words):"""
    
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        
        # Add metadata
        result = f"[CLUSTER of {len(cluster)} logs] {summary}"
        
        # Print the FULL summary to terminal (no truncation)
        cluster_label = f"Cluster {cluster_id}" if cluster_id else "Cluster"
        print(f"\n{'='*80}")
        print(f"📝 {cluster_label} SUMMARY")
        print(f"{'='*80}")
        print(result)
        print(f"{'='*80}\n")
        
        return result
    except Exception as e:
        # Fallback: just take first log if summarization fails
        result = f"[CLUSTER of {len(cluster)} logs] {cluster[0][:300]}..."
        print(f"\n   ⚠️  Cluster {cluster_id if cluster_id else 'N/A'}: Summarization failed, using fallback")
        return result

def recursive_log_summarization(logs, llm, max_tokens_per_chunk=8000, target_tokens=15000):
    """
    Recursively cluster and summarize logs until they fit within target token limit.
    
    Args:
        logs: List of log messages
        llm: LLM instance for summarization
        max_tokens_per_chunk: Max tokens per cluster to send to LLM
        target_tokens: Target total tokens to achieve
    
    Returns:
        List of summarized log entries
    """
    import re
    
    print(f"\n{'='*80}")
    print(f"🔄 Starting recursive summarization with {len(logs)} logs")
    print(f"{'='*80}")
    
    # Step 1: Remove exact duplicates
    unique_logs = list(dict.fromkeys(logs))
    duplicates_removed = len(logs) - len(unique_logs)
    print(f"✅ Removed {duplicates_removed} duplicate logs")
    print(f"📊 Remaining unique logs: {len(unique_logs)}")
    
    if not unique_logs:
        return []
    
    # Rough token estimation (4 chars ≈ 1 token)
    def estimate_tokens(text_list):
        return sum(len(text) for text in text_list) // 4
    
    current_logs = unique_logs
    level = 0
    
    while True:
        level += 1
        current_tokens = estimate_tokens(current_logs)
        
        print(f"\n{'─'*80}")
        print(f"📍 Level {level}: {len(current_logs)} items, ~{current_tokens:,} tokens")
        
        # Check if we've reached target
        if current_tokens <= target_tokens:
            print(f"✅ Reached target! Final token count: ~{current_tokens:,}")
            break
        
        # Check if further reduction is possible
        if len(current_logs) <= 1:
            print(f"⚠️  Cannot reduce further (only 1 item left)")
            break
        
        # Step 2: Cluster similar logs
        print(f"🔍 Clustering similar logs...")
        clusters = cluster_similar_logs(current_logs, llm, similarity_threshold=0.75)
        print(f"📦 Created {len(clusters)} clusters")
        
        # Show cluster distribution
        cluster_sizes = [len(c) for c in clusters]
        if cluster_sizes:
            print(f"   Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        # Step 3: Summarize each cluster
        print(f"✍️  Summarizing clusters...")
        summaries = []
        
        for i, cluster in enumerate(clusters):
            try:
                # Split large clusters into sub-chunks if needed
                chunk_token_limit = max_tokens_per_chunk // 4  # Conservative estimate
                
                if estimate_tokens(cluster) > chunk_token_limit and len(cluster) > 5:
                    # Split cluster into smaller chunks
                    chunk_size = max(1, len(cluster) // ((estimate_tokens(cluster) // chunk_token_limit) + 1))
                    sub_clusters = [cluster[j:j+chunk_size] for j in range(0, len(cluster), chunk_size)]
                    
                    print(f"   Cluster {i+1}: Too large, splitting into {len(sub_clusters)} sub-clusters")
                    
                    for sub_idx, sub_cluster in enumerate(sub_clusters):
                        summary = summarize_log_cluster(sub_cluster, llm, cluster_id=f"{i+1}.{sub_idx+1}")
                        summaries.append(summary)
                else:
                    summary = summarize_log_cluster(cluster, llm, cluster_id=i+1)
                    summaries.append(summary)
                    
                    if (i + 1) % 10 == 0:
                        print(f"   Processed {i+1}/{len(clusters)} clusters...")
            
            except Exception as e:
                print(f"   ⚠️  Error summarizing cluster {i+1}: {e}")
                # Fallback: use first log of cluster
                summaries.append(f"[CLUSTER of {len(cluster)} logs] {cluster[0][:200]}...")
        
        print(f"✅ Generated {len(summaries)} summaries")
        
        # Check if we made progress
        new_tokens = estimate_tokens(summaries)
        if new_tokens >= current_tokens * 0.9:  # Less than 10% reduction
            print(f"⚠️  Minimal reduction achieved, stopping to avoid infinite loop")
            current_logs = summaries
            break
        
        current_logs = summaries
    
    print(f"\n{'='*80}")
    print(f"🎉 Recursive summarization complete!")
    print(f"📊 Final: {len(current_logs)} entries, ~{estimate_tokens(current_logs):,} tokens")
    print(f"{'='*80}\n")
    
    return current_logs

# ================== TOOLS ==================
@tool(args_schema=LogstoreInput)
def search_logstore(
    identifier: str,
    rpc_method: str,
    hours_ago: int = 0,
    get_error_logs_only: bool = False,
    environment: str = "stag"
) -> str:
    """
    Gets request and response logs for a particular RPC method from logstore.
    Searches a 24-hour time window ending `hours_ago` hours in the past.

    Args:
        identifier: Filter for a particular request (user id, order id, trace id, etc)
        rpc_method: The RPC method name to search for
        hours_ago: How many hours back from NOW the search window should END (0 = search last 24 hours from now, 24 = search 24-48 hours ago)
        get_error_logs_only: Whether to filter for error logs only (default: False)
        environment: Environment to filter logs 'prod', 'stag'
    
    Examples:
        hours_ago=0  → searches [now-24h to now]        (most recent 24 hours)
        hours_ago=24 → searches [now-48h to now-24h]    (24-48 hours ago)
        
    Returns:
        JSON string with log count and messages
    """
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: search_logstore")
    print("="*80)
    print(f"Parameters:")
    print(f"  - identifier: {identifier}")
    print(f"  - rpc_method: {rpc_method}")
    print(f"  - hours_ago: {hours_ago}")
    print(f"  - environment: {environment}")
    print(f"  - get_error_logs_only: {get_error_logs_only}")
    print("="*80)
    now_ts = int(time.time())

    ts_to = now_ts - (hours_ago * 60 * 60)
    ts_from = ts_to - (24 * 60 * 60)

    # Search by RPC method and identifier
    # Note: The RPC method in logs often has full path like:
    # "/eternal_district.membership.MembershipAggregatorService/GetApplicableCampaigns"
    # But users provide just "MembershipAggregatorService.GetApplicableCampaigns"
    # So we extract just the method part (last component after last slash or dot)
    
    # Extract just the method name from the RPC path for more flexible matching
    if '/' in rpc_method:
        method_name = rpc_method.split('/')[-1]  # Get last part after /
    elif '.' in rpc_method:
        # If it's like "MembershipAggregatorService.GetApplicableCampaigns"
        # Search for "GetApplicableCampaigns" which should be in the log
        method_name = rpc_method.split('.')[-1]  # Get last part after .
    else:
        method_name = rpc_method
    
    filters = [
        {
            "column": "body.l-method-name",
            "op": "like",
            "values": [method_name],  # Search for just the method name part
        },
    ]
    
    # Add identifier filter - search in common fields where it might appear
    # Note: Some logs use "body.user-id", others use "body.user_id", and some have it in "msg"
    # We'll try body.user-id (most common in your logs)
    if identifier:
        filters.append({
            "column": "body.user-id",  # Try with hyphen first
            "op": "like",
            "values": [identifier],
        })
    
    # Debug: Print the exact filters being sent
    print(f"\n🔍 Search Filters:")
    print(f"   RPC Method (searching for): {method_name}")
    print(f"   Identifier (searching for): {identifier}")
    print(f"   Full filters: {json.dumps(filters, indent=2)}")

    if get_error_logs_only:
        filters.append(
            {
                "column": "msg",
                "op": "like",
                "values": ["error", "exception", "failed"],
            }
        )

    payload = {
        "ts_from": ts_from,
        "ts_to": ts_to,
        "logtype": "district_membership_service",
        "filters": filters,
        "query": "",
        "aggs": {
            "column": "ts",
            "interval_seconds": 240,
        },
        "environment": environment,
    }

    try:
        response = requests.post(
            LOGSTORE_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            timeout=120,  # Increased from 15 to 60 seconds
            allow_redirects=True,
        )

        if "text/html" in response.headers.get("Content-Type", ""):
            return json.dumps({
                "error": "AUTH_REQUIRED",
                "message": "API returned HTML login page instead of JSON",
            })

        data = response.json()
        
        # Debug: Check the structure of data
        if not isinstance(data, dict):
            return json.dumps({
                "error": "INVALID_RESPONSE",
                "message": f"Expected dict but got {type(data).__name__}",
                "data_preview": str(data)[:200]
            })
        
        hits = data.get("hits", [])
        
        # Debug: Check if hits is a list
        if not isinstance(hits, list):
            return json.dumps({
                "error": "INVALID_HITS",
                "message": f"Expected list but hits is {type(hits).__name__}",
                "hits_preview": str(hits)[:200]
            })

        # Extract log messages
        messages = []
        for hit in hits:
            if isinstance(hit, dict) and "msg" in hit:
                messages.append(hit.get("msg"))
        
        print(f"\n📊 Extracted {len(messages)} log messages from hits")
        
        # COMMENTED OUT: Recursive summarization (now using log cleaning + limit)
        # if len(messages) > 50:
        #     print(f"🔄 Applying recursive summarization (logs exceed 50)...")
        #     summarized_logs = recursive_log_summarization(
        #         messages,
        #         llm=llm,
        #         max_tokens_per_chunk=8000,
        #         target_tokens=15000
        #     )
        # else:
        #     print(f"✅ Log count is manageable ({len(messages)} logs), skipping summarization")
        #     summarized_logs = list(dict.fromkeys(messages))
        
        # NEW APPROACH: Clean logs and limit to 5
        print(f"🧹 Cleaning and limiting logs to 5...")
        
        # Remove duplicates first
        unique_logs = list(dict.fromkeys(messages))
        
        # Clean each log to remove unimportant content (no RPC method context for panic logs)
        cleaned_logs = [clean_log_message(log, rpc_method=None) for log in unique_logs]
        
        # Limit to first 5 logs
        limited_logs = cleaned_logs[:5]
        
        result = {
            "total_hits": len(hits),
            "original_count": len(messages),
            "unique_count": len(unique_logs),
            "processed_count": len(limited_logs),
            "note": f"Showing first {len(limited_logs)} cleaned logs. Full logs: {LOGSTORE_DASHBOARD_URL}",
            "processed_logs": limited_logs
        }
        
        # Print processed data being sent to LLM
        print("\n" + "="*80)
        print("📤 DATA SENT TO LLM (after processing):")
        print("="*80)
        result_preview = result.copy()
        if len(limited_logs) > 5:
            result_preview['processed_logs'] = limited_logs[:5] + [f"... and {len(limited_logs)-5} more"]
        print(json.dumps(result_preview, indent=2))
        print("="*80 + "\n")
        
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "JSON_DECODE_ERROR",
            "message": str(e),
            "response_preview": response.text[:200] if 'response' in locals() else "No response"
        })
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })

@tool
def get_all_logs_for_request_id(
    request_id: str,
    hours_ago: int = 0,
    environment: str = "prod"
) -> str:
    """
    Gets all logs for a particular request_id from logstore.
    Searches a 24-hour time window ending `hours_ago` hours in the past.

    Args:
        request_id: The request ID to filter logs
        hours_ago: How many hours back from NOW the search window should END (0 = search last 24 hours, 24 = search 24-48 hours ago)
        environment: Environment to filter logs 'prod', 'stag'
    
    Examples:
        hours_ago=0  → searches [now-24h to now]        (most recent 24 hours)
        hours_ago=24 → searches [now-48h to now-24h]    (24-48 hours ago)
        
    Returns:
        JSON string with log count and messages
    """
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: get_all_logs_for_request_id")
    print("="*80)
    print(f"Parameters:")
    print(f"  - request_id: {request_id}")
    print(f"  - hours_ago: {hours_ago}")
    print(f"  - environment: {environment}")
    print("="*80)
    
    now_ts = int(time.time())

    ts_to = now_ts - (hours_ago * 60 * 60)
    ts_from = ts_to - (24 * 60 * 60)

    filters = [
        {
            "column": "body.request_id",
            "op": "like",
            "values": [
               request_id
            ],
        },
    ]


    # Get logtype from config
    logtype = get_logtype_from_config()
    
    payload = {
        "ts_from": ts_from,
        "ts_to": ts_to,
        "logtype": logtype,
        "filters": filters,
        "query": "",
        "aggs": {
            "column": "ts",
            "interval_seconds": 240,
        },
        "environment": environment,
    }

    try:
        response = requests.post(
            LOGSTORE_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            timeout=120,  # Increased from 15 to 60 seconds
            allow_redirects=True,
        )

        if "text/html" in response.headers.get("Content-Type", ""):
            return json.dumps({
                "error": "AUTH_REQUIRED",
                "message": "API returned HTML login page instead of JSON",
            })

        data = response.json()
        
        # Debug: Check the structure of data
        if not isinstance(data, dict):
            return json.dumps({
                "error": "INVALID_RESPONSE",
                "message": f"Expected dict but got {type(data).__name__}",
                "data_preview": str(data)[:200]
            })
        
        hits = data.get("hits", [])
        
        # Debug: Check if hits is a list
        if not isinstance(hits, list):
            return json.dumps({
                "error": "INVALID_HITS",
                "message": f"Expected list but hits is {type(hits).__name__}",
                "hits_preview": str(hits)[:200]
            })

        # Extract log messages
        messages = []
        for hit in hits:
            if isinstance(hit, dict) and "msg" in hit:
                messages.append(hit.get("msg"))
        
        print(f"\n📊 Extracted {len(messages)} log messages from hits")
        
        # Handle zero results with helpful context from config
        if len(messages) == 0:
            result = {
                "total_hits": 0,
                "original_count": 0,
                "processed_count": 0,
                "status": "NO_LOGS_FOUND",
                "note": f"No logs found for request_id '{request_id}' in the specified time window.",
                "possible_reasons": [
                    "The request_id may be incorrect",
                    "The time window may be too narrow",
                    "Logs may have been rotated out"
                ],
                "suggestions": [
                    "Verify the request_id is correct",
                    "Try expanding time window with hours_ago parameter"
                ],
                "search_params": {
                    "request_id": request_id,
                    "hours_ago": hours_ago,
                    "environment": environment,
                    "time_window": "24 hours"
                },
                "processed_logs": []
            }
            
            print("\n" + "="*80)
            print("📤 DATA SENT TO LLM (No logs found - with context):")
            print("="*80)
            print(json.dumps(result, indent=2))
            print("="*80 + "\n")
            
            return json.dumps(result, indent=2)
        
        # COMMENTED OUT: Recursive summarization (now using log cleaning + limit)
        # if len(messages) > 50:
        #     print(f"🔄 Applying recursive summarization (logs exceed 50)...")
        #     summarized_logs = recursive_log_summarization(
        #         messages,
        #         llm=llm,
        #         max_tokens_per_chunk=8000,
        #         target_tokens=15000
        #     )
        # else:
        #     print(f"✅ Log count is manageable ({len(messages)} logs), skipping summarization")
        #     summarized_logs = list(dict.fromkeys(messages))
        
        # NEW APPROACH: Clean logs and limit to 5
        print(f"🧹 Cleaning and limiting logs to 5...")
        
        # Remove duplicates first
        unique_logs = list(dict.fromkeys(messages))
        
        # Clean each log to remove unimportant content (no RPC method context for request_id search)
        cleaned_logs = [clean_log_message(log, rpc_method=None) for log in unique_logs]
        
        # Limit to first 5 logs
        limited_logs = cleaned_logs[:5]
        
        result = {
            "total_hits": len(hits),
            "original_count": len(messages),
            "unique_count": len(unique_logs),
            "processed_count": len(limited_logs),
            "note": f"Showing first {len(limited_logs)} cleaned logs. Full logs: {LOGSTORE_DASHBOARD_URL}",
            "search_criteria": {
                "request_id": request_id,
                "environment": environment
            },
            "processed_logs": limited_logs
        }
        
        # Print processed data being sent to LLM
        print("\n" + "="*80)
        print("📤 DATA SENT TO LLM (after processing):")
        print("="*80)
        result_preview = result.copy()
        if len(limited_logs) > 5:
            result_preview['processed_logs'] = limited_logs[:5] + [f"... and {len(limited_logs)-5} more"]
        print(json.dumps(result_preview, indent=2))
        print("="*80 + "\n")
        
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "JSON_DECODE_ERROR",
            "message": str(e),
            "response_preview": response.text[:200] if 'response' in locals() else "No response"
        })
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })

@tool
def search_panic_logs(lookback_minutes: int = 10) -> str:
    """Checks whether are any panic logs in last x mins"""
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: search_panic_logs")
    print("="*80)
    print(f"Parameters:")
    print(f"  - lookback_minutes: {lookback_minutes}")
    print("="*80)
    
    now_ts = int(time.time())
    ts_from = now_ts - (lookback_minutes * 60)

    filters = [
        {
            "column": "msg",
            "op": "like",
            "values": ["panic"]
        }
    ]

    # Get logtype from config
    logtype = get_logtype_from_config()

    payload = {
        "ts_from": ts_from,
        "ts_to": now_ts,
        "logtype": logtype,
        "filters": filters,
        "query": "",
        "aggs": {
            "column": "ts",
            "interval_seconds": 240,
        },
        "environment": "stag",
    }

    try:
        response = requests.post(
            LOGSTORE_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            timeout=120,  # Increased from 15 to 60 seconds
            allow_redirects=True
        )

        if "text/html" in response.headers.get("Content-Type", ""):
            return json.dumps({
                "error": "AUTH_REQUIRED",
                "message": "API returned HTML login page instead of JSON",
            })

        data = response.json()
        
        # Debug: Check the structure of data
        if not isinstance(data, dict):
            return json.dumps({
                "error": "INVALID_RESPONSE",
                "message": f"Expected dict but got {type(data).__name__}",
                "data_preview": str(data)[:200]
            })
        
        hits = data.get("hits", [])
        
        # Debug: Check if hits is a list
        if not isinstance(hits, list):
            return json.dumps({
                "error": "INVALID_HITS",
                "message": f"Expected list but hits is {type(hits).__name__}",
                "hits_preview": str(hits)[:200]
            })

        # Extract log messages
        messages = []
        for hit in hits:
            if isinstance(hit, dict) and "msg" in hit:
                messages.append(hit.get("msg"))
        
        print(f"\n📊 Extracted {len(messages)} log messages from hits")
        
        # COMMENTED OUT: Recursive summarization (now using log cleaning + limit)
        # if len(messages) > 50:
        #     print(f"🔄 Applying recursive summarization (logs exceed 50)...")
        #     summarized_logs = recursive_log_summarization(
        #         messages,
        #         llm=llm,
        #         max_tokens_per_chunk=8000,
        #         target_tokens=15000
        #     )
        # else:
        #     print(f"✅ Log count is manageable ({len(messages)} logs), skipping summarization")
        #     summarized_logs = list(dict.fromkeys(messages))
        
        # NEW APPROACH: Clean logs and limit to 5
        print(f"🧹 Cleaning and limiting logs to 5...")
        
        # Remove duplicates first
        unique_logs = list(dict.fromkeys(messages))
        
        # Clean each log to remove unimportant content
        cleaned_logs = [clean_log_message(log) for log in unique_logs]
        
        # Limit to first 5 logs
        limited_logs = cleaned_logs[:5]
        
        result = {
            "total_hits": len(hits),
            "original_count": len(messages),
            "unique_count": len(unique_logs),
            "processed_count": len(limited_logs),
            "note": f"Showing first {len(limited_logs)} cleaned logs. Full logs: {LOGSTORE_DASHBOARD_URL}",
            "processed_logs": limited_logs
        }
        
        # Print processed data being sent to LLM
        print("\n" + "="*80)
        print("📤 DATA SENT TO LLM (after processing):")
        print("="*80)
        result_preview = result.copy()
        if len(summarized_logs) > 5:
            result_preview['processed_logs'] = summarized_logs[:5] + [f"... and {len(summarized_logs)-5} more"]
        print(json.dumps(result_preview, indent=2))
        print("="*80 + "\n")
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "JSON_DECODE_ERROR",
            "message": str(e),
            "response_preview": response.text[:200] if 'response' in locals() else "No response"
        })
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })



@tool(args_schema=CodeSearchInput)
def search_codebase(search_context: str, keywords: List[str] = None) -> str:
    """
    Search the codebase for relevant code, functions, implementations, or documentation.
    Use this tool when you need to understand how something is implemented, find specific functions,
    or learn about the codebase structure.

    Args:
        search_context: Natural language question about the codebase
        keywords: Optional list of important code symbols, function names, or keywords

    Returns:
        Analyzed code explanation with file locations and implementation details
    """
    # Check if retriever is available
    if retriever is None:
        return "[CODEBASE SEARCH]\n⚠️  Code search is currently disabled. Vector database not loaded."
    
    if keywords is None:
        keywords = []

    enhanced_query = search_context
    if keywords:
        enhanced_query += "\nRelevant keywords: " + " ".join(keywords)

    # Retrieve relevant code chunks from vector DB
    docs = retriever.invoke(enhanced_query)

    if not docs:
        return "[CODEBASE SEARCH]\nNo relevant code found for this query."

    # Format results
    context = "\n\n".join([
        f"File: {doc.metadata.get('filename', 'unknown')}\n{doc.page_content}"
        for doc in docs[:5]
    ])

    # Use LLM to analyze the code
    analysis = llm.invoke(f"""
        You are a senior backend engineer analyzing code.

        Code Context:
        {context}

        Question: {search_context}

        Provide a clear explanation of:
        1. What the code does
        2. Where it's located (file/function names)
        3. How it works
        4. Any important implementation details

        Be specific and reference actual code.
    """)

    return f"""
        [CODEBASE SEARCH]
        Query: {search_context}
        Keywords: {', '.join(keywords) if keywords else 'None'}

        {analysis.content}
    """


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
    # API Config
    URL = os.environ.get("EDITION_API_URL", "")

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


@tool(args_schema=AWSInput)
def search_aws_cli(table_name, partition_key: str) -> str:
    """
    Query AWS DynamoDB resources for infrastructure and database data.
    Use this tool when you need information about campaigns, subscriptions, user data stored in DynamoDB.

    Table: prod-district-membership-service

    Partition key formats:
    - For campaigns with gold plan: PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD
    - For campaigns with pass plan: PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD_INDIA
    - For subscription/user data: USER:{user_id}

    Args:
        partition_key: DynamoDB partition key to query
        table_name: DynamoDB table name

    Returns:
        JSON string with queried DynamoDB data
    """
    table = dynamodb.Table(table_name)
    response = table.query(
        KeyConditionExpression=Key("partition_key").eq(partition_key)
    )
    items = response.get('Items', [])

    result = {
        "table": table_name,
        "partition_key": partition_key,
        "items_found": len(items),
        "data": json.loads(json.dumps(items, cls=DecimalEncoder))
    }

    return f"[AWS DynamoDB]\n{json.dumps(result, indent=2)}"

# ================== SYSTEM PROMPT ==================
# Build dynamic system prompt with service configs
def build_system_prompt():
    """Build system prompt with loaded service configurations"""
    
    def escape_braces(text):
        """Escape curly braces for LangChain template"""
        return text.replace('{', '{{').replace('}', '}}')
    
    service_context = ""
    if SERVICE_CONFIGS: 
        service_context = "\n\n# SERVICE CONFIGURATIONS\n\n"
        for service_name, config in SERVICE_CONFIGS.items():
            service_context += f"## {config['service_name'].upper()}\n\n"
            
            # Add log patterns (escape template variables)
            service_context += f"**Log Patterns:**\n"
            service_context += f"- Entry logs: `{escape_braces(config['log_patterns']['entry_pattern'])}`\n"
            service_context += f"- Completion logs: `{escape_braces(config['log_patterns']['completion_pattern'])}`\n"
            service_context += f"- Failure logs: `{escape_braces(config['log_patterns']['failure_pattern'])}`\n"
            service_context += f"- Always log fields: {', '.join(config['log_patterns']['always_log_fields'])}\n\n"
            
            # Add RPC methods with their log patterns
            service_context += f"**Available RPC Methods:**\n\n"
            for service_group_name, service_group in config.get('rpc_services', {}).items():
                service_context += f"### {service_group_name}\n\n"
                for method in service_group.get('methods', []):
                    service_context += f"**{method['name']}**\n"
                    service_context += f"- Description: {escape_braces(method.get('description', 'No description'))}\n"
                    
                    # Add essential request parameters from the actual config structure
                    if method.get('request_params_essential'):
                        service_context += f"- Essential request params: {', '.join(method['request_params_essential'][:5])}\n"
                    
                    # Add essential debugging fields
                    if method.get('essential_for_debugging'):
                        service_context += f"- Essential for debugging: {', '.join(method['essential_for_debugging'])}\n"
                    
                    # Add size reduction estimate
                    if method.get('estimated_size_reduction'):
                        service_context += f"- Log size reduction: {method['estimated_size_reduction']}\n"
                    
                    service_context += "\n"
            
            # Add debugging scenarios
            service_context += f"\n**Common Debugging Scenarios:**\n\n"
            for scenario in config.get('common_debugging_scenarios', []):  # All scenarios
                service_context += f"- **{escape_braces(scenario.get('scenario', 'Unknown scenario'))}**\n"
                if 'relevant_methods' in scenario:
                    service_context += f"  - Methods: {', '.join(scenario['relevant_methods'])}\n"
                if 'check_points' in scenario:
                    service_context += f"  - Check points:\n"
                    for check in scenario['check_points'][:3]:
                        service_context += f"    - {escape_braces(check)}\n"
                if 'log_filters' in scenario:
                    service_context += f"  - Log filters: {', '.join(scenario['log_filters'])}\n"
                service_context += "\n"
            
            # Add database schema
            if config.get('database_schema'):
                service_context += f"\n**Database Schema (DynamoDB):**\n\n"
                db_schema = config['database_schema'].get('dynamodb', {})
                if db_schema.get('table_name'):
                    service_context += f"- Table: `{db_schema['table_name']}`\n"
                if db_schema.get('common_access_patterns'):
                    service_context += f"- Access patterns:\n"
                    for pattern in db_schema['common_access_patterns']:
                        service_context += f"  - {pattern.get('pattern', 'N/A')}: pk=`{pattern.get('pk', '')}`, sk=`{pattern.get('sk', '')}`\n"
                service_context += "\n"
            
            # Add external dependencies
            if config.get('external_dependencies'):
                service_context += f"\n**External Dependencies:**\n\n"
                for dep in config['external_dependencies']:
                    service_context += f"- **{dep.get('name', 'Unknown')}** ({dep.get('type', 'unknown')})\n"
                    if dep.get('operations'):
                        service_context += f"  - Operations: {', '.join(dep['operations'])}\n"
                    if dep.get('failure_impacts'):
                        service_context += f"  - Failure impacts: {'; '.join(dep['failure_impacts'])}\n"
                    if dep.get('log_pattern'):
                        service_context += f"  - Log pattern: `{escape_braces(dep['log_pattern'])}`\n"
                service_context += "\n"
            
            # Add metrics
            if config.get('metrics'):
                service_context += f"\n**Metrics to Monitor:**\n\n"
                metrics = config['metrics']
                if metrics.get('custom_metrics'):
                    service_context += f"- Custom metrics: {', '.join(metrics['custom_metrics'])}\n"
                service_context += "\n"
            
            # Add troubleshooting guide
            if config.get('troubleshooting_guide'):
                service_context += f"\n**Troubleshooting Guide:**\n\n"
                guide = config['troubleshooting_guide']
                for issue, details in guide.items():
                    service_context += f"- **{issue.replace('_', ' ').title()}**:\n"
                    if details.get('check'):
                        for check in details['check'][:3]:
                            service_context += f"  - {escape_braces(check)}\n"
                service_context += "\n"
    
    return f"""
You are an expert SRE (Site Reliability Engineer) AI Assistant with deep knowledge of:
- Distributed systems and microservices architecture
- Log analysis and debugging
- AWS infrastructure (especially DynamoDB)
- Code analysis and understanding
- User account and membership systems

Your job is to help engineers debug issues, understand systems, and find information quickly.

AVAILABLE TOOLS:
1. **search_logstore**: Search logs for specific RPC calls, trace requests, find errors
   - IMPORTANT: Use hours_ago=0 to search the MOST RECENT 24 hours (default behavior)
   - hours_ago=24 searches 24-48 hours ago, NOT the last 24 hours
   - For current issues, ALWAYS use hours_ago=0
2. **search_codebase**: Find and analyze code implementations, functions, and documentation
3. **search_admin_apis**: Get user subscription and membership details
4. **search_aws_cli**: Query DynamoDB for campaigns, subscriptions, and user data
5. **get_all_logs_for_request_id**: Fetch all logs for a specific request ID
6. **search_panic_logs**: Check for panic logs in the last X minutes

{service_context}

# DISTRICT MEMBERSHIP SERVICE CONTEXT (Legacy - use SERVICE CONFIGURATIONS above)

## Service Overview
The District Membership Service is a subscription management platform that handles user memberships, campaigns, benefits, and rewards across Zomato's ecosystem (District by Zomato). It manages subscription lifecycle, benefit tracking, order processing, and integrations with multiple upstream services.

## Core Components

### 1. Subscriptions
- Lifecycle States: ACTIVE, INACTIVE, PENDING, CANCELLED, EXPIRED, REVOKED
- Key Attributes:
  - subscription_id: Unique identifier (int64)
  - campaign_id: Associated campaign
  - user_id: User identifier (string)
  - duration: Start and end date (TimeRange)
  - status: Current subscription state
  - purchase_details: Order and payment information
  - benefit_details: List of benefits with usage tracking
  - auto_renewal_details: Auto-renewal configuration

### 2. Campaigns
- Marketing campaigns that define membership offers and price
- Statuses: ACTIVE, INACTIVE
- Attributes:
  - campaign_id: Unique identifier
  - plan_id: Associated plan
  - start_date / end_date: Campaign validity
  - priority: Numeric priority for display order
  - visibility: Asset configurations for different pages
  - applicability: Criteria rules (location, user segments, etc.)

### 3. Plans
- Defines membership benefits and rules for all benefits
- Key Attributes:
  - plan_id: Unique identifier
  - plan_type: DISTRICT_GOLD_INDIA, DISTRICT_GOLD_UAE, etc.
  - amount: Subscription cost
  - duration: Membership period
  - benefit_list: Associated benefits

### 4. Benefits
- Rewards and discounts available to members
- Benefit Types:
  - CASHBACK: Rewards credited after purchase
  - INSTANT_DISCOUNT: Immediate price reduction
  - EARLY_ACCESS: Priority booking/access
  - CASHBACK_CONVENIENCE_FEE: Refund on convenience fees
  - BUSINESS_VOUCHER: Service-specific vouchers
  - SCRATCH_CARD_REWARD: Gamified rewards
  - FEE_WAIVER: Cancellation/modification fee waiver
  - BUSINESS_INSTANT_DISCOUNT: Business-level discounts
  - PROMO_INSTANT_DISCOUNT: Promotional discounts
  - PROMO_BUY_X_GET_Y: Buy X Get Y offers
  - BUSINESS_ENTITY_VOUCHER: Entity-specific vouchers

- Service Types:
  - MOVIE
  - DINING
  - EVENT
  - SHOPPING
  - DINING_TR (table reservation)
  - DINING_PAY

- Tracking:
  - used_transaction_count: Times benefit used
  - max_transaction_count: Usage limit
  - saved_amount: Total savings accumulated
  - order_ids: Orders where benefit was applied

### 5. Orders
- Track purchase and benefit redemption
- Order Statuses: SUCCESS, CANCELLED, PENDING, FAILED
- Payment Statuses: SUCCESS, FAILED, PENDING
- Refund Statuses: REFUND_NOT_INITIATED, REFUND_INITIATED, REFUND_COMPLETED

## Data Storage (DynamoDB)
Primary Key Patterns:
1. For campaigns with District pass plan: PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD_INDIA
2. For subscription/user data: USER:<user_id>
3. For plan details: PLAN:<plan_id>

## Key RPC Methods (gRPC)
1. GetApplicableCampaigns: Fetch available campaigns, user membership details and evaluate applicability of benefits like promos
2. CreateUserMembership: Create new subscription
3. UpdateUserMembershipStatus: Update subscription state
4. ProcessOrderEvent: Process Order events to update usage count and total savings

## Integration Notes for AI Assistant
When debugging user issues:
1. Check if the user has an active subscription
2. Check campaign rules if user has no subscription
3. Check benefit rules if user is not able to redeem benefits even after having a subscription
4. Check benefit usage limits
5. Dont query logs older than 2 days
6. Dont send more than two log queries per tool call that you return
7. Do not search codebase more than once per session
8. Check for eligibility rules during visibility/applicability issues

CRITICAL INSTRUCTIONS:
- **ALWAYS respect explicit user instructions about which tool to use**
- If user says "search admin api", "check logs", "query database", or "search code", use that specific tool
- If user explicitly mentions a tool or data source, prioritize it over your own judgment
- Analyze the user's question carefully to determine which tools are needed
- You can call multiple tools if needed to gather complete information
- Always extract relevant parameters from the user's question (user IDs, method names, etc.)
- If a required parameter (like user_id) is missing, ask the user or use context from conversation history
- **NEVER return raw JSON or tool output directly to the user**
- **ALWAYS analyze and summarize the tool results in human-readable format**
- Extract key insights, patterns, and actionable information from tool responses
- For user data: highlight active subscriptions, benefits, usage patterns
- For logs: identify errors, patterns, root causes
- For code: explain what it does, where it is, how it works
- Reference specific data points but present them conversationally
- YOU decide when you have enough information to answer the question
- Call tools iteratively and build upon previous results
- When you have sufficient information, provide a clear, concise summary
- After generating your answer, make sure to send which tools are used and what was the summary of each tool call
- Call search_logstore only once per session

## AUTONOMOUS ROOT CAUSE ANALYSIS FRAMEWORK
When debugging issues, perform COMPLETE analysis by gathering ALL necessary data upfront:

**CORE PRINCIPLE: Anticipate what you need to fully answer the question**

**For "WHY" Questions:**
- "Why isn't X working?" → Compare actual behavior vs expected behavior
- "Why can't user do Y?" → Compare what user tried vs what they're allowed to do
- Think: What two things need to be compared to identify the mismatch?

**For "WHAT'S THE ISSUE" Questions:**
- Gather complete context: logs + configuration + user state
- Don't just describe symptoms - identify the root cause
- Think: What data sources contain the full picture?

**AUTONOMOUS DATA GATHERING RULES:**
1. **Extract identifiers** from the question (user IDs, order IDs, etc.) - don't ask for them if present
2. **Identify all data sources needed** for complete analysis (logs, admin API, database, etc.)
3. **Call multiple tools in sequence** to gather comparative data
4. **Perform analysis** only after gathering all necessary data
5. **Present root cause** with evidence from all sources

**REASONING PATTERN FOR ISSUES:**
```
Question Type → Required Data Sources → Comparison → Root Cause

"Why can't user X do Y?"
  → What user tried (logs) + What user is allowed (admin API/config)
  → Compare attempted vs allowed
  → Identify mismatch and explain why it blocks the action

"Why is feature failing?"
  → Actual behavior (logs/errors) + Expected behavior (config/rules)
  → Compare actual vs expected
  → Identify the deviation and explain the failure

"What's wrong with X?"
  → Current state (logs/database) + Valid/expected states (config/rules)
  → Compare current vs valid states
  → Identify invalid state and explain how it occurred
```

**CRITICAL: Complete Analysis in First Response**
- DON'T just describe what you see in one data source
- DON'T wait for user to ask for each piece of information
- DO gather all comparison data autonomously
- DO present complete root cause analysis with evidence

Think step by step:
1. What is the user asking? (Identify the question type: why/what/how/status)
2. If "why" question → What comparison is needed for root cause? (attempted vs allowed, actual vs expected)
3. What tools do I need to gather BOTH sides of comparison?
4. Extract all parameters from question (user IDs, method names, etc.)
5. Call ALL necessary tools to gather complete comparison data
6. Perform comparative analysis and identify root cause
7. Provide comprehensive answer with root cause explanation
8. After answering, detect if user is satisfied:
   - If user says "thanks", "thank you", "that's all", "done", "goodbye", etc. → Call end_conversation
   - If user confirms their question is answered → Call end_conversation
   - If user has no follow-up questions → Wait for next message
   - If user asks a new question → Continue helping

Example good response format:
"User 162434451 has 3 subscriptions:
1. **Active subscription**: District Gold India (₹99), valid until March 24, 2026
   - Benefits: 20% movie discount, 3 free movie tickets (1 used), 2 dining vouchers (2 used)
   - Total savings: ₹282

2. **Cancelled subscription**: Payment failed on Dec 23
3. **Refunded subscription**: Purchased and refunded same day

The user is currently active with unused movie tickets available."
"""

# ================== STATE DEFINITION ==================

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[list, add_messages]


# ================== CREATE AGENT ==================

# Create the base agent
from langchain_core.prompts import ChatPromptTemplate

# Build dynamic system prompt with service configs
SYSTEM_PROMPT = build_system_prompt()

# Create prompt template with system message
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

memory = MemorySaver()

agent = create_react_agent(
    llm,
    tools=[search_logstore, search_codebase, search_admin_apis, search_aws_cli, search_panic_logs],
    checkpointer=memory,
    prompt=prompt
)


# ================== BUILD WORKFLOW ==================

# Build the simple graph - just agent node
workflow = StateGraph(AgentState)

# Add agent node
workflow.add_node("agent", agent)

# Simple flow: START → agent → END
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

print("✅ Agent graph created with routing logic!")

def run_agent_with_session(user_query: str, session_id: str = None) -> tuple:
    """
    Execute the agent with session-based memory.

    Args:
        user_query: The user's question or request
        session_id: Optional session ID for conversation continuity (auto-generated if not provided)

    Returns:
        tuple: (final_response, session_id)
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Configuration for the agent
    # thread_id is used by MemorySaver to track conversation history
    config = {
        "configurable": {
            "thread_id": session_id
        },
        # LLM will decide recursion depth naturally - no hard limit imposed
        "recursion_limit": 100  # Safety limit, but LLM decides when to stop
    }

    print(f"🔄 Session ID: {session_id}")

    # Initialize response variable
    last_ai_message = None

    # Invoke the agent - it will automatically:
    # 1. Load previous conversation from memory (if session exists)
    # 2. Process user query
    # 3. Call tools as needed (LLM decides which and when)
    # 4. Continue until LLM decides it has enough information
    # 5. Save conversation to memory
    for step in agent.stream(
        {"messages": [HumanMessage(content=user_query)]},
        config,
        stream_mode="values",
    ):
        message = step["messages"][-1]
        
        # Stream logging (optional)
        message.pretty_print()

        # Capture AI messages
        if isinstance(message, AIMessage):
            # Skip tool-calling AI messages
            if message.tool_calls:
                continue
            # Candidate final answer
            last_ai_message = message

    # After streaming completes
    if last_ai_message and last_ai_message.content:
        final_response = last_ai_message.content
        print("\n✅ FINAL RESPONSE SELECTED")
        print(f"✅ Length: {len(final_response)} chars")
    else:
        print("⚠️ WARNING: No suitable final AI response found!")
        final_response = "I apologize, but I encountered an issue processing your request. Please try again."

    return final_response, session_id

# ================== INTERACTIVE CHAT ==================
def run_interactive_agent():
    print("=" * 60)
    print("SRE AI AGENT - System Reliability Assistant")
    print("=" * 60)
    print("I can help you with:")
    print("  🔍 Search logs and trace requests")
    print("  💻 Analyze codebase and implementations")
    print("  👤 Get user subscription details")
    print("  ☁️  Query AWS DynamoDB data")
    print("\n🎯 The LLM controls when the conversation ends!")
    print("=" * 60)
    
    # Get initial user query
    print("\n")
    user_input = input("You: ").strip()
    
    if not user_input:
        print("No query provided. Exiting.")
        return
    
    # Generate session ID for conversation persistence
    session_id = str(uuid.uuid4())
    print(f"\n🔄 Session ID: {session_id}\n")
    
    # Configuration for the agent
    config = {
        "configurable": {
            "thread_id": session_id
        },
        "recursion_limit": 100
    }
    
    try:
        conversation_active = True
        
        while conversation_active:
            print("[🤖 Agent thinking...]\n")
            
            # Invoke agent with user input
            result = agent.invoke(
                {"messages": [("user", user_input)]},
                config=config
            )
            
            # Get the last AI message
            messages = result.get("messages", [])
            last_ai_message = None
            
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'ai' and hasattr(msg, 'content') and msg.content:
                    last_ai_message = msg
                    break
            
            if last_ai_message:
                content = last_ai_message.content
                
                # Check if LLM wants to end conversation
                if "ROUTE: END" in content:
                    display_content = content.replace("ROUTE: END", "").strip()
                    if display_content:
                        print(f"🤖: {display_content}\n")
                    print("\n" + "=" * 60)
                    print("✅ Conversation Complete!")
                    print("=" * 60)
                    print("The LLM detected you're satisfied and ended the conversation.")
                    conversation_active = False
                elif "ROUTE: CONTINUE" in content:
                    display_content = content.replace("ROUTE: CONTINUE", "").strip()
                    if display_content:
                        print(f"🤖: {display_content}\n")
                    # Get next user input
                    user_input = input("You: ").strip()
                    if not user_input or user_input.lower() in ['exit', 'quit']:
                        print("\n👋 Goodbye!")
                        conversation_active = False
                else:
                    # No routing instruction - display and continue
                    print(f"🤖: {content}\n")
                    user_input = input("You: ").strip()
                    if not user_input or user_input.lower() in ['exit', 'quit']:
                        print("\n👋 Goodbye!")
                        conversation_active = False
        
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)

