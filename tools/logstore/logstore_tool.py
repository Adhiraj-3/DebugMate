
"""
Logstore tool for searching and retrieving logs with vector database semantic search
"""

import time
import json
import requests
import hashlib
from langchain_core.tools import tool
from core.models import LogstoreInput
from core.config import LOGSTORE_URL, LOGSTORE_DASHBOARD_URL, DEFAULT_HEADERS, get_logtype_from_config, AI_GATEWAY_CONFIG, SERVICE_CONFIGS
from tools.logstore.utils import clean_log_message
from core.ai_gateway_llm import AIGatewayLLM, AIGatewayEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# --- Lazy initialization for Vector DB components ---
# These are initialized on first use to avoid crashing at import time
# if AI Gateway is unreachable or embedding models are unavailable.
# Instead of ChromaDB (incompatible with Python 3.14), we use LangChain's
# InMemoryVectorStore — same semantic search, zero external dependencies.
_vector_stores = {}  # Maps collection_name → InMemoryVectorStore instance
_stored_doc_hashes = {}  # Maps collection_name → set of content hashes (for dedup)
embeddings = None
llm = None
vector_db_enabled = False
_vector_db_initialized = False

# --- Session-scoped embedding persistence ---
# The current session_id is set by agent.py before each invocation via set_session_context().
# Collections are named by session + method (not timestamp), so embeddings persist across
# multiple messages in the same Slack thread. A midnight cleanup purges everything.
_current_session_id = None
# Registry: maps session_id → set of collection_names created in that session
_session_collections = {}

# Embedding model and dimension for AI Gateway
EMBEDDING_MODEL = "text-embed-3-large"
EMBEDDING_DIM = 3072  # text-embed-3-large outputs 3072 dimensions


def initialize_vector_db():
    """
    Lazily initialize AI Gateway embeddings and LLM.
    Embeddings use AI Gateway with text-embed-3-large model (output_dim=3072).
    LLM for prompt refinement also uses AI Gateway.
    Sets vector_db_enabled = True on success, False on failure.
    Safe to call multiple times — only initializes once.
    Uses InMemoryVectorStore (no ChromaDB dependency).
    """
    global embeddings, llm, vector_db_enabled, _vector_db_initialized

    if _vector_db_initialized:
        return vector_db_enabled

    _vector_db_initialized = True
    print("\n🔧 Initializing Vector DB components (lazy)...")

    # Check if AI Gateway config is available
    if not AI_GATEWAY_CONFIG.get("base_url"):
        print("  ❌ AI_GATEWAY_BASE_URL not set. Vector DB disabled.")
        vector_db_enabled = False
        return False

    # Step 1: Initialize embeddings using AI Gateway with text-embed-3-large
    try:
        print(f"  🔄 Trying AI Gateway embedding model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})...")
        candidate = AIGatewayEmbeddings(
            model_name=EMBEDDING_MODEL,
            base_url=AI_GATEWAY_CONFIG["base_url"],
            project_name=AI_GATEWAY_CONFIG["project_name"],
            project_auth_key=AI_GATEWAY_CONFIG["project_auth_key"],
            timeout_ms=AI_GATEWAY_CONFIG.get("timeout_ms", 30000),
            output_dim=EMBEDDING_DIM,
        )
        # Test with a small embed call to verify the model works
        candidate.embed_query("test")
        embeddings = candidate
        print(f"  ✅ AI Gateway embedding model ready: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"  ❌ AI Gateway embedding model {EMBEDDING_MODEL} failed: {e}")
        vector_db_enabled = False
        return False

    # Step 2: Initialize LLM for prompt refinement (still uses AI Gateway)
    try:
        llm = AIGatewayLLM(**AI_GATEWAY_CONFIG)
        print("  ✅ LLM initialized for prompt refinement (via AI Gateway)")
    except Exception as e:
        print(f"  ⚠️ LLM init failed: {e}. Prompt refinement will use fallback.")
        # LLM failure is non-fatal; refine_search_prompt has its own fallback

    vector_db_enabled = True
    print("  ✅ Vector DB fully initialized and enabled\n")
    return True


def set_session_context(session_id: str):
    """
    Set the current session context for logstore operations.
    Called by agent.py before each invocation so that log collections
    are scoped to the Slack thread session.
    """
    global _current_session_id
    _current_session_id = session_id
    if session_id and session_id not in _session_collections:
        _session_collections[session_id] = set()
    print(f"📎 Logstore session context set: {session_id}")


def _build_collection_name(label: str, environment: str = "prod") -> str:
    """
    Build a deterministic, session-scoped collection name.
    Same session + same label + same env = same collection → embeddings accumulate.
    ChromaDB collection names: 3-63 chars, alphanumeric + underscores/hyphens.
    """
    sid = (_current_session_id or "default")[:12]  # Truncate UUID for name safety
    # Sanitize label: keep only alphanumeric and underscores
    safe_label = "".join(c if c.isalnum() or c == "_" else "_" for c in label)[:20]
    safe_env = environment[:4]
    name = f"s_{sid}_{safe_label}_{safe_env}"
    # Ensure 3-63 char limit
    return name[:63] if len(name) >= 3 else name + "_col"


def get_or_create_log_collection(collection_name: str):
    """Get or create an InMemoryVectorStore for logs, registering it in the session."""
    if not initialize_vector_db() or embeddings is None:
        return None
    if collection_name in _vector_stores:
        store = _vector_stores[collection_name]
        doc_count = len(_stored_doc_hashes.get(collection_name, set()))
        print(f"📚 Using existing log collection: {collection_name} (count={doc_count})")
    else:
        store = InMemoryVectorStore(embeddings)
        _vector_stores[collection_name] = store
        _stored_doc_hashes[collection_name] = set()
        print(f"📚 Created new log collection: {collection_name}")
    # Register in session
    if _current_session_id and _current_session_id in _session_collections:
        _session_collections[_current_session_id].add(collection_name)
    return store


def cleanup_all_collections():
    """
    Delete ALL vector stores and clear the session registry.
    Called by the midnight scheduler to free memory.
    """
    global _session_collections
    count = len(_vector_stores)
    _vector_stores.clear()
    _stored_doc_hashes.clear()
    _session_collections.clear()
    print(f"🧹 Midnight cleanup: deleted {count} vector store(s), registry cleared.")
    return count


def refine_search_prompt(user_query: str, rpc_method: str = None, identifier: str = None) -> str:
    """
    Use LLM with config knowledge to refine the user's search intent into a better semantic search query.
    Falls back to a simple concatenated query if LLM is unavailable.
    """
    initialize_vector_db()  # Ensure LLM is initialized (non-fatal if it fails)
    print(f"\n🔍 Refining search prompt using config knowledge...")
    
    # Build context from service configs
    config_context = ""
    if SERVICE_CONFIGS:
        for service_name, config in SERVICE_CONFIGS.items():
            config_context += f"\nService: {config.get('service_name', service_name)}\n"
            
            # Add RPC methods info
            if config.get('rpc_services'):
                config_context += "Available RPC Methods:\n"
                for service_group in config['rpc_services'].values():
                    for method in service_group.get('methods', []):
                        if rpc_method and rpc_method in method['name']:
                            config_context += f"- {method['name']}: {method.get('description', '')}\n"
                            if method.get('essential_for_debugging'):
                                config_context += f"  Essential fields: {', '.join(method['essential_for_debugging'])}\n"
            
            # Add debugging scenarios
            if config.get('common_debugging_scenarios'):
                config_context += "\nCommon Issues:\n"
                for scenario in config['common_debugging_scenarios'][:3]:
                    config_context += f"- {scenario.get('scenario', '')}\n"
                    if scenario.get('check_points'):
                        config_context += f"  Check: {', '.join(scenario['check_points'][:2])}\n"
    
    refinement_prompt = f"""Given the user's search intent and service configuration, generate a refined search query that will help find the most relevant logs.

User's Original Query Context: {user_query}
RPC Method: {rpc_method or 'Not specified'}
Identifier (user_id/order_id): {identifier or 'Not specified'}

Service Configuration Context:
{config_context}

Generate a concise, focused search query (1-2 sentences) that captures:
1. The core issue or information needed
2. Relevant technical terms from the config
3. Key fields or patterns to look for in logs

Refined Search Query:"""
    
    if llm is None:
        print("⚠️ LLM not available for prompt refinement. Using original query.")
        return f"{rpc_method or ''} {identifier or ''} {user_query}".strip()

    try:
        response = llm.invoke(refinement_prompt)
        refined_query = response.content.strip()
        print(f"✅ Refined query: {refined_query}")
        return refined_query
    except Exception as e:
        print(f"⚠️ Failed to refine prompt: {e}. Using original query.")
        return f"{rpc_method or ''} {identifier or ''} {user_query}".strip()


def store_logs_in_vector_db(logs: list, collection_name: str, metadata: dict = None) -> int:
    """
    Store logs in InMemoryVectorStore for semantic search.
    Returns 0 if vector DB is not available (triggers fallback in callers).
    Deduplicates by content hash — same log won't be stored twice.
    
    Args:
        logs: List of log messages (strings)
        collection_name: Name for the vector store
        metadata: Additional metadata to store with logs
        
    Returns:
        Number of NEW logs stored (0 if vector DB unavailable or on error)
    """
    if not logs:
        return 0

    if not initialize_vector_db():
        print("⚠️ Vector DB not available. Skipping vector storage.")
        return 0

    store = get_or_create_log_collection(collection_name)
    if store is None:
        return 0
    
    print(f"\n💾 Storing {len(logs)} logs in vector database...")
    
    # Deduplicate: only add logs we haven't seen before
    new_docs = []
    existing_hashes = _stored_doc_hashes.get(collection_name, set())
    
    for log in logs:
        log_hash = hashlib.md5(log.encode()).hexdigest()
        if log_hash not in existing_hashes:
            cleaned_log = clean_log_message(log)
            new_docs.append(cleaned_log)
            existing_hashes.add(log_hash)
    
    _stored_doc_hashes[collection_name] = existing_hashes
    
    if not new_docs:
        print(f"ℹ️ All {len(logs)} logs already stored (dedup). Skipping.")
        return len(logs)  # Return total count, not 0, since they ARE stored
    
    try:
        # InMemoryVectorStore.add_texts handles embedding internally
        store.add_texts(new_docs)
        
        print(f"✅ Successfully stored {len(new_docs)} new logs (skipped {len(logs) - len(new_docs)} dupes) in: {collection_name}")
        return len(logs)
        
    except Exception as e:
        print(f"❌ Error storing logs in vector database: {e}")
        return 0


def semantic_search_logs(refined_query: str, collection_name: str, n_results: int = 10) -> list:
    """
    Perform semantic search on stored logs using InMemoryVectorStore.
    Returns empty list if vector DB is not available.
    
    Args:
        refined_query: The refined search query
        collection_name: Name of the vector store
        n_results: Number of results to return
        
    Returns:
        List of relevant log messages
    """
    if not initialize_vector_db() or embeddings is None:
        print("⚠️ Vector DB not available for semantic search.")
        return []

    store = _vector_stores.get(collection_name)
    if store is None:
        print(f"⚠️ No vector store found for collection: {collection_name}")
        return []

    try:
        print(f"\n🔎 Performing semantic search with query: '{refined_query}'")
        
        # Use similarity_search which handles embedding + search internally
        results = store.similarity_search(refined_query, k=n_results)
        
        if results:
            relevant_logs = [doc.page_content for doc in results]
            print(f"✅ Found {len(relevant_logs)} relevant logs via semantic search")
            return relevant_logs
        else:
            print("⚠️ No relevant logs found")
            return []
            
    except Exception as e:
        print(f"❌ Error during semantic search: {e}")
        return []

@tool(args_schema=LogstoreInput)
def search_logstore(
    identifier: str,
    rpc_method: str,
    hours_ago: int = 0,
    get_error_logs_only: bool = False,
    environment: str = "stag"
) -> str:
    """
    Gets request and response logs for a particular RPC method from logstore using vector database semantic search.
    Searches a 24-hour time window ending `hours_ago` hours in the past, stores logs in vector DB,
    then performs semantic search to find most relevant logs.

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
        JSON string with semantically relevant logs
    """
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: search_logstore (with Vector DB Semantic Search)")
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

    # Extract just the method name from the RPC path for more flexible matching
    if '/' in rpc_method:
        method_name = rpc_method.split('/')[-1]  # Get last part after /
    elif '.' in rpc_method:
        method_name = rpc_method.split('.')[-1]  # Get last part after .
    else:
        method_name = rpc_method
    
    filters = [
        {
            "column": "message.log.l-method-name",
            "op": "like",
            "values": [method_name],
        },
    ]
    
    # Add identifier filter
    if identifier:
        filters.append({
            "column": "message.log.user-id",
            "op": "like",
            "values": [identifier],
        })
    
    print(f"\n🔍 Search Filters:")
    print(f"   RPC Method (searching for): {method_name}")
    print(f"   Identifier (searching for): {identifier}")

    if get_error_logs_only:
        filters.append({
            "column": "msg",
            "op": "like",
            "values": ["error", "exception", "failed"],
        })

    # Get logtype from config (consistent with other tools)
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

    # Debug: print the full payload being sent
    print(f"\n📦 Logstore API Payload:")
    print(f"   URL: {LOGSTORE_URL}")
    print(f"   logtype: {logtype}")
    print(f"   ts_from: {ts_from} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_from))})")
    print(f"   ts_to: {ts_to} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_to))})")
    print(f"   filters: {json.dumps(filters, indent=2)}")
    print(f"   environment: {environment}")

    try:
        # Step 1: Fetch logs from logstore API
        response = requests.post(
            LOGSTORE_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            timeout=120,
            allow_redirects=True,
        )

        print(f"\n📡 Logstore API Response:")
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")

        if "text/html" in response.headers.get("Content-Type", ""):
            return json.dumps({
                "error": "AUTH_REQUIRED",
                "message": "API returned HTML login page instead of JSON",
            })

        data = response.json()
        
        if not isinstance(data, dict):
            return json.dumps({
                "error": "INVALID_RESPONSE",
                "message": f"Expected dict but got {type(data).__name__}",
                "data_preview": str(data)[:200]
            })
        
        # Debug: print ALL response keys and their types/sizes
        print(f"   Response keys: {list(data.keys())}")
        for key in data.keys():
            val = data[key]
            if isinstance(val, list):
                print(f"   '{key}': list with {len(val)} items")
                if val:
                    print(f"     First item type: {type(val[0]).__name__}")
                    print(f"     First item preview: {str(val[0])[:300]}")
            else:
                print(f"   '{key}': {type(val).__name__} = {str(val)[:300]}")
        
        hits = data.get("hits", [])
        
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
        
        if len(messages) == 0:
            return json.dumps({
                "total_hits": 0,
                "note": "No logs found for the specified criteria",
                "processed_logs": []
            })
        
        # Step 2: Store logs in vector database (session-scoped — persists across thread messages)
        collection_name = _build_collection_name(f"logs_{method_name}", environment)
        metadata = {
            "rpc_method": rpc_method,
            "identifier": identifier,
            "timestamp": time.time()
        }
        
        stored_count = store_logs_in_vector_db(messages, collection_name, metadata)
        
        if stored_count == 0:
            # Fallback to simple deduplication if vector storage fails
            print("⚠️ Vector storage failed, falling back to deduplication")
            unique_logs = list(dict.fromkeys(messages))
            limited_logs = [clean_log_message(log) for log in unique_logs[:10]]
        else:
            # Step 3: Refine search prompt using LLM and config knowledge
            user_query = f"Find logs related to {rpc_method} for {identifier}"
            if get_error_logs_only:
                user_query += " focusing on errors and failures"
            
            refined_query = refine_search_prompt(user_query, rpc_method, identifier)
            
            # Step 4: Perform semantic search to get most relevant logs
            relevant_logs = semantic_search_logs(refined_query, collection_name, n_results=10)
            limited_logs = relevant_logs[:10] if relevant_logs else []
        
        result = {
            "total_hits": len(hits),
            "original_count": len(messages),
            "stored_in_vector_db": stored_count,
            "relevant_logs_count": len(limited_logs),
            "search_method": "vector_db_semantic_search" if stored_count > 0 else "fallback_deduplication",
            "note": f"Showing {len(limited_logs)} most relevant logs via semantic search. Full logs: {LOGSTORE_DASHBOARD_URL}",
            "processed_logs": limited_logs
        }
        
        # Print processed data being sent to LLM
        print("\n" + "="*80)
        print("📤 DATA SENT TO LLM (after semantic search):")
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
    Gets all logs for a particular request_id from logstore using vector database semantic search.
    Searches a 24-hour time window ending `hours_ago` hours in the past.

    Args:
        request_id: The request ID to filter logs
        hours_ago: How many hours back from NOW the search window should END (0 = search last 24 hours, 24 = search 24-48 hours ago)
        environment: Environment to filter logs 'prod', 'stag'
    
    Examples:
        hours_ago=0  → searches [now-24h to now]        (most recent 24 hours)
        hours_ago=24 → searches [now-48h to now-24h]    (24-48 hours ago)
        
    Returns:
        JSON string with semantically relevant logs
    """
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: get_all_logs_for_request_id (with Vector DB Semantic Search)")
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
            "column": "message.log.request_id",
            "op": "like",
            "values": [request_id],
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
        # Step 1: Fetch logs from logstore API
        response = requests.post(
            LOGSTORE_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            timeout=120,
            allow_redirects=True,
        )

        if "text/html" in response.headers.get("Content-Type", ""):
            return json.dumps({
                "error": "AUTH_REQUIRED",
                "message": "API returned HTML login page instead of JSON",
            })

        data = response.json()
        
        if not isinstance(data, dict):
            return json.dumps({
                "error": "INVALID_RESPONSE",
                "message": f"Expected dict but got {type(data).__name__}",
                "data_preview": str(data)[:200]
            })
        
        hits = data.get("hits", [])
        
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
        
        # Handle zero results with helpful context
        if len(messages) == 0:
            result = {
                "total_hits": 0,
                "original_count": 0,
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
        
        # Step 2: Store logs in vector database (session-scoped — persists across thread messages)
        collection_name = _build_collection_name(f"req_{request_id}", environment)
        metadata = {
            "request_id": request_id,
            "identifier": request_id,
            "timestamp": time.time()
        }
        
        stored_count = store_logs_in_vector_db(messages, collection_name, metadata)
        
        if stored_count == 0:
            # Fallback to simple deduplication if vector storage fails
            print("⚠️ Vector storage failed, falling back to deduplication")
            unique_logs = list(dict.fromkeys(messages))
            limited_logs = [clean_log_message(log) for log in unique_logs[:10]]
        else:
            # Step 3: Refine search prompt using LLM and config knowledge
            user_query = f"Find all relevant logs for request_id {request_id}, including request flow, responses, and any errors"
            refined_query = refine_search_prompt(user_query, rpc_method=None, identifier=request_id)
            
            # Step 4: Perform semantic search to get most relevant logs
            relevant_logs = semantic_search_logs(refined_query, collection_name, n_results=10)
            limited_logs = relevant_logs[:10] if relevant_logs else []
        
        result = {
            "total_hits": len(hits),
            "original_count": len(messages),
            "stored_in_vector_db": stored_count,
            "relevant_logs_count": len(limited_logs),
            "search_method": "vector_db_semantic_search" if stored_count > 0 else "fallback_deduplication",
            "note": f"Showing {len(limited_logs)} most relevant logs via semantic search. Full logs: {LOGSTORE_DASHBOARD_URL}",
            "search_criteria": {
                "request_id": request_id,
                "environment": environment
            },
            "processed_logs": limited_logs
        }
        
        # Print processed data being sent to LLM
        print("\n" + "="*80)
        print("📤 DATA SENT TO LLM (after semantic search):")
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
    """
    Checks whether there are any panic logs in last x mins using vector database semantic search
    """
    # Log tool invocation
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: search_panic_logs (with Vector DB Semantic Search)")
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
        # Step 1: Fetch panic logs from logstore API
        response = requests.post(
            LOGSTORE_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            timeout=120,
            allow_redirects=True
        )

        if "text/html" in response.headers.get("Content-Type", ""):
            return json.dumps({
                "error": "AUTH_REQUIRED",
                "message": "API returned HTML login page instead of JSON",
            })

        data = response.json()
        
        if not isinstance(data, dict):
            return json.dumps({
                "error": "INVALID_RESPONSE",
                "message": f"Expected dict but got {type(data).__name__}",
                "data_preview": str(data)[:200]
            })
        
        hits = data.get("hits", [])
        
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
        
        if len(messages) == 0:
            return json.dumps({
                "total_hits": 0,
                "note": "No panic logs found in the specified time window",
                "processed_logs": []
            })
        
        # Step 2: Store logs in vector database (session-scoped — persists across thread messages)
        collection_name = _build_collection_name("panic_logs", "stag")
        metadata = {
            "log_type": "panic",
            "timestamp": time.time()
        }
        
        stored_count = store_logs_in_vector_db(messages, collection_name, metadata)
        
        if stored_count == 0:
            # Fallback to simple deduplication if vector storage fails
            print("⚠️ Vector storage failed, falling back to deduplication")
            unique_logs = list(dict.fromkeys(messages))
            limited_logs = [clean_log_message(log) for log in unique_logs[:10]]
        else:
            # Step 3: Refine search prompt for panic logs
            user_query = f"Find critical panic and crash logs with error details and stack traces"
            refined_query = refine_search_prompt(user_query, rpc_method=None, identifier=None)
            
            # Step 4: Perform semantic search to get most relevant panic logs
            relevant_logs = semantic_search_logs(refined_query, collection_name, n_results=10)
            limited_logs = relevant_logs[:10] if relevant_logs else []
        
        result = {
            "total_hits": len(hits),
            "original_count": len(messages),
            "stored_in_vector_db": stored_count,
            "relevant_logs_count": len(limited_logs),
            "search_method": "vector_db_semantic_search" if stored_count > 0 else "fallback_deduplication",
            "note": f"Showing {len(limited_logs)} most critical panic logs via semantic search. Full logs: {LOGSTORE_DASHBOARD_URL}",
            "processed_logs": limited_logs
        }
        
        # Print processed data being sent to LLM
        print("\n" + "="*80)
        print("📤 DATA SENT TO LLM (after semantic search):")
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
def retrieve_stored_logs(
    search_query: str,
    n_results: int = 10
) -> str:
    """
    Search through previously stored log embeddings WITHOUT fetching new logs from logstore.
    Use this tool when you already have logs from a previous search in this conversation thread
    and want to find specific information within those stored logs using a different query.
    This is faster than fetching logs again and works on the accumulated log embeddings.

    Args:
        search_query: Natural language query to search stored logs (e.g. "error in payment flow", "timeout exceptions")
        n_results: Maximum number of relevant logs to return (default: 10)

    Returns:
        JSON string with semantically relevant logs from previously stored embeddings
    """
    print("\n" + "="*80)
    print("🔧 TOOL CALLED: retrieve_stored_logs (searching existing embeddings)")
    print("="*80)
    print(f"Parameters:")
    print(f"  - search_query: {search_query}")
    print(f"  - n_results: {n_results}")
    print(f"  - session: {_current_session_id}")
    print("="*80)

    if not initialize_vector_db() or embeddings is None:
        return json.dumps({
            "error": "VECTOR_DB_UNAVAILABLE",
            "message": "Vector database is not available. Use search_logstore to fetch and store logs first."
        })

    # Find all collections for the current session
    session_id = _current_session_id or "default"
    session_cols = _session_collections.get(session_id, set())

    if not session_cols:
        # Fallback: scan all vector stores for this session prefix
        sid_prefix = f"s_{session_id[:12]}"
        session_cols = {name for name in _vector_stores if name.startswith(sid_prefix)}

    if not session_cols:
        return json.dumps({
            "status": "NO_STORED_LOGS",
            "message": "No logs have been stored in this session yet. Use search_logstore, get_all_logs_for_request_id, or search_panic_logs to fetch and store logs first.",
            "session_id": session_id
        })

    print(f"📚 Searching across {len(session_cols)} collection(s) in session: {list(session_cols)}")

    # Refine the search query using LLM + config knowledge
    refined_query = refine_search_prompt(search_query)

    # Search across all session collections and merge results
    all_results = []
    for col_name in session_cols:
        try:
            store = _vector_stores.get(col_name)
            if store is None:
                continue

            results = store.similarity_search_with_score(refined_query, k=n_results)

            for doc, score in results:
                all_results.append({
                    "log": doc.page_content,
                    "relevance_score": score,
                    "collection": col_name
                })
        except Exception as e:
            print(f"⚠️ Error searching collection {col_name}: {e}")
            continue

    if not all_results:
        return json.dumps({
            "status": "NO_RELEVANT_LOGS",
            "message": f"No relevant logs found for query: '{search_query}'. Try a different search query or fetch new logs.",
            "collections_searched": list(session_cols),
            "processed_logs": []
        })

    # Sort by relevance and take top n_results
    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_results = all_results[:n_results]

    result = {
        "status": "SUCCESS",
        "search_query": search_query,
        "refined_query": refined_query,
        "collections_searched": list(session_cols),
        "total_matches": len(all_results),
        "returned_count": len(top_results),
        "search_method": "vector_db_semantic_search_existing_embeddings",
        "note": f"Searched {len(session_cols)} existing collection(s) from this session. No new logs were fetched.",
        "processed_logs": [r["log"] for r in top_results]
    }

    # Print processed data being sent to LLM
    print("\n" + "="*80)
    print("📤 DATA SENT TO LLM (from existing embeddings):")
    print("="*80)
    result_preview = result.copy()
    logs = result_preview['processed_logs']
    if len(logs) > 5:
        result_preview['processed_logs'] = logs[:5] + [f"... and {len(logs)-5} more"]
    print(json.dumps(result_preview, indent=2))
    print("="*80 + "\n")

    return json.dumps(result, indent=2)
