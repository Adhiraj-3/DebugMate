
# SRE AI Agent

An AI-powered Site Reliability Engineering assistant that integrates with Slack to help engineers debug issues, search logs, analyze code, query AWS DynamoDB, and check user subscription data. Uses a **RAG (Retrieval Augmented Generation)** architecture with persistent session-scoped log embeddings.

---

## Architecture

```
┌─────────────┐     Socket Mode     ┌──────────────────┐
│  Slack Users │◄──────────────────►│  bot/slack_bot.py │  (Entry Point)
└─────────────┘                     └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │ Midnight Cleanup  │  (threading.Timer)
                                    │ Scheduler         │  Fires at 00:00 daily
                                    └────────┬─────────┘
                                             │
                                             ▼
                                      ┌────────────┐
                                      │  agent.py   │  LangGraph ReAct Agent
                                      │             │  + MemorySaver (sessions)
                                      └──────┬─────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
                   │  prompt.py  │   │  config.py   │   │ ai_gateway_  │
                   │  (system    │   │  (settings,  │   │ llm.py       │
                   │   prompt)   │   │   configs)   │   │ (LLM + Embed)│
                   └─────────────┘   └─────────────┘   └──────────────┘
                          │
              ┌───────────┼───────────┬───────────┬────────────┐
              ▼           ▼           ▼           ▼            ▼
        ┌───────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐
        │ logstore  │ │codebase│ │admin_  │ │ aws_   │ │ retrieve_  │
        │ _tool.py  │ │_tool.py│ │api_tool│ │tool.py │ │ stored_logs│
        └─────┬─────┘ └────────┘ └────────┘ └────────┘ └──────┬─────┘
              │                                                │
              ▼                                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │              ChromaDB (In-Memory Vector DB)             │
        │  Session-scoped collections with persistent embeddings  │
        │  Purged daily at midnight                               │
        └─────────────────────────────────────────────────────────┘
```

---

## How the Bot Runs — Complete Flow

### Step 1: Bot Startup

```bash
python bot/slack_bot.py
```

When the bot starts:
1. **Environment variables loaded** from `.env` via `dotenv`
2. **Slack Bolt app** initialized with `SLACK_BOT_TOKEN`
3. **Agent module imported** → `agent.py` creates the LangGraph ReAct agent with:
   - `AIGatewayLLM` as the LLM (Zomato's internal AI Gateway)
   - Dynamic system prompt built from service configs in `configs/`
   - `MemorySaver` for conversation history
   - 6 tools registered: `search_logstore`, `get_all_logs_for_request_id`, `search_panic_logs`, `search_codebase`, `search_admin_apis`, `search_aws_cli`, `retrieve_stored_logs`
4. **Midnight cleanup scheduler** starts — uses `threading.Timer` to fire `cleanup_all_collections()` at 00:00 daily
5. **Socket Mode** connection established to Slack

### Step 2: User Sends a Message

A user can interact via:
- **Direct message** to the bot
- **@mention** in a channel: `@SRE_Agent check logs for user 123`
- **Slash command**: `/sre check logs for user 123`

### Step 3: Message Routing

```
Slack Event → handle_app_mention / handle_direct_message / handle_sre_command
           → process_agent_query (in a background thread)
           → run_agent_with_session(query, session_id)
```

- The bot uses **`thread_ts`** (Slack thread timestamp) as the session key
- First message in a thread → new `session_id` (UUID) created
- Follow-up messages in the same thread → reuses the existing `session_id`
- Max 100 concurrent sessions, FIFO eviction of oldest

### Step 4: Agent Processes the Query

Inside `run_agent_with_session()`:

1. **Session context set**: `set_session_context(session_id)` tells the logstore tool which session is active (for scoping ChromaDB collections)
2. **System prompt** includes:
   - Service knowledge from JSON configs (RPC methods, error patterns, debugging scenarios)
   - Tool descriptions and usage instructions
   - SRE best practices
3. **LLM decides** which tools to call based on the query
4. **ReAct loop**: The agent iterates — calls tools, reads results, decides if it needs more info, calls more tools, until it has enough to answer

### Step 5: Tool Execution (Logstore Example)

When the LLM decides to call `search_logstore`:

```
┌──────────────────────────────────────────────────────────┐
│ 1. FETCH: Query logstore API for logs matching filters   │
│    (RPC method, user_id, time window, environment)       │
│                                                          │
│ 2. STORE: Logs → clean → embed → upsert into ChromaDB   │
│    Collection name: s_<session>_logs_<method>_<env>      │
│    (session-scoped, deterministic, accumulates over time)│
│                                                          │
│ 3. REFINE: LLM + service config → refined search query  │
│    Uses knowledge of RPC methods, common errors, etc.    │
│                                                          │
│ 4. SEARCH: Semantic search on the collection             │
│    Returns top 10 most relevant logs                     │
│                                                          │
│ 5. RETURN: Relevant logs sent back to the LLM            │
└──────────────────────────────────────────────────────────┘
```

### Step 6: Follow-Up Messages (RAG Persistence)

When the user sends another message **in the same Slack thread**:

**Scenario A — LLM wants more logs:**
- Calls `search_logstore` again (maybe different method/time window)
- New logs are **upserted** into the same session-scoped collection
- Duplicate logs (same content hash) are safely skipped via `upsert()`
- Semantic search now runs on the **accumulated** dataset

**Scenario B — LLM wants to search existing logs differently:**
- Calls `retrieve_stored_logs` with a new search query
- Searches **all existing collections** in the current session
- No network calls to logstore API — pure vector DB search
- Fast, efficient, uses already-computed embeddings

**Scenario C — LLM has enough context:**
- Uses conversation memory (MemorySaver) + previous tool results
- Responds directly without calling any tools

### Step 7: Response Delivery

- Agent's final response formatted for Slack (markdown conversion)
- Long responses split into chunks (Slack's 4000 char limit)
- Posted as a reply in the original thread

### Step 8: Midnight Cleanup

At 00:00 local time every day:
1. **All ChromaDB collections deleted** — frees memory
2. **Session registry cleared** — `_session_collections` emptied
3. **Slack thread sessions cleared** — `thread_sessions` dict emptied
4. Next cleanup scheduled for following midnight
5. Uses `threading.Timer` (stdlib) — no external scheduler dependencies

---

## RAG Architecture Deep Dive

### Embedding Persistence Model

```
Session = Slack Thread (thread_ts → session_id UUID)

Per session, collections accumulate:
  s_<sid>_logs_GetMembership_prod    ← search_logstore calls
  s_<sid>_req_abc123def_prod         ← get_all_logs_for_request_id calls
  s_<sid>_panic_logs_stag            ← search_panic_logs calls

Same session + same method + same env = SAME collection
  → Embeddings accumulate across multiple messages
  → upsert() prevents duplicates (content-hash based IDs)
  → retrieve_stored_logs searches ALL session collections
```

### Lazy Initialization with Fallbacks

ChromaDB, embeddings, and LLM are **not** initialized at import time. Instead:

```python
initialize_vector_db()  # Called on first tool use
```

1. ChromaDB client created (in-memory, always succeeds)
2. Embedding model tried in fallback order:
   - `text-embedding-3-large` (1536 dim) → preferred
   - `text-embedding-3-small` (1536 dim) → fallback
   - `text-embedding-ada-002` (1536 dim) → last resort
3. If ALL embedding models fail → `vector_db_enabled = False` → tools fall back to simple deduplication (no semantic search)
4. LLM initialized for prompt refinement (non-fatal if it fails)

### Collection Naming Strategy

```python
# OLD (before): Timestamp-based — new collection every call, no persistence
collection_name = f"logs_{method}_{env}_{int(time.time())}"

# NEW (after): Session-scoped — deterministic, accumulates embeddings
collection_name = _build_collection_name(f"logs_{method}", environment)
# → "s_abc123def456_logs_GetMembership_prod"
```

---

## Project Structure

```
├── agent.py                        # LangGraph ReAct agent + session management
├── core/                           # Shared internal library
│   ├── __init__.py                 # Re-exports for convenience imports
│   ├── ai_gateway_llm.py          # Custom LangChain LLM & Embeddings (AI Gateway)
│   ├── config.py                   # Configuration: API keys, URLs, service configs
│   ├── models.py                   # Pydantic input schemas for tools
│   └── prompt.py                   # Dynamic system prompt builder
├── bot/
│   ├── __init__.py
│   └── slack_bot.py                # Slack bot entry point + midnight scheduler
├── tools/                          # Each tool is its own sub-package
│   ├── __init__.py                 # Re-exports all tools + utilities
│   ├── logstore/                   # Log search + RAG pipeline
│   │   ├── __init__.py
│   │   ├── logstore_tool.py        # 4 tools + vector DB management
│   │   └── utils.py                # Log cleaning & field filtering
│   ├── codebase/
│   │   ├── __init__.py
│   │   └── codebase_tool.py        # Codebase search
│   ├── admin_api/
│   │   ├── __init__.py
│   │   └── admin_api_tool.py       # Admin API queries
│   └── aws/
│       ├── __init__.py
│       └── aws_tool.py             # DynamoDB queries
├── configs/
│   └── district_membership_service.json  # Service config for prompt building
├── legacy/
│   └── sre_agent.py               # Original monolithic agent (reference only)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Tools Reference

| Tool | Description | Network Call? |
|------|-------------|:---:|
| `search_logstore` | Search RPC request/response logs, store embeddings, semantic search | ✅ Logstore API |
| `get_all_logs_for_request_id` | Get all logs for a specific request ID, store embeddings | ✅ Logstore API |
| `search_panic_logs` | Check for panic/crash logs in recent time window | ✅ Logstore API |
| `retrieve_stored_logs` | Search through previously stored log embeddings (no re-fetch) | ❌ Local only |
| `search_codebase` | Semantic search across the codebase | ✅ Codebase API |
| `search_admin_apis` | Query admin/internal APIs for user data | ✅ Admin API |
| `search_aws_cli` | Query AWS DynamoDB tables | ✅ AWS |

---

## Session & Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Slack Thread (thread_ts)                  │
│                                                             │
│  Message 1: "Check logs for user 123 GetMembership"         │
│    → session_id = UUID-abc123                               │
│    → set_session_context(UUID-abc123)                       │
│    → search_logstore → collection: s_abc123_logs_GetM_prod  │
│    → 50 logs fetched, embedded, stored                      │
│    → semantic search → top 10 returned to LLM               │
│    → LLM responds with analysis                             │
│                                                             │
│  Message 2: "What about timeout errors specifically?"       │
│    → same session_id = UUID-abc123                          │
│    → LLM decides: use retrieve_stored_logs                  │
│    → searches s_abc123_logs_GetM_prod (50 embeddings)       │
│    → finds timeout-related logs → responds                  │
│                                                             │
│  Message 3: "Also check the payment logs"                   │
│    → same session_id = UUID-abc123                          │
│    → LLM decides: search_logstore with new RPC method       │
│    → NEW collection: s_abc123_logs_ProcessPay_prod          │
│    → 30 new logs fetched, embedded, stored                  │
│    → Now session has 80 total embeddings across 2 colls     │
│                                                             │
│  Message 4: "Compare errors across both methods"            │
│    → retrieve_stored_logs searches BOTH collections         │
│    → Returns relevant logs from both GetMembership + Pay    │
└─────────────────────────────────────────────────────────────┘

At midnight: ALL collections deleted, all sessions cleared ♻️
```

### Memory Layers

| Layer | Scope | Storage | Persistence |
|-------|-------|---------|-------------|
| **Conversation Memory** | Per Slack thread | LangGraph `MemorySaver` (in-process dict) | Until bot restart |
| **Log Embeddings** | Per session + method | ChromaDB in-memory collections | Until midnight cleanup |
| **Thread → Session Map** | Global | Python dict (`thread_sessions`) | Until midnight cleanup or bot restart |

---

## Prerequisites

- Python 3.10+
- Access to Zomato AI Gateway (`ai-gateway.eks.zdev.net`)
- Slack workspace with admin access to install apps
- AWS credentials (for DynamoDB queries)

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# AI Gateway
AI_GATEWAY_BASE_URL=http://ai-gateway.eks.zdev.net/ai-gateway/api/v1
AI_GATEWAY_PROJECT_NAME=your-project
AI_GATEWAY_PROJECT_AUTH_KEY=your-auth-key

# AWS (for DynamoDB)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=ap-south-1
```

### 3. Run

```bash
python bot/slack_bot.py
```

Expected startup output:
```
✅ Agent graph created with routing logic!
============================================================
🤖 SRE AI Agent - Slack Bot
============================================================
✅ Bot Token: Configured
✅ App Token: Configured
⏰ Midnight cleanup scheduler started. First run in X.X hours

🚀 Starting bot in Socket Mode...
============================================================
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The Docker setup mounts `configs/` as read-only and `code_vector_db/` for ChromaDB persistence (if switched from in-memory to persistent mode).

---

## Slack Interaction Methods

| Method | Usage | Example |
|--------|-------|---------|
| Direct Message | Send any message to the bot | `check logs for user 123` |
| @Mention | Mention in any channel | `@SRE_Agent check logs for user 123` |
| Slash Command | Use `/sre` command | `/sre check logs for user 123` |
| `help` | Show capabilities | `help` |
| `reset` | Clear conversation history for thread | `reset` |
| `session` | Show current session ID | `session` |

---

## Adding a New Service Config

1. Create a JSON file in `configs/` (see `district_membership_service.json` as template)
2. The agent automatically loads all configs at startup
3. Configs feed into:
   - **System prompt**: Service knowledge, RPC methods, error patterns
   - **Log field filtering**: Which fields to keep/drop during cleaning
   - **Search refinement**: LLM uses config context to generate better search queries

## Adding a New Tool

1. Create a new sub-package in `tools/` (e.g., `tools/my_tool/`)
2. Add `__init__.py` with re-exports
3. Add `my_tool.py` with `@tool` decorated function
4. Register in `tools/__init__.py`
5. Add to the agent's tool list in `agent.py`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Lazy Vector DB init** | ChromaDB + embeddings init on first tool call, not at import. Prevents crashes if AI Gateway is down at startup. |
| **Embedding model fallback** | Tries `text-embedding-3-large` → `text-embedding-3-small` → `text-embedding-ada-002`. If all fail, falls back to simple deduplication. |
| **Session-scoped collections** | Collection names keyed by `session_id` + `method` + `env`. Same thread reuses same collection — embeddings accumulate across messages. |
| **Upsert not Add** | `collection.upsert()` with content-hash IDs prevents duplicate embeddings when same logs are fetched again. |
| **Midnight cleanup** | `threading.Timer` loop deletes all collections at 00:00. No external cron or scheduler deps. Daemon thread won't block shutdown. |
| **retrieve_stored_logs tool** | Lets LLM search existing embeddings without re-fetching from logstore API. Fast for follow-up questions in the same thread. |
| **Config-driven prompts** | System prompt dynamically built from JSON configs. Add new service configs without code changes. |
| **Thread-based sessions** | Slack `thread_ts` maps to `session_id` (UUID) which maps to both `MemorySaver` thread and ChromaDB collection prefix. |
