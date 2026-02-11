
# DebugMate — Complete System Flowchart

> Every step, every possibility, every function call — in detail.

---

## 1. SYSTEM STARTUP FLOW

```mermaid
flowchart TD
    START["🚀 python3 bot/slack_bot.py"] --> DOTENV["load_dotenv()<br/>Loads .env → os.environ"]
    DOTENV --> SYSPATH["sys.path.insert(0, parent_dir)<br/>So 'agent' module is importable"]
    SYSPATH --> IMPORT_AGENT["import agent.py<br/>├─ import core.config (load_dotenv again, safe)<br/>│  ├─ Load .env vars: AI_GATEWAY_*, AWS_*, LOGSTORE_*, EDITION_*<br/>│  ├─ load_service_config('district_membership_service')<br/>│  │  └─ Read configs/district_membership_service.json<br/>│  └─ Build: AI_GATEWAY_CONFIG, AWS_CONFIG, DEFAULT_HEADERS, EDITION_HEADERS<br/>├─ import core.ai_gateway_llm (AIGatewayLLM, AIGatewayEmbeddings)<br/>├─ import core.prompt → build_system_prompt()<br/>│  └─ Reads SERVICE_CONFIGS → builds dynamic prompt with:<br/>│     RPC methods, debugging scenarios, DB schema, troubleshooting<br/>├─ import tools/__init__.py<br/>│  ├─ import tools.logstore (search_logstore, search_panic_logs, etc.)<br/>│  ├─ import tools.codebase (search_codebase)<br/>│  ├─ import tools.admin_api (search_admin_apis)<br/>│  └─ import tools.aws (search_aws_cli + boto3.Session init)"]
    
    IMPORT_AGENT --> CREATE_LLM["llm = AIGatewayLLM(**AI_GATEWAY_CONFIG)<br/>Model: gpt-5, Temp: 0, Timeout: 180s"]
    CREATE_LLM --> CREATE_PROMPT["prompt = ChatPromptTemplate([system, messages])<br/>System prompt ~320 lines with service context"]
    CREATE_PROMPT --> CREATE_MEMORY["memory = MemorySaver()<br/>In-memory conversation history per thread_id"]
    CREATE_MEMORY --> CREATE_AGENT["agent = create_react_agent(<br/>  llm, tools=[6 tools], checkpointer=memory, prompt)"]
    CREATE_AGENT --> BUILD_GRAPH["workflow = StateGraph(AgentState)<br/>START → agent → END"]
    
    BUILD_GRAPH --> IMPORT_CLEANUP["from tools import cleanup_all_collections"]
    IMPORT_CLEANUP --> VALIDATE_TOKENS{"Check SLACK_BOT_TOKEN<br/>& SLACK_APP_TOKEN"}
    VALIDATE_TOKENS -->|Missing| RAISE_ERROR["❌ raise ValueError"]
    VALIDATE_TOKENS -->|Present| INIT_APP["app = App(token=SLACK_BOT_TOKEN)"]
    INIT_APP --> START_SCHEDULER["start_midnight_scheduler()<br/>├─ Calculate seconds until 00:00<br/>├─ threading.Timer(delay, _midnight_cleanup_loop)<br/>└─ timer.daemon = True, timer.start()"]
    START_SCHEDULER --> SOCKET_MODE["handler = SocketModeHandler(app, SLACK_APP_TOKEN)<br/>handler.start()<br/>🟢 Bot is LIVE — listening for events"]

    style START fill:#4CAF50,color:white
    style SOCKET_MODE fill:#2196F3,color:white
    style RAISE_ERROR fill:#f44336,color:white
```

---

## 2. SLACK EVENT HANDLING — Three Entry Points

```mermaid
flowchart TD
    SLACK["📨 Slack Event Received"] --> TYPE{Event Type?}
    
    TYPE -->|"@mention in channel"| MENTION["@app_mention handler"]
    TYPE -->|"Direct Message"| DM["message handler"]
    TYPE -->|"/sre command"| SLASH["slash command handler"]
    TYPE -->|"Home tab opened"| HOME["app_home_opened handler"]

    %% ── @mention flow ──
    MENTION --> M_EXTRACT["Extract: user_id, channel, thread_ts<br/>Remove bot mention from text"]
    M_EXTRACT --> M_EMPTY{Text empty?}
    M_EMPTY -->|Yes| M_GREET["say('👋 Hi! I'm your SRE AI Assistant...')<br/>Return"]
    M_EMPTY -->|No| M_DEDUP{"message_key in<br/>processing_messages?"}
    M_DEDUP -->|Yes| M_SKIP["Return (already processing)"]
    M_DEDUP -->|No| M_ADD["processing_messages.add(message_key)"]
    M_ADD --> M_THREAD["threading.Thread(<br/>  target=process_agent_query,<br/>  args=(user_id, text, channel, thread_ts)<br/>).start()"]

    %% ── DM flow ──
    DM --> D_FILTER{"subtype/bot_id present?<br/>OR channel_type ≠ 'im'?"}
    D_FILTER -->|Yes| D_IGNORE["Return (ignore)"]
    D_FILTER -->|No| D_TEXT["Extract: user_id, channel, text, thread_ts"]
    D_TEXT --> D_CMD{Special command?}
    D_CMD -->|"help"| D_HELP["Post help text with examples"]
    D_CMD -->|"reset"| D_RESET["Delete thread_sessions[session_key]<br/>Post '✅ Session reset!'"]
    D_CMD -->|"session"| D_SESSION["Post current session_id"]
    D_CMD -->|Normal text| D_DEDUP{"Already processing?"}
    D_DEDUP -->|Yes| D_SKIP["Return"]
    D_DEDUP -->|No| D_THREAD["threading.Thread → process_agent_query()"]

    %% ── Slash command flow ──
    SLASH --> S_ACK["ack() — acknowledge within 3s"]
    S_ACK --> S_EMPTY{Text empty?}
    S_EMPTY -->|Yes| S_ERR["chat_postEphemeral: '❌ Please provide a query'"]
    S_EMPTY -->|No| S_POST["chat_postMessage: '@user asked: text'<br/>Get thread_ts from response"]
    S_POST --> S_THREAD["threading.Thread → process_agent_query()"]

    %% ── Home tab ──
    HOME --> H_VIEW["client.views_publish(<br/>  Home tab with capabilities,<br/>  examples, usage instructions)"]

    style SLACK fill:#4A154B,color:white
    style M_THREAD fill:#2196F3,color:white
    style D_THREAD fill:#2196F3,color:white
    style S_THREAD fill:#2196F3,color:white
```

---

## 3. AGENT QUERY PROCESSING — The Core Loop

```mermaid
flowchart TD
    PQ["process_agent_query(<br/>user_id, query, channel, thread_ts)"] --> SESSION_KEY["session_key = thread_ts or channel"]
    SESSION_KEY --> GET_SESSION["session_id = thread_sessions.get(session_key)<br/>Could be None (new thread) or existing UUID"]
    GET_SESSION --> THINKING["Post '🤖 Processing your request...'<br/>to Slack thread"]
    
    THINKING --> RUN_AGENT["response, session_id = run_agent_with_session(query, session_id)"]
    
    RUN_AGENT --> RA_UUID{"session_id is None?"}
    RA_UUID -->|Yes| RA_NEW["session_id = uuid4()"]
    RA_UUID -->|No| RA_EXISTING["Use existing session_id"]
    RA_NEW --> RA_CONFIG
    RA_EXISTING --> RA_CONFIG["config = {thread_id: session_id, recursion_limit: 100}"]
    
    RA_CONFIG --> SET_CTX["set_session_context(session_id)<br/>├─ _current_session_id = session_id<br/>└─ _session_collections[session_id] = set()"]
    
    SET_CTX --> STREAM["agent.stream(<br/>  {messages: [HumanMessage(query)]},<br/>  config, stream_mode='values')"]
    
    STREAM --> LOOP["For each step in stream..."]
    LOOP --> MSG["message = step['messages'][-1]"]
    MSG --> MSG_TYPE{Message type?}
    
    MSG_TYPE -->|"AIMessage with tool_calls"| TOOL_DECIDE["LLM decided to call a tool<br/>ReAct: Reason → Act"]
    TOOL_DECIDE --> TOOL_EXEC["LangGraph auto-executes the tool<br/>(see Tool Execution section below)"]
    TOOL_EXEC --> TOOL_RESULT["ToolMessage with result<br/>fed back to LLM"]
    TOOL_RESULT --> LOOP
    
    MSG_TYPE -->|"AIMessage without tool_calls"| FINAL_CANDIDATE["Candidate final answer<br/>last_ai_message = message"]
    FINAL_CANDIDATE --> LOOP
    
    MSG_TYPE -->|"HumanMessage/ToolMessage"| LOOP
    
    LOOP -->|"Stream complete"| CHECK_FINAL{"last_ai_message<br/>has content?"}
    CHECK_FINAL -->|Yes| FINAL["final_response = last_ai_message.content"]
    CHECK_FINAL -->|No| FALLBACK["final_response = 'I apologize...'"]
    
    FINAL --> RETURN["Return (final_response, session_id)"]
    FALLBACK --> RETURN
    
    RETURN --> STORE_SESSION["thread_sessions[session_key] = session_id"]
    STORE_SESSION --> CLEANUP_OLD["cleanup_old_sessions()<br/>If len > 100, remove oldest"]
    CLEANUP_OLD --> DELETE_THINKING["Delete '🤖 Processing...' message"]
    
    DELETE_THINKING --> FORMAT["format_response_for_slack(response)<br/>├─ Replace ``` with `<br/>└─ Collapse excessive newlines"]
    
    FORMAT --> LENGTH{len > 3900?}
    LENGTH -->|Yes| CHUNK["Split into 3900-char chunks<br/>Post each as 'Part X/Y'"]
    LENGTH -->|No| POST["chat_postMessage(formatted_response)"]
    
    CHUNK --> DONE["✅ Response delivered"]
    POST --> DONE
    
    PQ --> ERROR["Exception handler:<br/>Post '❌ Error: ...' to Slack thread"]
    PQ --> FINALLY["processing_messages.discard(message_key)"]

    style PQ fill:#FF9800,color:white
    style STREAM fill:#9C27B0,color:white
    style DONE fill:#4CAF50,color:white
    style TOOL_EXEC fill:#E91E63,color:white
```

---

## 4. AI GATEWAY LLM — How Each LLM Call Works

```mermaid
flowchart TD
    INVOKE["llm._generate(messages, **kwargs)"] --> CONVERT_MSGS["Convert each LangChain message → Gateway format<br/>├─ SystemMessage → role: 'SYSTEM', type: 'TEXT'<br/>├─ HumanMessage → role: 'USER', type: 'TEXT'<br/>├─ AIMessage(tool_calls) → role: 'ASSISTANT', type: 'TOOL_USE'<br/>└─ ToolMessage → role: 'TOOL', type: 'TOOL_USE' with result"]
    
    CONVERT_MSGS --> BUILD_PAYLOAD["Build payload:<br/>{model, messages, config:{temperature, max_tokens},<br/>client_options:{retries, timeout, source:'sre_agent'}}"]
    
    BUILD_PAYLOAD --> HAS_TOOLS{"Tools bound<br/>or in kwargs?"}
    HAS_TOOLS -->|Yes| ADD_TOOLS["Convert tools → Gateway format<br/>payload.tool_option = {tools: [...], tool_choice: 'AUTO'}"]
    HAS_TOOLS -->|No| SKIP_TOOLS["No tool_option in payload"]
    
    ADD_TOOLS --> API_CALL
    SKIP_TOOLS --> API_CALL
    
    API_CALL["POST {base_url}/chat/completion<br/>Headers: Content-Type, x-project-name, x-project-auth-key"]
    
    API_CALL --> STATUS{Status 200?}
    STATUS -->|No| API_ERROR["raise Exception('API returned {status}: {error}')"]
    STATUS -->|Yes| PARSE["Parse response JSON"]
    
    PARSE --> EXTRACT["Extract choices[0].content parts"]
    EXTRACT --> PART_TYPE{Part type?}
    
    PART_TYPE -->|"TEXT (type=1)"| TEXT_CONTENT["Append to text_content string"]
    PART_TYPE -->|"TOOL_USE (type=6)"| TOOL_CALLS["Extract tool_calls:<br/>├─ name (function name)<br/>├─ args (parameters, parse if string)<br/>└─ id (tool call ID)"]
    
    TEXT_CONTENT --> BUILD_AI
    TOOL_CALLS --> BUILD_AI
    
    BUILD_AI{"Has tool_calls?"}
    BUILD_AI -->|Yes| AI_TOOL["AIMessage(content=text, tool_calls=[...])"]
    BUILD_AI -->|No| AI_TEXT["AIMessage(content=text)"]
    
    AI_TOOL --> RESULT["Return ChatResult(generations=[ChatGeneration(message)])"]
    AI_TEXT --> RESULT

    style INVOKE fill:#673AB7,color:white
    style API_CALL fill:#2196F3,color:white
    style API_ERROR fill:#f44336,color:white
    style RESULT fill:#4CAF50,color:white
```

---

## 5. TOOL: search_logstore — Full RAG Pipeline

```mermaid
flowchart TD
    SL["search_logstore(<br/>identifier, rpc_method,<br/>hours_ago, get_error_logs_only, environment)"] --> CALC_TIME["Calculate time window:<br/>ts_to = now - (hours_ago * 3600)<br/>ts_from = ts_to - 86400"]
    
    CALC_TIME --> PARSE_METHOD["Extract method name:<br/>'pkg/Service/Method' → 'Method'<br/>'Service.Method' → 'Method'"]
    
    PARSE_METHOD --> BUILD_FILTERS["Build filters:<br/>├─ body.l-method-name LIKE method_name<br/>├─ body.user-id LIKE identifier (if present)<br/>└─ msg LIKE [error, exception, failed] (if errors_only)"]
    
    BUILD_FILTERS --> API_POST["POST LOGSTORE_URL<br/>Headers: DEFAULT_HEADERS (auth + cookie from .env)<br/>Payload: {ts_from, ts_to, logtype, filters, environment}<br/>Timeout: 120s"]
    
    API_POST --> CONTENT_TYPE{Response Content-Type?}
    CONTENT_TYPE -->|"text/html"| AUTH_ERR["Return: AUTH_REQUIRED<br/>'API returned HTML login page'"]
    CONTENT_TYPE -->|"application/json"| PARSE_JSON["Parse response JSON"]
    
    PARSE_JSON --> VALID_DICT{Is dict?}
    VALID_DICT -->|No| INVALID["Return: INVALID_RESPONSE"]
    VALID_DICT -->|Yes| GET_HITS["hits = data.get('hits', [])"]
    
    GET_HITS --> EXTRACT_MSGS["Extract messages:<br/>for hit in hits:<br/>  if 'msg' in hit: messages.append(hit['msg'])"]
    
    EXTRACT_MSGS --> HAS_MSGS{len(messages) > 0?}
    HAS_MSGS -->|No| NO_LOGS["Return: {total_hits: 0, processed_logs: []}"]
    HAS_MSGS -->|Yes| COLLECTION["collection_name = _build_collection_name(<br/>  f'logs_{method_name}', environment)<br/>e.g. 's_abc123def456_logs_GetMembership_prod'"]
    
    COLLECTION --> STORE["store_logs_in_vector_db(messages, collection_name)"]
    
    STORE --> STORE_INIT["initialize_vector_db() — lazy init"]
    STORE_INIT --> STORE_INIT_CHECK{Already initialized?}
    STORE_INIT_CHECK -->|Yes| STORE_GET_COLLECTION
    STORE_INIT_CHECK -->|No| TRY_EMBEDDINGS["Try embedding models in order:<br/>1. text-embedding-3-large (1536d)<br/>2. text-embedding-3-small (1536d)<br/>3. text-embedding-ada-002 (1536d)<br/>Each: create AIGatewayEmbeddings → embed_query('test')"]
    TRY_EMBEDDINGS --> EMB_SUCCESS{Any model worked?}
    EMB_SUCCESS -->|No| VDB_DISABLED["vector_db_enabled = False"]
    EMB_SUCCESS -->|Yes| VDB_ENABLED["vector_db_enabled = True<br/>Initialize LLM for prompt refinement"]
    VDB_DISABLED --> STORE_FAIL
    VDB_ENABLED --> STORE_GET_COLLECTION
    
    STORE_GET_COLLECTION["get_or_create_log_collection(name)<br/>├─ Exists in _vector_stores? → reuse<br/>└─ New? → InMemoryVectorStore(embeddings)"]
    
    STORE_GET_COLLECTION --> DEDUP["Hash-based deduplication:<br/>for each log:<br/>  hash = md5(log)<br/>  if hash not in _stored_doc_hashes[name]:<br/>    new_docs.append(clean_log_message(log))<br/>    add hash to set"]
    
    DEDUP --> HAS_NEW{New docs to store?}
    HAS_NEW -->|No| DEDUP_SKIP["All duplicates — skip storage<br/>Return total count (they ARE stored)"]
    HAS_NEW -->|Yes| ADD_TEXTS["store.add_texts(new_docs)<br/>Internally: embed each doc → store in memory"]
    
    ADD_TEXTS --> STORED_COUNT["stored_count = len(messages)"]
    DEDUP_SKIP --> STORED_COUNT
    
    STORED_COUNT --> STORE_CHECK{stored_count > 0?}
    STORE_CHECK -->|No| STORE_FAIL["⚠️ Fallback: simple deduplication<br/>unique_logs = dict.fromkeys(messages)<br/>limited_logs = [clean_log_message(log) for log in unique[:10]]"]
    
    STORE_CHECK -->|Yes| REFINE["refine_search_prompt(user_query, rpc_method, identifier)<br/>├─ Build context from SERVICE_CONFIGS<br/>├─ Ask LLM to generate refined search query<br/>└─ Fallback: concatenate method + identifier + query"]
    
    REFINE --> SEMANTIC["semantic_search_logs(refined_query, collection_name, n=10)<br/>├─ store.similarity_search(query, k=10)<br/>└─ Return [doc.page_content for doc in results]"]
    
    SEMANTIC --> LIMITED["limited_logs = relevant_logs[:10]"]
    STORE_FAIL --> LIMITED
    
    LIMITED --> RESULT_JSON["Return JSON:<br/>{total_hits, original_count, stored_in_vector_db,<br/>relevant_logs_count, search_method, processed_logs}"]

    style SL fill:#E91E63,color:white
    style API_POST fill:#2196F3,color:white
    style ADD_TEXTS fill:#9C27B0,color:white
    style SEMANTIC fill:#FF9800,color:white
    style AUTH_ERR fill:#f44336,color:white
```

---

## 6. TOOL: get_all_logs_for_request_id

```mermaid
flowchart TD
    GL["get_all_logs_for_request_id(<br/>request_id, hours_ago, environment)"] --> CALC["Calculate time window<br/>(same as search_logstore)"]
    CALC --> FILTER["Filter: body.request_id LIKE request_id"]
    FILTER --> LOGTYPE["logtype = get_logtype_from_config()<br/>→ SERVICE_CONFIGS lookup or default"]
    LOGTYPE --> POST["POST LOGSTORE_URL with payload"]
    POST --> CHECK_HTML{HTML response?}
    CHECK_HTML -->|Yes| AUTH["Return: AUTH_REQUIRED"]
    CHECK_HTML -->|No| PARSE["Parse hits, extract messages"]
    PARSE --> ZERO{0 messages?}
    ZERO -->|Yes| NO_LOGS["Return: NO_LOGS_FOUND<br/>+ possible_reasons + suggestions"]
    ZERO -->|No| COLL["collection = _build_collection_name(<br/>  f'req_{request_id}', env)"]
    COLL --> STORE_FLOW["Same RAG pipeline as search_logstore:<br/>store → refine → semantic search → return"]

    style GL fill:#E91E63,color:white
```

---

## 7. TOOL: search_panic_logs

```mermaid
flowchart TD
    PL["search_panic_logs(lookback_minutes=10)"] --> TIME["ts_from = now - (minutes * 60)<br/>ts_to = now"]
    TIME --> FILTER["Filter: msg LIKE 'panic'"]
    FILTER --> POST["POST LOGSTORE_URL<br/>environment: 'stag'"]
    POST --> CHECK{HTML/empty?}
    CHECK -->|Error| ERR["Return error JSON"]
    CHECK -->|OK| EXTRACT["Extract messages from hits"]
    EXTRACT --> ZERO{0 messages?}
    ZERO -->|Yes| NONE["Return: {total_hits: 0}"]
    ZERO -->|No| COLL["collection = _build_collection_name(<br/>  'panic_logs', 'stag')"]
    COLL --> RAG["Same RAG pipeline:<br/>store → refine('critical panic/crash logs') → semantic search"]

    style PL fill:#E91E63,color:white
```

---

## 8. TOOL: retrieve_stored_logs (Cross-Session Search)

```mermaid
flowchart TD
    RL["retrieve_stored_logs(<br/>search_query, n_results=10)"] --> INIT{"initialize_vector_db()<br/>& embeddings available?"}
    INIT -->|No| VDB_ERR["Return: VECTOR_DB_UNAVAILABLE"]
    INIT -->|Yes| FIND_COLS["Find session collections:<br/>session_cols = _session_collections[session_id]<br/>Fallback: scan _vector_stores for 's_{session[:12]}_*'"]
    
    FIND_COLS --> HAS_COLS{Any collections?}
    HAS_COLS -->|No| NO_STORED["Return: NO_STORED_LOGS<br/>'Use search_logstore first'"]
    HAS_COLS -->|Yes| REFINE["refine_search_prompt(search_query)"]
    
    REFINE --> SEARCH_ALL["For each collection in session:<br/>  store.similarity_search_with_score(refined_query, k=n)<br/>  Collect: {log, relevance_score, collection}"]
    
    SEARCH_ALL --> HAS_RESULTS{Any results?}
    HAS_RESULTS -->|No| NO_RELEVANT["Return: NO_RELEVANT_LOGS"]
    HAS_RESULTS -->|Yes| SORT["Sort by relevance_score DESC<br/>Take top n_results"]
    
    SORT --> RETURN["Return JSON:<br/>{collections_searched, total_matches,<br/>returned_count, processed_logs}"]

    style RL fill:#FF9800,color:white
    style SEARCH_ALL fill:#9C27B0,color:white
```

---

## 9. TOOL: search_admin_apis

```mermaid
flowchart TD
    AA["search_admin_apis(<br/>search_context, user_identifier, plan_type)"] --> DUMMY{user_id == '777'?}
    DUMMY -->|Yes| NO_DATA["Return: 'User has no membership data.'"]
    DUMMY -->|No| POST["POST EDITION_API_URL<br/>Headers: EDITION_HEADERS (Cookie from .env)<br/>Body: {user_id, plan_type}"]
    POST --> STATUS{HTTP OK?}
    STATUS -->|No| ERR["Return: [ADMIN API ERROR] {error}"]
    STATUS -->|Yes| PARSE["Parse JSON response"]
    PARSE --> CLEAN["remove_assets_and_images(data)<br/>Recursively remove: assets, image, logo, banner, lottie keys"]
    CLEAN --> FORMAT["Return: '[ADMIN API - User: {id}]\\n{json}'"]

    style AA fill:#4CAF50,color:white
```

---

## 10. TOOL: search_aws_cli

```mermaid
flowchart TD
    AWS["search_aws_cli(table_name, partition_key)"] --> TABLE["table = dynamodb.Table(table_name)"]
    TABLE --> QUERY["table.query(<br/>  KeyConditionExpression=Key('partition_key').eq(pk))"]
    QUERY --> ITEMS["items = response.get('Items', [])"]
    ITEMS --> ENCODE["json.dumps(items, cls=DecimalEncoder)<br/>Converts Decimal → float"]
    ENCODE --> RETURN["Return: '[AWS DynamoDB]\\n{json}'"]

    style AWS fill:#FF9800,color:white
```

---

## 11. TOOL: search_codebase (Currently Disabled)

```mermaid
flowchart TD
    CB["search_codebase(search_context, keywords)"] --> CHECK{retriever is None?}
    CHECK -->|Yes| DISABLED["Return: '⚠️ Code search is currently disabled.'"]
    CHECK -->|No| ENHANCE["enhanced_query = context + keywords"]
    ENHANCE --> RETRIEVE["docs = retriever.invoke(enhanced_query)"]
    RETRIEVE --> HAS_DOCS{docs found?}
    HAS_DOCS -->|No| NO_CODE["Return: 'No relevant code found.'"]
    HAS_DOCS -->|Yes| FORMAT["Format top 5 docs with file/content"]
    FORMAT --> LLM_ANALYZE["LLM analyzes code:<br/>What it does, where, how, details"]
    LLM_ANALYZE --> RETURN["Return: formatted analysis"]

    style CB fill:#607D8B,color:white
    style DISABLED fill:#9E9E9E,color:white
```

---

## 12. LOG CLEANING PIPELINE (clean_log_message)

```mermaid
flowchart TD
    CLM["clean_log_message(log_message,<br/>keep_fields, rpc_method)"] --> LOAD_CONFIG["Load droppable_fields from SERVICE_CONFIGS<br/>Match rpc_method → get always_drop + response_droppable"]
    LOAD_CONFIG --> DEFAULT_FIELDS["Default keep_fields if not found:<br/>request-id, user-id, l-method-name, l-latency,<br/>subscription_id, order_id, campaign_id, city_id, etc."]
    DEFAULT_FIELDS --> CLEAN1["Remove: \\n, \\t, \\, double quotes<br/>Keep ASCII printable only"]
    CLEAN1 --> PRESERVE["Extract & preserve important fields:<br/>Pattern match 'field_name: value' for each keep_field"]
    PRESERVE --> DROP["Drop config-specified fields:<br/>Regex remove JSON objects, arrays, strings, values"]
    DROP --> EXTRACT_ERR["Extract error patterns:<br/>error:, failed:, exception:, unable to"]
    EXTRACT_ERR --> COMBINE{"preserved_parts found?"}
    COMBINE -->|Yes| JOIN["cleaned = ' | '.join(preserved_parts)"]
    COMBINE -->|No| KEEP_ALL["Use full cleaned message"]
    JOIN --> STOPWORDS["Remove stopwords (is, am, are, the, etc.)"]
    KEEP_ALL --> STOPWORDS
    STOPWORDS --> TRUNCATE["Collapse spaces, limit to 500 chars"]
    TRUNCATE --> RETURN["Return cleaned log"]

    style CLM fill:#795548,color:white
```

---

## 13. SESSION & MEMORY ARCHITECTURE

```mermaid
flowchart TD
    subgraph SLACK["Slack Thread"]
        MSG1["Message 1: 'Check logs for user 123'"]
        MSG2["Message 2: 'What errors did you find?'"]
        MSG3["Message 3: 'Check their subscription too'"]
    end

    subgraph SESSION_MAP["thread_sessions (dict)"]
        TS1["thread_ts → session_id (UUID)"]
    end

    subgraph LANGGRAPH["LangGraph MemorySaver"]
        MEM["thread_id=session_id → Full conversation history<br/>All HumanMessages + AIMessages + ToolMessages"]
    end

    subgraph VECTOR_STORES["_vector_stores (dict)"]
        VS1["s_abc123_logs_GetMembership_prod → InMemoryVectorStore"]
        VS2["s_abc123_req_xyz789_prod → InMemoryVectorStore"]
    end

    subgraph SESSION_COLS["_session_collections (dict)"]
        SC1["session_id → {collection_name_1, collection_name_2, ...}"]
    end

    MSG1 -->|"New thread"| TS1
    MSG2 -->|"Same thread_ts"| TS1
    MSG3 -->|"Same thread_ts"| TS1

    TS1 -->|"run_agent_with_session(query, session_id)"| MEM
    TS1 -->|"set_session_context(session_id)"| SC1

    MEM -->|"LLM sees full conversation"| TOOLS["Tool calls use same collections"]
    TOOLS --> VS1
    TOOLS --> VS2
    SC1 --> VS1
    SC1 --> VS2

    style SLACK fill:#4A154B,color:white
    style LANGGRAPH fill:#9C27B0,color:white
    style VECTOR_STORES fill:#E91E63,color:white
```

---

## 14. MIDNIGHT CLEANUP FLOW

```mermaid
flowchart TD
    BOOT["Bot startup"] --> CALC["_seconds_until_midnight()<br/>e.g. 43200 seconds (12 hours)"]
    CALC --> TIMER1["threading.Timer(delay, _midnight_cleanup_loop)<br/>timer.daemon = True<br/>timer.start()"]
    
    TIMER1 -->|"At 00:00:00"| CLEANUP["_midnight_cleanup_loop()"]
    CLEANUP --> C1["cleanup_all_collections()<br/>├─ _vector_stores.clear()<br/>├─ _stored_doc_hashes.clear()<br/>└─ _session_collections.clear()"]
    C1 --> C2["thread_sessions.clear()<br/>(Slack session mapping)"]
    C2 --> RESCHEDULE["Schedule next run:<br/>delay = _seconds_until_midnight()<br/>threading.Timer(delay, self).start()"]
    RESCHEDULE -->|"Next midnight"| CLEANUP

    style CLEANUP fill:#f44336,color:white
    style BOOT fill:#4CAF50,color:white
```

---

## 15. AI GATEWAY EMBEDDINGS FLOW

```mermaid
flowchart TD
    EMBED["AIGatewayEmbeddings._embed(texts)"] --> PAYLOAD["Build payload:<br/>{model, inputs: [{text: ...}],<br/>config: {encoding: FLOAT32_ARRAY, normalization: L2},<br/>client_options: {source: 'sre_agent'}}"]
    PAYLOAD --> DIM{output_dim set?}
    DIM -->|Yes| ADD_DIM["payload.config.output_dim = N"]
    DIM -->|No| SKIP_DIM["No dimension override"]
    ADD_DIM --> POST
    SKIP_DIM --> POST
    POST["POST {base_url}/embeddings<br/>Headers: x-project-name, x-project-auth-key"]
    POST --> STATUS{200?}
    STATUS -->|No| ERR["raise Exception"]
    STATUS -->|Yes| PARSE["Try 3 response formats:<br/>1. outputs[].embedding<br/>2. data[].embedding<br/>3. Direct array"]
    PARSE --> HAS_EMB{Embeddings found?}
    HAS_EMB -->|No| EMB_ERR["raise Exception('No embeddings')"]
    HAS_EMB -->|Yes| RETURN["Return List[List[float]]"]

    style EMBED fill:#673AB7,color:white
```

---

## 16. COMPLETE END-TO-END EXAMPLE

```
User in Slack: "@DebugMate check logs for user 162434451 GetApplicableCampaigns"

1. SLACK EVENT
   └─ @app_mention → extract text, thread_ts
   └─ threading.Thread → process_agent_query()

2. SESSION
   └─ session_key = thread_ts = "1234567890.123456"
   └─ session_id = None (new thread) → uuid4() = "a1b2c3d4-..."
   └─ set_session_context("a1b2c3d4-...")

3. AGENT (ReAct Loop)
   └─ LLM receives: system_prompt + HumanMessage("check logs for user...")
   └─ LLM decides: Call search_logstore(
        identifier="162434451",
        rpc_method="GetApplicableCampaigns",
        hours_ago=0,
        environment="prod"
      )

4. TOOL: search_logstore
   ├─ Time window: [now-24h, now]
   ├─ Filters: l-method-name LIKE "GetApplicableCampaigns", user-id LIKE "162434451"
   ├─ POST https://kyno.z.tt/logstore/api/v1/query → 47 hits
   ├─ Extract 47 messages from hits
   ├─ Collection: "s_a1b2c3d4e5f6_logs_GetApplicableCa_prod"
   ├─ store_logs_in_vector_db():
   │   ├─ initialize_vector_db() → try text-embedding-3-large ✅
   │   ├─ InMemoryVectorStore(embeddings) created
   │   ├─ Hash dedup: 47 unique → 47 new docs
   │   ├─ store.add_texts(47 cleaned logs) → embedded + stored
   │   └─ stored_count = 47
   ├─ refine_search_prompt():
   │   ├─ Load SERVICE_CONFIGS context for GetApplicableCampaigns
   │   ├─ LLM generates refined query: "GetApplicableCampaigns user 162434451 campaign eligibility..."
   │   └─ Return refined query
   ├─ semantic_search_logs():
   │   ├─ store.similarity_search(refined_query, k=10)
   │   └─ Return 10 most relevant log messages
   └─ Return JSON: {total_hits: 47, relevant_logs_count: 10, processed_logs: [...]}

5. AGENT (ReAct continued)
   └─ LLM receives ToolMessage with 10 relevant logs
   └─ LLM analyzes logs, formats response:
      "User 162434451's GetApplicableCampaigns call shows:
       - Request received at 14:32 UTC
       - 3 campaigns evaluated, 2 eligible
       - Campaign C-1234 filtered out due to city_id mismatch
       ..."

6. SLACK RESPONSE
   └─ Delete "🤖 Processing..." message
   └─ format_response_for_slack(response)
   └─ Post formatted response to thread
   └─ thread_sessions["1234567890.123456"] = "a1b2c3d4-..."

7. FOLLOW-UP MESSAGE (same thread)
   User: "What errors did you find?"
   └─ Same thread_ts → same session_id
   └─ LLM sees full conversation history (MemorySaver)
   └─ LLM decides: Call retrieve_stored_logs(
        search_query="errors failures exceptions",
        n_results=10
      )
   └─ Searches EXISTING embeddings (no new API call!)
   └─ Returns error-focused logs from session collections
```

---

## 17. FILE DEPENDENCY MAP

```
.env                          ← All secrets, URLs, tokens, cookies
.env.example                  ← Safe template for commits
│
bot/slack_bot.py              ← Entry point: load_dotenv() → import agent
│   ├── agent.py              ← ReAct agent: LLM + tools + memory
│   │   ├── core/config.py    ← load_dotenv(), AI_GATEWAY_CONFIG, AWS_CONFIG, headers
│   │   ├── core/prompt.py    ← build_system_prompt() from SERVICE_CONFIGS
│   │   ├── core/ai_gateway_llm.py ← AIGatewayLLM (BaseChatModel), AIGatewayEmbeddings
│   │   ├── core/models.py    ← Pydantic schemas: LogstoreInput, AdminAPIInput, etc.
│   │   └── tools/__init__.py ← Re-exports all tools
│   │       ├── tools/logstore/logstore_tool.py  ← RAG pipeline: fetch → store → embed → search
│   │       │   └── tools/logstore/utils.py      ← clean_log_message() with config-based field dropping
│   │       ├── tools/codebase/codebase_tool.py  ← Code search (currently disabled, retriever=None)
│   │       ├── tools/admin_api/admin_api_tool.py← POST Edition API for user subscriptions
│   │       └── tools/aws/aws_tool.py            ← DynamoDB queries via boto3
│   └── tools.cleanup_all_collections            ← Midnight purge of all vector stores
│
configs/district_membership_service.json          ← Service-specific config: RPC methods, debug scenarios
legacy/sre_agent.py                               ← Old monolithic version (not used by bot)
docs/FLOWCHART.md                                 ← This file
docs/FLOW_SUMMARY.md                              ← Layman-friendly flow explanation
README.md                                         ← Project documentation
```

---

## 18. DATA FLOW SUMMARY TABLE

| Step | Component | Input | Output | External Call? |
|------|-----------|-------|--------|----------------|
| 1 | Slack Bot | Slack event (mention/DM/command) | Parsed text + thread_ts | No |
| 2 | Session Manager | thread_ts | session_id (UUID) | No |
| 3 | Agent (LangGraph) | HumanMessage + history | Tool calls or final answer | AI Gateway LLM |
| 4a | search_logstore | identifier, method, env | 10 relevant logs (JSON) | Logstore API + AI Gateway (embed + LLM) |
| 4b | get_all_logs_for_request_id | request_id, env | 10 relevant logs (JSON) | Logstore API + AI Gateway |
| 4c | search_panic_logs | lookback_minutes | Panic logs (JSON) | Logstore API + AI Gateway |
| 4d | retrieve_stored_logs | search_query | Logs from existing embeddings | AI Gateway (embed only) |
| 4e | search_admin_apis | user_id, plan_type | User subscription data (JSON) | Edition API |
| 4f | search_aws_cli | table, partition_key | DynamoDB items (JSON) | AWS DynamoDB |
| 4g | search_codebase | context, keywords | Code analysis (disabled) | None (disabled) |
| 5 | Agent (LangGraph) | Tool results | Analyzed response text | AI Gateway LLM |
| 6 | Slack Bot | Response text | Formatted Slack message | Slack API |
| 7 | Midnight Scheduler | Timer event | Cleared memory | No |
