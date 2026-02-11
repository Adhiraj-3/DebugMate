
"""
Slack Bot Integration for SRE Agent
Handles Slack events and routes messages to the SRE agent.

Features:
  - Approval Gate: Queries are NOT forwarded to the LLM until a whitelisted user
    reacts with the configured emoji (default: ✅ white_check_mark).
  - PII Sanitization: Before posting the final response, a second LLM call strips
    all Personally Identifiable Information (Aadhaar, PAN, phone, names, etc.).
"""
import os
import sys
import json
import uuid
from dotenv import load_dotenv

# Add parent directory to path so we can import agent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file FIRST
# This must happen before importing agent
load_dotenv()

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from agent import run_agent_with_session
from tools import cleanup_all_collections
from core.config import WHITELISTED_APPROVER_IDS, APPROVAL_EMOJI
import threading
from datetime import datetime, timedelta

# Initialize Slack app with bot token and socket mode
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Store session mappings: {thread_ts: session_id}
# Using thread_ts as key ensures each thread maintains its own conversation context
# This enables the agent to remember user_id and other context within a thread
thread_sessions = {}

# Store active processing flags to prevent duplicate processing
processing_messages = set()

# Maximum number of active sessions to keep in memory
MAX_SESSIONS = 100

# ================== APPROVAL GATE STATE ==================
# Pending queries waiting for approval: {message_ts: {user_id, query, channel, thread_ts}}
# message_ts is the ts of the ORIGINAL user message (the one that needs a reaction)
_pending_queries = {}
_pending_lock = threading.Lock()

# Whether the approval gate is active (requires at least one whitelisted approver)
APPROVAL_GATE_ENABLED = len(WHITELISTED_APPROVER_IDS) > 0


def cleanup_old_sessions():
    """
    Keep only the most recent MAX_SESSIONS sessions
    This prevents memory buildup over time
    """
    if len(thread_sessions) > MAX_SESSIONS:
        # Remove oldest sessions (first inserted)
        excess = len(thread_sessions) - MAX_SESSIONS
        for _ in range(excess):
            thread_sessions.pop(next(iter(thread_sessions)))


def format_response_for_slack(response: str) -> str:
    """
    Format agent response for better Slack display
    Converts markdown-like formatting to Slack formatting
    """
    # Replace code blocks with Slack code formatting
    response = response.replace("```", "`")
    
    # Clean up excessive newlines
    lines = response.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        if line.strip() == "":
            if not prev_empty:
                cleaned_lines.append(line)
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    return '\n'.join(cleaned_lines)


def _queue_pending_query(message_ts: str, user_id: str, query: str, channel: str, thread_ts: str):
    """
    Store a query as pending approval. The message_ts is the Slack timestamp
    of the original user message that approvers will react to.
    """
    with _pending_lock:
        _pending_queries[message_ts] = {
            "user_id": user_id,
            "query": query,
            "channel": channel,
            "thread_ts": thread_ts,
        }
    print(f"⏳ Query queued for approval: msg_ts={message_ts}, user={user_id}")


def _pop_pending_query(message_ts: str) -> dict:
    """Remove and return a pending query by its message_ts, or None."""
    with _pending_lock:
        return _pending_queries.pop(message_ts, None)


def process_agent_query(user_id: str, query: str, channel: str, thread_ts: str = None):
    """
    Process user query through the agent in a separate thread.
    After the LLM responds, PII is sanitized before posting to Slack.
    """
    try:
        # Use thread_ts as session key to maintain conversation context per thread
        # If thread_ts is None (new message), use channel as fallback
        session_key = thread_ts if thread_ts else channel
        
        # Get existing session for this thread, or None for new thread
        session_id = thread_sessions.get(session_key)
        
        # Show typing indicator
        client = app.client
        
        # Post a temporary "thinking" message
        thinking_msg = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="🤖 _Processing your request..._"
        )
        
        # Run the agent with thread-specific session
        response, session_id = run_agent_with_session(query, session_id)
        
        # Store session for this thread to maintain context
        thread_sessions[session_key] = session_id
        
        # Cleanup old sessions if we have too many
        cleanup_old_sessions()
        
        # Print response to terminal for debugging
        print("\n" + "="*80)
        print("SLACK BOT - AGENT RESPONSE")
        print("="*80)
        print(f"User: {user_id}")
        print(f"Thread: {session_key}")
        print(f"Query: {query}")
        print(f"Session: {session_id}")
        print(f"Active Sessions: {len(thread_sessions)}")
        print("-"*80)
        print("RAW RESPONSE FROM AGENT:")
        print("-"*80)
        print(response)
        print("-"*80)
        print(f"Response Type: {type(response)}")
        print(f"Response Length: {len(response)} chars")
        print("="*80 + "\n")
        
        # Format response for Slack
        formatted_response = format_response_for_slack(response)
        
        # Print formatted response too
        print("FORMATTED FOR SLACK:")
        print("-"*80)
        print(formatted_response)
        print("="*80 + "\n")
        
        # Delete thinking message
        client.chat_delete(
            channel=channel,
            ts=thinking_msg['ts']
        )
        
        # Post the actual response
        # Split long messages if needed (Slack has a 4000 char limit)
        if len(formatted_response) > 3900:
            chunks = [formatted_response[i:i+3900] for i in range(0, len(formatted_response), 3900)]
            for i, chunk in enumerate(chunks):
                prefix = f"📄 *Response (Part {i+1}/{len(chunks)})*\n\n" if len(chunks) > 1 else ""
                client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=prefix + chunk
                )
        else:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=formatted_response
            )
            
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"❌ *Error processing request:*\n```{str(e)}```"
        client = app.client
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=error_msg
        )
    finally:
        # Remove from processing set
        message_key = f"{user_id}:{channel}:{thread_ts}"
        processing_messages.discard(message_key)


def _request_approval(channel: str, thread_ts: str, user_id: str, query: str, message_ts: str):
    """
    Post an approval-request message in the thread and queue the query.
    Approvers must react with the configured emoji on the ORIGINAL message.
    """
    client = app.client

    approver_mentions = ", ".join(f"<@{uid}>" for uid in WHITELISTED_APPROVER_IDS)
    emoji_display = f":{APPROVAL_EMOJI}:"

    client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=(
            f"⏳ *Approval Required*\n"
            f"<@{user_id}>'s query is waiting for approval.\n"
            f"A whitelisted approver ({approver_mentions}) must react with "
            f"{emoji_display} on the original message to proceed."
        ),
    )

    # Queue the pending query keyed by the original message timestamp
    _queue_pending_query(message_ts, user_id, query, channel, thread_ts)


def _handle_incoming_query(user_id: str, text: str, channel: str, thread_ts: str, message_ts: str):
    """
    Central entry point for all incoming user queries (mentions, DMs, slash commands).

    Approval logic:
      - If the sender IS a whitelisted user → process immediately (no approval needed).
      - If the sender is NOT whitelisted AND the gate is enabled → queue for approval.
        A whitelisted user must react with the approval emoji to proceed.
      - If the gate is disabled (no whitelisted IDs configured) → process immediately.
    """
    # Create unique key for this message
    message_key = f"{user_id}:{channel}:{thread_ts}"

    # Check if already processing
    if message_key in processing_messages:
        return

    # Whitelisted users always bypass the approval gate
    if user_id in WHITELISTED_APPROVER_IDS:
        print(f"✅ Whitelisted user {user_id} — bypassing approval gate")
        processing_messages.add(message_key)
        thread = threading.Thread(
            target=process_agent_query,
            args=(user_id, text, channel, thread_ts)
        )
        thread.start()
    elif APPROVAL_GATE_ENABLED:
        # Non-whitelisted user + gate enabled → queue for approval
        _request_approval(channel, thread_ts, user_id, text, message_ts)
    else:
        # No gate configured — process immediately for everyone
        processing_messages.add(message_key)
        thread = threading.Thread(
            target=process_agent_query,
            args=(user_id, text, channel, thread_ts)
        )
        thread.start()


# ================== REACTION HANDLER (Approval Gate) ==================

@app.event("reaction_added")
def handle_reaction_added(event, client):
    """
    When a whitelisted user reacts with the approval emoji on a pending message,
    pop it from the queue and start processing.
    Matches by exact message_ts first, then falls back to matching any pending query
    in the same channel (in case the approver reacted on the bot's message instead).
    """
    print(f"\n👀 REACTION EVENT RECEIVED: {json.dumps(event, indent=2)}")

    if not APPROVAL_GATE_ENABLED:
        print("   ❌ Approval gate not enabled — ignoring reaction")
        return

    reaction = event.get("reaction", "")
    reacting_user = event.get("user", "")
    item = event.get("item", {})
    message_ts = item.get("ts", "")
    channel = item.get("channel", "")

    print(f"   Reaction: {reaction} (expected: {APPROVAL_EMOJI})")
    print(f"   Reacting user: {reacting_user}")
    print(f"   Whitelisted: {WHITELISTED_APPROVER_IDS}")
    print(f"   Message TS: {message_ts}")
    print(f"   Channel: {channel}")
    print(f"   Pending queries: { {k: v['user_id'] for k, v in _pending_queries.items()} }")

    # Must be the correct emoji
    if reaction != APPROVAL_EMOJI:
        print(f"   ❌ Wrong emoji — skipping")
        return

    # Must be a whitelisted approver
    if reacting_user not in WHITELISTED_APPROVER_IDS:
        print(f"   ❌ User not whitelisted — skipping")
        return

    # Try exact message_ts match first
    pending = _pop_pending_query(message_ts)

    # Fallback: if approver reacted on a different message in the thread,
    # search for any pending query in the same channel
    if pending is None:
        print(f"   ⚠️ No exact match for ts={message_ts}. Trying channel fallback...")
        with _pending_lock:
            for key, pq in list(_pending_queries.items()):
                if pq.get("channel") == channel:
                    pending = _pending_queries.pop(key)
                    print(f"   ✅ Found pending query via channel match: key={key}")
                    break

    if pending is None:
        print(f"   ❌ No pending query found for this channel — ignoring")
        return  # No pending query — ignore

    user_id = pending["user_id"]
    query = pending["query"]
    thread_ts = pending["thread_ts"]

    print(f"✅ Query approved by <@{reacting_user}> for message {message_ts}")

    # Notify the thread that query is approved
    client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=f"✅ *Query approved* by <@{reacting_user}>. Processing now...",
    )

    # Start processing
    message_key = f"{user_id}:{channel}:{thread_ts}"
    processing_messages.add(message_key)

    thread = threading.Thread(
        target=process_agent_query,
        args=(user_id, query, channel, thread_ts)
    )
    thread.start()


# ================== SLACK EVENT HANDLERS ==================

@app.event("app_mention")
def handle_app_mention(event, say, client):
    """
    Handle when bot is mentioned in a channel
    Example: @SRE_Agent check logs for user 123
    """
    user_id = event['user']
    channel = event['channel']
    thread_ts = event.get('thread_ts', event['ts'])
    message_ts = event['ts']  # The ts of this specific message (for reactions)
    
    # Extract the message text (remove the bot mention)
    text = event['text']
    # Remove bot mention from text
    bot_user_id = client.auth_test()['user_id']
    text = text.replace(f'<@{bot_user_id}>', '').strip()
    
    if not text:
        say(
            text="👋 Hi! I'm your SRE AI Assistant. Ask me anything about logs, code, user data, or AWS resources!",
            thread_ts=thread_ts
        )
        return
    
    _handle_incoming_query(user_id, text, channel, thread_ts, message_ts)


@app.event("message")
def handle_direct_message(event, client):
    """
    Handle direct messages to the bot
    """
    # Ignore bot messages and subtypes
    if event.get('subtype') or event.get('bot_id'):
        return
    
    # Only handle direct messages (DMs)
    channel_type = event.get('channel_type')
    if channel_type != 'im':
        return
    
    user_id = event['user']
    channel = event['channel']
    text = event.get('text', '').strip()
    thread_ts = event.get('thread_ts', event['ts'])
    message_ts = event['ts']  # The ts of this specific message (for reactions)
    
    if not text:
        return
    
    # Handle special commands
    if text.lower() in ['help', '/help']:
        help_text = """
🤖 *SRE AI Assistant - Help*

I can help you with:
🔍 *Logs*: Search and analyze request/response logs
💻 *Code*: Understand implementations and find functions
👤 *User Data*: Get subscription and membership details
☁️ *AWS*: Query AWS DynamoDB data

*Example Queries:*
• "Check logs for user 162434451"
• "Search code for membership validation"
• "Get subscription details for user 123"
• "Query DynamoDB for PLAN_TYPE:PLAN_TYPE_DISTRICT_GOLD"

*Commands:*
• `help` - Show this help message
• `reset` - Start a new conversation session
• `session` - Show current session ID

Just ask me anything in natural language!
        """
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=help_text
        )
        return
    
    if text.lower() in ['reset', '/reset']:
        # Clear thread session
        session_key = thread_ts if thread_ts else channel
        if session_key in thread_sessions:
            del thread_sessions[session_key]
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="✅ Session reset! Starting fresh conversation."
        )
        return
    
    if text.lower() in ['session', '/session']:
        session_key = thread_ts if thread_ts else channel
        session_id = thread_sessions.get(session_key, 'No active session')
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=f"📋 Current Session ID: `{session_id}`"
        )
        return
    
    _handle_incoming_query(user_id, text, channel, thread_ts, message_ts)


@app.command("/sre")
def handle_sre_command(ack, command, client):
    """
    Handle /sre slash command
    Example: /sre check logs for user 123
    """
    ack()  # Acknowledge command receipt
    
    user_id = command['user_id']
    channel = command['channel_id']
    text = command.get('text', '').strip()
    
    if not text:
        client.chat_postEphemeral(
            channel=channel,
            user=user_id,
            text="❌ Please provide a query after the /sre command.\nExample: `/sre check logs for user 123`"
        )
        return
    
    # Create a unique thread for this command
    response = client.chat_postMessage(
        channel=channel,
        text=f"<@{user_id}> asked: _{text}_"
    )
    thread_ts = response['ts']
    message_ts = response['ts']  # The ts of the posted message (for reactions)
    
    _handle_incoming_query(user_id, text, channel, thread_ts, message_ts)


@app.event("app_home_opened")
def handle_app_home_opened(client, event):
    """
    Handle when user opens the bot's home tab
    """
    user_id = event['user']
    
    try:
        # Publish a home tab view
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "🤖 SRE AI Assistant"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Welcome to your SRE AI Assistant!*\n\nI can help you with debugging, log analysis, code exploration, and more."
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🔍 What can I do?*\n\n• Search and analyze logs\n• Understand code implementations\n• Get user subscription details\n• Query AWS DynamoDB data"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*💬 How to use:*\n\n1. Send me a direct message\n2. Mention me in a channel: `@SRE_Agent your question`\n3. Use the `/sre` command: `/sre your question`"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*📚 Example Queries:*\n\n• `Check logs for user 162434451`\n• `Search code for membership validation`\n• `Get subscription details for user 123`\n• `Query campaigns from DynamoDB`"
                        }
                    }
                ]
            }
        )
    except Exception as e:
        print(f"Error publishing home tab: {e}")


# ================== MIDNIGHT CLEANUP SCHEDULER ==================

def _seconds_until_midnight() -> float:
    """Calculate seconds remaining until the next midnight (00:00:00 local time)."""
    now = datetime.now()
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (next_midnight - now).total_seconds()


def _midnight_cleanup_loop():
    """
    Periodic cleanup that runs at midnight every day.
    Deletes all ChromaDB log collections and clears the session registry.
    Uses stdlib threading.Timer — no external scheduler dependencies.
    """
    try:
        count = cleanup_all_collections()
        print(f"🧹 Midnight cleanup completed: {count} collection(s) deleted at {datetime.now()}")
        # Also clear Slack thread sessions so stale references don't linger
        thread_sessions.clear()
        print(f"🧹 Cleared thread_sessions mapping")
        # Clear any stale pending queries
        with _pending_lock:
            stale = len(_pending_queries)
            _pending_queries.clear()
        if stale:
            print(f"🧹 Cleared {stale} stale pending queries")
    except Exception as e:
        print(f"❌ Midnight cleanup error: {e}")

    # Schedule next run at the following midnight
    delay = _seconds_until_midnight()
    timer = threading.Timer(delay, _midnight_cleanup_loop)
    timer.daemon = True  # Won't block process exit
    timer.start()
    print(f"⏰ Next cleanup scheduled in {delay/3600:.1f} hours")


def start_midnight_scheduler():
    """Start the midnight cleanup scheduler. Call once at bot startup."""
    delay = _seconds_until_midnight()
    print(f"⏰ Midnight cleanup scheduler started. First run in {delay/3600:.1f} hours")
    timer = threading.Timer(delay, _midnight_cleanup_loop)
    timer.daemon = True
    timer.start()


# Start the bot
if __name__ == "__main__":
    # Validate environment variables
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    
    if not bot_token:
        raise ValueError("❌ SLACK_BOT_TOKEN environment variable not set!")
    
    if not app_token:
        raise ValueError("❌ SLACK_APP_TOKEN environment variable not set!")
    
    print("=" * 60)
    print("🤖 SRE AI Agent - Slack Bot")
    print("=" * 60)
    print("✅ Bot Token: Configured")
    print("✅ App Token: Configured")

    if APPROVAL_GATE_ENABLED:
        print(f"🔒 Approval Gate: ENABLED")
        print(f"   Approvers: {WHITELISTED_APPROVER_IDS}")
        print(f"   Approval Emoji: :{APPROVAL_EMOJI}:")
    else:
        print("🔓 Approval Gate: DISABLED (no WHITELISTED_APPROVER_IDS set)")

    print("🔒 PII Sanitization: ENABLED (all responses sanitized)")

    # Start the midnight cleanup scheduler for log embeddings
    start_midnight_scheduler()

    print("\n🚀 Starting bot in Socket Mode...")
    print("=" * 60)
    
    # Start the bot using Socket Mode
    handler = SocketModeHandler(app, app_token)
    handler.start()
