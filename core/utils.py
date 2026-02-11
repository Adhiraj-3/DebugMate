
"""
Utility functions for SRE Agent core.
Includes PII sanitization middleware that uses LLM to strip personal information.
"""

from core.ai_gateway_llm import AIGatewayLLM
from core.config import AI_GATEWAY_CONFIG


def sanitize_pii(response_text: str) -> str:
    """
    Make a final LLM call to remove/mask all Personally Identifiable Information (PII)
    from the agent's response before it is sent back to the user.

    PII includes: Aadhaar numbers, PAN numbers, mobile/phone numbers, email addresses,
    full names, addresses, bank account numbers, credit/debit card numbers, etc.

    Args:
        response_text: The raw LLM response text.

    Returns:
        Sanitized text with PII masked/removed. Falls back to original text on error.
    """
    if not response_text or not response_text.strip():
        return response_text

    sanitization_prompt = (
        "You are a PII sanitization filter. Your ONLY job is to take the following text "
        "and return it with ALL Personally Identifiable Information (PII) masked.\n\n"
        "Rules:\n"
        "1. Replace Aadhaar numbers (12-digit) with [AADHAAR_MASKED]\n"
        "2. Replace PAN numbers (format: ABCDE1234F) with [PAN_MASKED]\n"
        "3. Replace mobile/phone numbers (10+ digits, may have +91 prefix) with [PHONE_MASKED]\n"
        "4. Replace email addresses with [EMAIL_MASKED]\n"
        "5. Replace full person names (first + last name or full name references) with [NAME_MASKED]\n"
        "6. Replace physical addresses with [ADDRESS_MASKED]\n"
        "7. Replace bank account numbers with [BANK_ACCOUNT_MASKED]\n"
        "8. Replace credit/debit card numbers with [CARD_MASKED]\n"
        "9. Replace passport numbers with [PASSPORT_MASKED]\n"
        "10. Do NOT change any technical content, log data, error messages, RPC method names, "
        "UUIDs, trace IDs, session IDs, or user IDs used for debugging.\n"
        "11. Do NOT add any commentary, explanation, or extra text. Return ONLY the sanitized version.\n"
        "12. Preserve all formatting (markdown, code blocks, bullet points, etc.) exactly as-is.\n\n"
        "Text to sanitize:\n"
        "---\n"
        f"{response_text}\n"
        "---\n\n"
        "Sanitized text:"
    )

    try:
        # Use a lightweight LLM call for sanitization
        sanitizer_llm = AIGatewayLLM(
            base_url=AI_GATEWAY_CONFIG["base_url"],
            project_name=AI_GATEWAY_CONFIG["project_name"],
            project_auth_key=AI_GATEWAY_CONFIG["project_auth_key"],
            model_name=AI_GATEWAY_CONFIG.get("model_name", "gpt-5"),
            temperature=0,  # Deterministic — no creativity needed
            max_completion_tokens=AI_GATEWAY_CONFIG.get("max_completion_tokens", 4096),
            timeout_ms=AI_GATEWAY_CONFIG.get("timeout_ms", 180000),
        )

        from langchain_core.messages import HumanMessage
        result = sanitizer_llm.invoke([HumanMessage(content=sanitization_prompt)])

        sanitized = result.content.strip()
        if sanitized:
            print("✅ PII sanitization completed successfully")
            return sanitized
        else:
            print("⚠️ PII sanitization returned empty — using original response")
            return response_text

    except Exception as e:
        print(f"⚠️ PII sanitization failed: {e}. Returning original response.")
        return response_text
