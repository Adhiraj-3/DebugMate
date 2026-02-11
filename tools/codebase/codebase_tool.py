
"""
Codebase search tool for finding and analyzing code
"""

from typing import List
from langchain_core.tools import tool
from core.models import CodeSearchInput

# Retriever is set to None by default (disabled)
retriever = None

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
    from config import AI_GATEWAY_CONFIG
    from ai_gateway_llm import AIGatewayLLM
    llm = AIGatewayLLM(**AI_GATEWAY_CONFIG)
    
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
