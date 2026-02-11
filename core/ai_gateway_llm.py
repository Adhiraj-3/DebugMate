
"""
Custom LangChain LLM wrapper for AI Gateway
Handles message format conversion, tool calling, and embeddings
"""

import json
import requests
from typing import Any, List, Optional, Dict, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field


class AIGatewayLLM(BaseChatModel):
    """Custom LLM that calls AI Gateway API"""
    
    model_name: str = Field(default="")
    base_url: str = Field(default="")
    project_name: str = Field(default="")
    project_auth_key: str = Field(default="")
    temperature: float = Field(default=0.7)
    max_completion_tokens: int = Field(default=2048)
    timeout_ms: int = Field(default=30000)
    bound_tools: Optional[List[Dict]] = Field(default=None)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "ai_gateway"
    
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "AIGatewayLLM":
        """Bind tools to this LLM."""
        from langchain_core.utils.function_calling import convert_to_openai_tool
        
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        # Create a copy with tools bound
        return self.__class__(
            model_name=self.model_name,
            base_url=self.base_url,
            project_name=self.project_name,
            project_auth_key=self.project_auth_key,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            timeout_ms=self.timeout_ms,
            bound_tools=formatted_tools
        )
    
    def _convert_message_to_gateway_format(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert LangChain message to AI Gateway format"""
        
        # Map role - AI Gateway expects string values
        role_map = {
            "system": "SYSTEM",
            "human": "USER",
            "ai": "ASSISTANT",
            "tool": "TOOL",
        }
        
        role = role_map.get(message.type, "USER")
        
        # Handle content
        content_parts = []
        
        # Check if message has tool calls (AI message with tool calls)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # This is an AI message requesting tool calls
            tool_uses = []
            for tool_call in message.tool_calls:
                tool_uses.append({
                    "id": tool_call.get("id", ""),
                    "type": "FUNCTION",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "parameters": tool_call.get("args", {})
                    }
                })
            
            content_parts.append({
                "type": "TOOL_USE",
                "toolUses": tool_uses  # Note: camelCase
            })
        
        # Check if this is a tool response message
        elif isinstance(message, ToolMessage):
            # Tool result message
            tool_uses = [{
                "id": message.tool_call_id,
                "type": "FUNCTION",
                "function": {
                    "name": message.name if hasattr(message, 'name') else "",
                    "result": [{
                        "type": "TEXT",
                        "data": message.content
                    }]
                }
            }]
            
            content_parts.append({
                "type": "TOOL_USE",
                "toolUses": tool_uses  # Note: camelCase
            })
        
        else:
            # Regular text message
            content_parts.append({
                "type": "TEXT",
                "data": str(message.content)
            })
        
        return {
            "role": role,
            "content": content_parts
        }
    
    def _convert_tools_to_gateway_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert LangChain tools to AI Gateway format"""
        gateway_tools = []
        
        for tool in tools:
            # Extract function schema
            if "function" in tool:
                func = tool["function"]
            else:
                func = tool
            
            gateway_tools.append({
                "type": "FUNCTION",  # String value
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                }
            })
        
        return gateway_tools
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using AI Gateway"""
        
        # Convert messages
        gateway_messages = [
            self._convert_message_to_gateway_format(msg) 
            for msg in messages
        ]
        
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": gateway_messages,
            "config": {
                "temperature": self.temperature,
                "max_completion_tokens": self.max_completion_tokens,
            },
            "client_options": {
                "retry_options": {
                    "max_retries": 1
                },
                "request_options": {
                    "content_type": "application/json",
                    "timeout_ms": self.timeout_ms
                },
                "source": "sre_agent"
            }
        }
        
        # Add tools if provided (either from kwargs or bound_tools)
        tools_to_use = kwargs.get("tools") or self.bound_tools
        if tools_to_use:
            gateway_tools = self._convert_tools_to_gateway_format(tools_to_use)
            payload["tool_option"] = {
                "tools": gateway_tools,
                "tool_choice": {
                    "mode": "AUTO"  # String value
                }
            }
        
        # Make API call
        headers = {
            "Content-Type": "application/json",
            "Grpc-Metadata-x-project-name": self.project_name,
            "Grpc-Metadata-x-project-auth-key": self.project_auth_key,
            "Keep-Alive": "timeout=60"
        }
        
        url = f"{self.base_url}/chat/completion"
        
        try:
            # Debug logging
            print(f"\n[AI Gateway Request]")
            print(f"URL: {url}")
            print(f"Model: {self.model_name}")
            print(f"Messages count: {len(gateway_messages)}")
            print(f"Has tools: {bool(tools_to_use)}")
            print(f"\nPayload (first 500 chars): {json.dumps(payload)[:500]}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_ms / 1000
            )
            
            # Check response before raising
            if response.status_code != 200:
                error_text = response.text[:1000]
                print(f"\n[AI Gateway Error]")
                print(f"Status: {response.status_code}")
                print(f"Response: {error_text}")
                raise Exception(f"API returned {response.status_code}: {error_text}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"[AI Gateway Response] Success!")
            print(f"Response keys: {list(result.keys())}")
            print(f"Full response: {json.dumps(result, indent=2)[:1000]}")
            
            # AI Gateway response structure:
            # {
            #   "choices": [
            #     {
            #       "role": 3,  # ASSISTANT
            #       "content": [
            #         {"type": 1, "data": "text content"}
            #         or
            #         {"type": 6, "tool_uses": [...]}
            #       ]
            #     }
            #   ]
            # }
            
            choices = result.get("choices", [])
            if not choices:
                raise Exception("No choices in AI Gateway response")
            
            choice = choices[0]
            content_parts = choice.get("content", [])
            
            # Check for tool calls
            tool_calls = []
            text_content = ""
            
            for part in content_parts:
                part_type = part.get("type")
                
                # AI Gateway uses string types: "TOOL_USE", "TEXT", etc.
                if part_type == "TOOL_USE" or part_type == 6:
                    # Extract tool calls - field name is "toolUses" not "tool_uses"
                    for tool_use in part.get("toolUses", part.get("tool_uses", [])):
                        # Get parameters - might be a string or dict
                        params = tool_use.get("function", {}).get("parameters", {})
                        
                        # If parameters is a string, parse it as JSON
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse parameters as JSON: {params}")
                                params = {}
                        
                        tool_calls.append({
                            "name": tool_use.get("function", {}).get("name", ""),
                            "args": params,
                            "id": tool_use.get("id", "")
                        })
                elif part_type == "TEXT" or part_type == 1:
                    text_content += part.get("data", "")
            
            # Create AI message
            if tool_calls:
                ai_message = AIMessage(
                    content=text_content if text_content else "",
                    tool_calls=tool_calls
                )
            else:
                ai_message = AIMessage(content=text_content if text_content else "")
            
            generation = ChatGeneration(message=ai_message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise Exception(f"AI Gateway API error: {str(e)}")
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream chat completion (not implemented, falls back to _generate)"""
        # For simplicity, fall back to non-streaming
        result = self._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            yield gen




class AIGatewayEmbeddings(Embeddings):
    """Custom embeddings that call AI Gateway API"""
    
    model_name: str = "text-embed-3-large"
    base_url: str = ""
    project_name: str = ""
    project_auth_key: str = ""
    timeout_ms: int = 30000
    output_dim: Optional[int] = None
    
    def __init__(
        self,
        model_name: str = "text-embed-3-large",
        base_url: str = "",
        project_name: str = "",
        project_auth_key: str = "",
        timeout_ms: int = 30000,
        output_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.base_url = base_url
        self.project_name = project_name
        self.project_auth_key = project_auth_key
        self.timeout_ms = timeout_ms
        self.output_dim = output_dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using AI Gateway"""
        return self._embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using AI Gateway"""
        return self._embed([text])[0]
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Internal method to call AI Gateway embeddings API"""
        
        # Build request payload
        inputs = [{"text": text} for text in texts]
        
        payload = {
            "model": self.model_name,
            "inputs": inputs,
            "config": {
                "encoding": "FLOAT32_ARRAY",
                "normalization": "Normalization_L2"
            },
            "client_options": {
                "source": "sre_agent"
            }
        }
        
        # Add output_dim if specified
        if self.output_dim:
            payload["config"]["output_dim"] = self.output_dim
        
        # Make API call
        headers = {
            "Content-Type": "application/json",
            "Grpc-Metadata-x-project-name": self.project_name,
            "Grpc-Metadata-x-project-auth-key": self.project_auth_key,
            "Keep-Alive": "timeout=60"
        }
        
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_ms / 1000
            )
            
            # Debug logging
            if response.status_code != 200:
                error_text = response.text[:500]  # First 500 chars
                raise Exception(
                    f"API returned {response.status_code}: {error_text}"
                )
            
            response.raise_for_status()
            result = response.json()
            
            # AI Gateway embeddings response can have different structures
            # Try multiple response formats
            embeddings = []
            
            # Format 1: {"embeddings": [{"values": [...], "id": "0"}, ...]}
            # This is the primary AI Gateway format
            if "embeddings" in result:
                for item in result["embeddings"]:
                    values = item.get("values", [])
                    if values:
                        embeddings.append(values)
            
            # Format 2: {"outputs": [{"embedding": [...]}, ...]}
            if not embeddings and "outputs" in result:
                for output in result["outputs"]:
                    embedding = output.get("embedding", [])
                    if embedding:
                        embeddings.append(embedding)
            
            # Format 3: {"data": [{"embedding": [...]}, ...]}
            if not embeddings and "data" in result:
                for item in result["data"]:
                    embedding = item.get("embedding", [])
                    if embedding:
                        embeddings.append(embedding)
            
            # Format 4: Direct array
            if not embeddings and isinstance(result, list):
                embeddings = result
            
            if not embeddings:
                raise Exception(f"No embeddings found in response. Keys: {list(result.keys())}")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"AI Gateway Embeddings API error: {str(e)}")
        except Exception as e:
            raise Exception(f"AI Gateway Embeddings API error: {str(e)}")
