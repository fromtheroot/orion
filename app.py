import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import streamlit as st
import httpx
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP


class ChatMessage(BaseModel):
    """Structure for chat messages"""
    role: str = Field(description="Role of the message sender")
    content: str = Field(description="Content of the message")


class AgentResponse(BaseModel):
    """Structure for agent responses"""
    response: str = Field(description="The agent's response")


class PydanticAIWebApp:
    """Main application class for the Pydantic AI Web UI"""
    
    def __init__(self, system_prompt: Optional[str] = None, model_name: str = 'qwen3:14b', 
                 ollama_url: str = 'http://localhost:11434', mcp_url: Optional[str] = None, 
                 mcp_token: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        # Store settings
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Initialize Ollama model with OpenAI compatibility
        ollama_base_url = f"{ollama_url}/v1"
        self.ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=ollama_base_url)
        )
        
        # Initialize MCP server if configured (UI settings override env)
        self.mcp_server = None
        self.mcp_status = None
        
        # Use provided settings or fall back to environment variables
        final_mcp_url = mcp_url or os.getenv('MCP_SERVER_URL')
        final_mcp_token = mcp_token or os.getenv('MCP_BEARER_TOKEN')
        
        if final_mcp_url and final_mcp_token:
            try:
                # Create headers for bearer authentication
                headers = {"Authorization": f"Bearer {final_mcp_token}"}
                self.mcp_server = MCPServerHTTP(url=final_mcp_url, headers=headers)
                self.mcp_status = "✅ MCP server configured successfully"
            except Exception as e:
                self.mcp_status = f"❌ MCP server configuration failed: {str(e)}"
        else:
            self.mcp_status = "⚠️ MCP server not configured (missing URL or token)"
        
        # Default system prompt if none provided
        default_system_prompt = """You are a helpful AI assistant powered by Pydantic AI and running on Ollama.
You provide accurate, helpful, and engaging responses to user questions.
Always be polite, informative, and concise in your responses.
If you're uncertain about something, acknowledge the uncertainty.

You have access to external tools through MCP (Model Context Protocol) when available.
When using tools, always explain what you're doing and what the tool returned.

When responding to complex questions, feel free to show your thinking process by using phrases like:
- "Let me think about this..."
- "I need to consider..."
- "First, I should..."
- "Let me analyze..."

This helps users understand your reasoning process."""
        
        # Create the agent with MCP server (if available) and system prompt
        mcp_servers = [self.mcp_server] if self.mcp_server else []
        self.agent = Agent(
            model=self.ollama_model,
            system_prompt=system_prompt or default_system_prompt,
            mcp_servers=mcp_servers
        )
        
        # Store the system prompt for reference
        self.system_prompt = system_prompt or default_system_prompt
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt and recreate the agent"""
        self.system_prompt = new_system_prompt
        mcp_servers = [self.mcp_server] if self.mcp_server else []
        self.agent = Agent(
            model=self.ollama_model,
            system_prompt=new_system_prompt,
            mcp_servers=mcp_servers
        )
    
    def update_settings(self, model_name: str = None, ollama_url: str = None, 
                       mcp_url: str = None, mcp_token: str = None, system_prompt: str = None):
        """Update any combination of settings and recreate the agent"""
        # Update model settings if provided
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            
        if ollama_url and ollama_url != self.ollama_url:
            self.ollama_url = ollama_url
            
        # Recreate Ollama model if model or URL changed
        if model_name or ollama_url:
            ollama_base_url = f"{self.ollama_url}/v1"
            self.ollama_model = OpenAIModel(
                model_name=self.model_name,
                provider=OpenAIProvider(base_url=ollama_base_url)
            )
        
        # Update MCP settings if provided
        if mcp_url is not None or mcp_token is not None:
            # Load environment variables as fallback
            load_dotenv()
            final_mcp_url = mcp_url if mcp_url else os.getenv('MCP_SERVER_URL')
            final_mcp_token = mcp_token if mcp_token else os.getenv('MCP_BEARER_TOKEN')
            
            self.mcp_server = None
            if final_mcp_url and final_mcp_token:
                try:
                    headers = {"Authorization": f"Bearer {final_mcp_token}"}
                    self.mcp_server = MCPServerHTTP(url=final_mcp_url, headers=headers)
                    self.mcp_status = "✅ MCP server configured successfully"
                except Exception as e:
                    self.mcp_status = f"❌ MCP server configuration failed: {str(e)}"
            else:
                self.mcp_status = "⚠️ MCP server not configured (missing URL or token)"
        
        # Update system prompt if provided
        if system_prompt:
            self.system_prompt = system_prompt
        
        # Recreate agent with updated settings
        mcp_servers = [self.mcp_server] if self.mcp_server else []
        self.agent = Agent(
            model=self.ollama_model,
            system_prompt=self.system_prompt,
            mcp_servers=mcp_servers
        )
    
    async def get_agent_response(self, message: str, conversation_history: List[ChatMessage]) -> str:
        """
        Get response from the agent
        
        Args:
            message: User's input message
            conversation_history: Previous conversation messages
            
        Returns:
            Response string
        """
        try:
            # Use MCP server context if available
            if self.mcp_server:
                async with self.agent.run_mcp_servers():
                    result = await self.agent.run(user_prompt=message)
                    return str(result.output)
            else:
                # Get response from agent without MCP servers
                result = await self.agent.run(user_prompt=message)
                return str(result.output)
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def get_streaming_response(self, message: str, conversation_history: List[ChatMessage]):
        """
        Get streaming response from the agent with thinking/response separation
        
        Args:
            message: User's input message
            conversation_history: Previous conversation messages
            
        Yields:
            Dict with 'type' (thinking/response), 'content', and 'status'
        """
        try:
            # Use MCP server context for streaming if available
            if self.mcp_server:
                async with self.agent.run_mcp_servers():
                    async with self.agent.run_stream(user_prompt=message) as result:
                        full_content = ""
                        thinking_phase = True
                        thinking_content = ""
                        response_content = ""
                        thinking_markers_found = False
                        
                        async for chunk in result.stream():
                            # Get the current content from chunk
                            if hasattr(chunk, 'response'):
                                current_content = str(chunk.response) if chunk.response else ""
                            elif hasattr(chunk, 'content'):
                                current_content = str(chunk.content) if chunk.content else ""
                            else:
                                current_content = str(chunk) if chunk else ""
                            
                            # Only process if we have new content
                            if current_content and len(current_content) > len(full_content):
                                new_chunk = current_content[len(full_content):]
                                full_content = current_content
                                
                                if new_chunk.strip():
                                    # Parse thinking tags properly
                                    lower_content = full_content.lower()
                                    
                                    # Check for think tags in the content
                                    think_start = '<think>'
                                    think_end = '</think>'
                                    
                                    if think_start in lower_content and think_end in lower_content:
                                        # We have complete thinking tags, parse them
                                        start_idx = lower_content.find(think_start)
                                        end_idx = lower_content.find(think_end)
                                        
                                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                            # Extract thinking content (between tags)
                                            thinking_start_pos = start_idx + len(think_start)
                                            extracted_thinking = full_content[thinking_start_pos:end_idx]
                                            
                                            # Extract response content (after end tag)
                                            response_start_pos = end_idx + len(think_end)
                                            extracted_response = full_content[response_start_pos:].strip()
                                            
                                            # If we haven't sent thinking content yet, send it all
                                            if not thinking_content and extracted_thinking:
                                                thinking_content = extracted_thinking
                                                yield {
                                                    'type': 'thinking',
                                                    'content': extracted_thinking,
                                                    'status': 'streaming'
                                                }
                                                yield {
                                                    'type': 'thinking_complete',
                                                    'content': '',
                                                    'status': 'complete'
                                                }
                                                thinking_phase = False
                                                thinking_markers_found = True
                                            
                                            # Send response content if we have any and it's new
                                            if extracted_response and len(extracted_response) > len(response_content):
                                                new_response_chunk = extracted_response[len(response_content):]
                                                response_content = extracted_response
                                                yield {
                                                    'type': 'response',
                                                    'content': new_response_chunk,
                                                    'status': 'streaming'
                                                }
                                            continue
                                    
                                    elif think_start in lower_content and think_end not in lower_content:
                                        # We have start tag but no end tag yet, extract thinking content so far
                                        start_idx = lower_content.find(think_start)
                                        if start_idx != -1:
                                            thinking_start_pos = start_idx + len(think_start)
                                            current_thinking = full_content[thinking_start_pos:]
                                            
                                            # Send new thinking content
                                            if len(current_thinking) > len(thinking_content):
                                                new_thinking_chunk = current_thinking[len(thinking_content):]
                                                thinking_content = current_thinking
                                                yield {
                                                    'type': 'thinking',
                                                    'content': new_thinking_chunk,
                                                    'status': 'streaming'
                                                }
                                                thinking_markers_found = True
                                            continue
                                    
                                    # Fallback: if no think tags found, use natural language detection
                                    natural_thinking_indicators = [
                                        'let me think', 'i need to think', 'analyzing', 'considering',
                                        'first, i should', 'let me consider', 'i should'
                                    ]
                                    has_natural_thinking = any(marker in lower_content for marker in natural_thinking_indicators)
                                    
                                    if has_natural_thinking and not thinking_markers_found:
                                        thinking_markers_found = True
                                    
                                    # Another fallback: if we haven't found any thinking markers and have enough content
                                    if not thinking_markers_found and thinking_phase and len(full_content) > 150:
                                        thinking_phase = False
                                        yield {
                                            'type': 'thinking_complete',
                                            'content': '',
                                            'status': 'complete'
                                        }
                                    
                                    # Emit the appropriate chunk type for fallback cases
                                    if thinking_phase and not thinking_markers_found:
                                        thinking_content += new_chunk
                                        yield {
                                            'type': 'thinking',
                                            'content': new_chunk,
                                            'status': 'streaming'
                                        }
                                    elif not thinking_markers_found:
                                        response_content += new_chunk
                                        yield {
                                            'type': 'response',
                                            'content': new_chunk,
                                            'status': 'streaming'
                                        }
                        
                        # Ensure thinking is marked complete if it wasn't already
                        if thinking_phase:
                            yield {
                                'type': 'thinking_complete',
                                'content': '',
                                'status': 'complete'
                            }
                        
                        # Mark final completion
                        yield {
                            'type': 'response_complete',
                            'content': '',
                            'status': 'complete'
                        }
            else:
                # No MCP server - same logic but without MCP context
                async with self.agent.run_stream(user_prompt=message) as result:
                    full_content = ""
                    thinking_phase = True
                    thinking_content = ""
                    response_content = ""
                    thinking_markers_found = False
                    
                    async for chunk in result.stream():
                        # Get the current content from chunk
                        if hasattr(chunk, 'response'):
                            current_content = str(chunk.response) if chunk.response else ""
                        elif hasattr(chunk, 'content'):
                            current_content = str(chunk.content) if chunk.content else ""
                        else:
                            current_content = str(chunk) if chunk else ""
                        
                        # Only process if we have new content
                        if current_content and len(current_content) > len(full_content):
                            new_chunk = current_content[len(full_content):]
                            full_content = current_content
                            
                            if new_chunk.strip():
                                # Parse thinking tags properly
                                lower_content = full_content.lower()
                                
                                # Check for think tags in the content
                                think_start = '<think>'
                                think_end = '</think>'
                                
                                if think_start in lower_content and think_end in lower_content:
                                    # We have complete thinking tags, parse them
                                    start_idx = lower_content.find(think_start)
                                    end_idx = lower_content.find(think_end)
                                    
                                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                        # Extract thinking content (between tags)
                                        thinking_start_pos = start_idx + len(think_start)
                                        extracted_thinking = full_content[thinking_start_pos:end_idx]
                                        
                                        # Extract response content (after end tag)
                                        response_start_pos = end_idx + len(think_end)
                                        extracted_response = full_content[response_start_pos:].strip()
                                        
                                        # If we haven't sent thinking content yet, send it all
                                        if not thinking_content and extracted_thinking:
                                            thinking_content = extracted_thinking
                                            yield {
                                                'type': 'thinking',
                                                'content': extracted_thinking,
                                                'status': 'streaming'
                                            }
                                            yield {
                                                'type': 'thinking_complete',
                                                'content': '',
                                                'status': 'complete'
                                            }
                                            thinking_phase = False
                                            thinking_markers_found = True
                                        
                                        # Send response content if we have any and it's new
                                        if extracted_response and len(extracted_response) > len(response_content):
                                            new_response_chunk = extracted_response[len(response_content):]
                                            response_content = extracted_response
                                            yield {
                                                'type': 'response',
                                                'content': new_response_chunk,
                                                'status': 'streaming'
                                            }
                                        continue
                                
                                elif think_start in lower_content and think_end not in lower_content:
                                    # We have start tag but no end tag yet, extract thinking content so far
                                    start_idx = lower_content.find(think_start)
                                    if start_idx != -1:
                                        thinking_start_pos = start_idx + len(think_start)
                                        current_thinking = full_content[thinking_start_pos:]
                                        
                                        # Send new thinking content
                                        if len(current_thinking) > len(thinking_content):
                                            new_thinking_chunk = current_thinking[len(thinking_content):]
                                            thinking_content = current_thinking
                                            yield {
                                                'type': 'thinking',
                                                'content': new_thinking_chunk,
                                                'status': 'streaming'
                                            }
                                            thinking_markers_found = True
                                        continue
                                
                                # Fallback: if no think tags found, use natural language detection
                                natural_thinking_indicators = [
                                    'let me think', 'i need to think', 'analyzing', 'considering',
                                    'first, i should', 'let me consider', 'i should'
                                ]
                                has_natural_thinking = any(marker in lower_content for marker in natural_thinking_indicators)
                                
                                if has_natural_thinking and not thinking_markers_found:
                                    thinking_markers_found = True
                                
                                # Another fallback: if we haven't found any thinking markers and have enough content
                                if not thinking_markers_found and thinking_phase and len(full_content) > 150:
                                    thinking_phase = False
                                    yield {
                                        'type': 'thinking_complete',
                                        'content': '',
                                        'status': 'complete'
                                    }
                                
                                # Emit the appropriate chunk type for fallback cases
                                if thinking_phase and not thinking_markers_found:
                                    thinking_content += new_chunk
                                    yield {
                                        'type': 'thinking',
                                        'content': new_chunk,
                                        'status': 'streaming'
                                    }
                                elif not thinking_markers_found:
                                    response_content += new_chunk
                                    yield {
                                        'type': 'response',
                                        'content': new_chunk,
                                        'status': 'streaming'
                                    }
                    
                    # Ensure thinking is marked complete if it wasn't already
                    if thinking_phase:
                        yield {
                            'type': 'thinking_complete',
                            'content': '',
                            'status': 'complete'
                        }
                    
                    # Mark final completion
                    yield {
                        'type': 'response_complete',
                        'content': '',
                        'status': 'complete'
                    }
                        
        except Exception as e:
            yield {
                'type': 'error',
                'content': f"Sorry, I encountered an error: {str(e)}",
                'status': 'error'
            }


def save_settings_to_file(settings: Dict[str, Any], filename: str = "orion_settings.json"):
    """Save settings to a JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save settings: {e}")
        return False


def load_settings_from_file(filename: str = "orion_settings.json") -> Dict[str, Any]:
    """Load settings from a JSON file"""
    default_settings = {
        "ollama_model": "qwen3:14b",
        "ollama_url": "http://localhost:11434",
        "mcp_url": "",
        "mcp_token": "",
        "system_prompt": """You are a helpful AI assistant powered by Pydantic AI and running on Ollama.
You provide accurate, helpful, and engaging responses to user questions.
Always be polite, informative, and concise in your responses.
If you're uncertain about something, acknowledge the uncertainty.

You have access to external tools through MCP (Model Context Protocol) when available.
When using tools, always explain what you're doing and what the tool returned.

When responding to complex questions, feel free to show your thinking process by using phrases like:
- "Let me think about this..."
- "I need to consider..."
- "First, I should..."
- "Let me analyze..."

This helps users understand your reasoning process."""
    }
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                saved_settings = json.load(f)
            # Merge with defaults to handle missing keys
            for key, default_value in default_settings.items():
                if key not in saved_settings:
                    saved_settings[key] = default_value
            return saved_settings
        else:
            return default_settings
    except Exception as e:
        st.warning(f"Failed to load settings file, using defaults: {e}")
        return default_settings


def get_available_ollama_models(server_url: str = "http://localhost:11434"):
    """Get list of available Ollama models"""
    try:
        response = httpx.get(f"{server_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        else:
            return []
    except Exception:
        return []


def check_ollama_connection(server_url: str = "http://localhost:11434", model_name: str = "qwen3:14b"):
    """Check if Ollama server is running and model is available"""
    try:
        response = httpx.get(f"{server_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if model_name in model_names:
                return True, f"✅ Ollama server is running and {model_name} model is available!"
            else:
                return False, f"⚠️ Ollama server is running but {model_name} model not found. Available models: {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}"
        else:
            return False, "❌ Ollama server is not responding properly"
            
    except Exception as e:
        return False, f"❌ Cannot connect to Ollama server: {e}. Make sure Ollama is running: `ollama serve`"


def auto_focus_chat_input():
    """Inject JavaScript to auto-focus the chat input"""
    st.markdown("""
    <script>
    // Auto-focus chat input - more aggressive approach
    (function() {
        let focusAttempts = 0;
        const maxAttempts = 100;
        
        function tryFocusInput() {
            if (focusAttempts >= maxAttempts) {
                console.log('Max focus attempts reached');
                return;
            }
            
            // Try multiple selectors
            const selectors = [
                'textarea[data-testid="stChatInputTextArea"]',
                'textarea[placeholder="Type your message here..."]',
                'textarea[aria-label="Type your message here..."]',
                'textarea[placeholder*="message"]',
                'div[data-testid="stChatInput"] textarea'
            ];
            
            for (let selector of selectors) {
                const input = document.querySelector(selector);
                if (input && input.offsetParent !== null) { // Check if element is visible
                    input.focus();
                    input.click(); // Also trigger click in case focus alone doesn't work
                    console.log('Successfully focused with selector:', selector);
                    return true;
                }
            }
            
            focusAttempts++;
            setTimeout(tryFocusInput, 50); // Try again in 50ms
            return false;
        }
        
        // Start trying immediately
        tryFocusInput();
        
        // Try again when DOM changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    setTimeout(tryFocusInput, 10);
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Also try when document is ready and loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', tryFocusInput);
        } else {
            tryFocusInput();
        }
        
        window.addEventListener('load', tryFocusInput);
        
        // Try every second for the first 10 seconds as a fallback
        let intervalCount = 0;
        const focusInterval = setInterval(function() {
            tryFocusInput();
            intervalCount++;
            if (intervalCount >= 20) { // Stop after 10 seconds
                clearInterval(focusInterval);
            }
        }, 500);
    })();
    </script>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load settings from file (this will initialize all settings)
    if "settings_loaded" not in st.session_state:
        saved_settings = load_settings_from_file()
        
        # Initialize all settings from saved file or defaults
        st.session_state.system_prompt = saved_settings["system_prompt"]
        st.session_state.ollama_model = saved_settings["ollama_model"]
        st.session_state.ollama_url = saved_settings["ollama_url"]
        
        # For MCP settings, prefer saved settings, but fall back to env vars if saved settings are empty
        st.session_state.mcp_url = saved_settings["mcp_url"] or os.getenv('MCP_SERVER_URL', '')
        st.session_state.mcp_token = saved_settings["mcp_token"] or os.getenv('MCP_BEARER_TOKEN', '')
        
        st.session_state.settings_loaded = True
    
    # Runtime-only settings (not persisted)
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    
    # App instance
    if "app" not in st.session_state:
        st.session_state.app = PydanticAIWebApp(
            system_prompt=st.session_state.system_prompt,
            model_name=st.session_state.ollama_model,
            ollama_url=st.session_state.ollama_url,
            mcp_url=st.session_state.mcp_url,
            mcp_token=st.session_state.mcp_token
        )
    
    # Connection tracking
    if "connection_checked" not in st.session_state:
        st.session_state.connection_checked = False
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = None


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="🤖 Pydantic AI Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stChat > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5 !important;
        border-left: 4px solid #9c27b0;
    }
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-1px);
    }
    
    /* Thinking UI Styles */
    .thinking-status {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 14px;
        margin-bottom: 10px;
        display: inline-block;
    }
    
    .thinking-expander {
        border: 1px solid #e0e4e7;
        border-radius: 8px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
    }
    
    .thinking-content {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
        color: #495057;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    
    /* Response content styling */
    .response-content {
        margin-top: 10px;
        line-height: 1.6;
    }
    
    /* Spinner Animation */
    .spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Alternative dot spinner */
    .dot-spinner {
        display: inline-block;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .dot-spinner:after {
        content: '⠋';
        animation: dots 1.2s linear infinite;
        font-family: monospace;
        font-size: 16px;
    }
    
    @keyframes dots {
        0% { content: '⠋'; }
        12.5% { content: '⠙'; }
        25% { content: '⠹'; }
        37.5% { content: '⠸'; }
        50% { content: '⠼'; }
        62.5% { content: '⠴'; }
        75% { content: '⠦'; }
        87.5% { content: '⠧'; }
        100% { content: '⠇'; }
    }
    </style>
    

    """, unsafe_allow_html=True)
    
    # Header
    st.title("Orion")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Information")
        
        # Connection status
        if not st.session_state.connection_checked:
            with st.spinner("Checking Ollama connection..."):
                is_connected, status_message = check_ollama_connection(
                    server_url=st.session_state.ollama_url,
                    model_name=st.session_state.ollama_model
                )
                st.session_state.connection_status = (is_connected, status_message)
                st.session_state.connection_checked = True
        
        is_connected, status_message = st.session_state.connection_status
        
        if is_connected:
            st.success(status_message)
        else:
            st.error(status_message)
            st.markdown(f"""
            **Setup Steps:**
            1. Install Ollama: [ollama.com](https://ollama.com)
            2. Start server: `ollama serve`
            3. Pull model: `ollama pull {st.session_state.ollama_model}`
            4. Check server URL: {st.session_state.ollama_url}
            """)
        
        st.divider()
        
        # Ollama Configuration
        st.subheader("🤖 Ollama Configuration")
        
        # Ollama URL
        ollama_url_input = st.text_input(
            "Ollama Server URL:",
            value=st.session_state.ollama_url,
            help="URL of your Ollama server",
            key="ollama_url_input"
        )
        
        # Refresh models button and model selection
        col1, col2 = st.columns([2, 1])
        with col2:
            refresh_models = st.button("🔄 Refresh", use_container_width=True, key="refresh_models_btn")
        
        # Get available models
        if refresh_models or not st.session_state.available_models:
            with st.spinner("Loading available models..."):
                st.session_state.available_models = get_available_ollama_models(ollama_url_input)
        
        with col1:
            if st.session_state.available_models:
                # Find current model in list or use first available
                current_model_index = 0
                if st.session_state.ollama_model in st.session_state.available_models:
                    current_model_index = st.session_state.available_models.index(st.session_state.ollama_model)
                
                ollama_model_input = st.selectbox(
                    "Model:",
                    options=st.session_state.available_models,
                    index=current_model_index,
                    help="Select an available Ollama model",
                    key="ollama_model_input"
                )
            else:
                ollama_model_input = st.text_input(
                    "Model:",
                    value=st.session_state.ollama_model,
                    help="Model name (couldn't load available models)",
                    key="ollama_model_input_fallback"
                )
        
        st.divider()
        
        # MCP Configuration
        st.subheader("🔗 MCP Configuration")
        
        mcp_url_input = st.text_input(
            "MCP Server URL:",
            value=st.session_state.mcp_url,
            help="URL of your MCP server (optional)",
            key="mcp_url_input"
        )
        
        mcp_token_input = st.text_input(
            "MCP Bearer Token:",
            value=st.session_state.mcp_token,
            type="password",
            help="Bearer token for MCP authentication (optional)",
            key="mcp_token_input"
        )
        
        # Apply Settings Button
        settings_changed = (
            ollama_url_input != st.session_state.ollama_url or
            (st.session_state.available_models and ollama_model_input != st.session_state.ollama_model) or
            (not st.session_state.available_models and ollama_model_input != st.session_state.ollama_model) or
            mcp_url_input != st.session_state.mcp_url or
            mcp_token_input != st.session_state.mcp_token
        )
        
        if st.button("🚀 Apply Settings", use_container_width=True, type="primary", disabled=not settings_changed, key="apply_settings_btn"):
            # Update session state
            st.session_state.ollama_url = ollama_url_input
            if st.session_state.available_models:
                st.session_state.ollama_model = ollama_model_input
            else:
                st.session_state.ollama_model = ollama_model_input
            st.session_state.mcp_url = mcp_url_input
            st.session_state.mcp_token = mcp_token_input
            
            # Save settings to file for persistence
            settings_to_save = {
                "ollama_model": st.session_state.ollama_model,
                "ollama_url": st.session_state.ollama_url,
                "mcp_url": st.session_state.mcp_url,
                "mcp_token": st.session_state.mcp_token,
                "system_prompt": st.session_state.system_prompt
            }
            save_settings_to_file(settings_to_save)
            
            # Update the app instance
            st.session_state.app.update_settings(
                model_name=st.session_state.ollama_model,
                ollama_url=st.session_state.ollama_url,
                mcp_url=st.session_state.mcp_url,
                mcp_token=st.session_state.mcp_token
            )
            
            # Force connection recheck
            st.session_state.connection_checked = False
            
            st.success("✅ Settings updated and saved successfully!")
            st.rerun()
        
        st.divider()
        
        # System Prompt Configuration
        st.subheader("🎯 System Prompt")
        
        # Show current active system prompt
        with st.expander("📋 Current Active System Prompt", expanded=False):
            st.code(st.session_state.system_prompt, language="text")
        
        # Text area for editing
        system_prompt_draft = st.text_area(
            "Edit system prompt:",
            value=st.session_state.system_prompt,
            height=150,
            help="This defines how the AI assistant will behave and respond to users.",
            key="system_prompt_input"
        )
        
        # Button to apply changes
        col1, col2 = st.columns([2, 1])
        with col1:
            apply_clicked = st.button("🚀 Apply System Prompt", use_container_width=True, type="primary")
            clear_history = st.checkbox("Clear chat history", value=True, help="Recommended when changing system prompt", key="clear_history_checkbox")
            
            if apply_clicked:
                if system_prompt_draft.strip() != st.session_state.system_prompt.strip():
                    st.session_state.system_prompt = system_prompt_draft.strip()
                    
                    # Save settings to file for persistence
                    settings_to_save = {
                        "ollama_model": st.session_state.ollama_model,
                        "ollama_url": st.session_state.ollama_url,
                        "mcp_url": st.session_state.mcp_url,
                        "mcp_token": st.session_state.mcp_token,
                        "system_prompt": st.session_state.system_prompt
                    }
                    save_settings_to_file(settings_to_save)
                    
                    # Use the update method to ensure proper agent recreation
                    st.session_state.app.update_system_prompt(system_prompt_draft.strip())
                    st.success("✅ System prompt updated and saved!")
                    # Clear messages to start fresh with new prompt
                    if clear_history:
                        st.session_state.messages = []
                        st.info("💬 Chat history cleared.")
                    st.rerun()
                else:
                    st.info("System prompt unchanged.")
        
        with col2:
            # Reset to default button
            default_prompt = """You are a helpful AI assistant powered by Pydantic AI and running on Ollama.
You provide accurate, helpful, and engaging responses to user questions.
Always be polite, informative, and concise in your responses.
If you're uncertain about something, acknowledge the uncertainty.

You have access to external tools through MCP (Model Context Protocol) when available.
When using tools, always explain what you're doing and what the tool returned.

When responding to complex questions, feel free to show your thinking process by using phrases like:
- "Let me think about this..."
- "I need to consider..."
- "First, I should..."
- "Let me analyze..."

This helps users understand your reasoning process."""
            
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.system_prompt = default_prompt
                
                # Save settings to file for persistence
                settings_to_save = {
                    "ollama_model": st.session_state.ollama_model,
                    "ollama_url": st.session_state.ollama_url,
                    "mcp_url": st.session_state.mcp_url,
                    "mcp_token": st.session_state.mcp_token,
                    "system_prompt": st.session_state.system_prompt
                }
                save_settings_to_file(settings_to_save)
                
                st.session_state.app.update_system_prompt(default_prompt)
                st.session_state.messages = []  # Clear history when resetting
                st.rerun()
        
        st.divider()
        
        # MCP Server Status
        st.subheader("🔗 MCP Integration")
        if hasattr(st.session_state.app, 'mcp_status'):
            if "✅" in st.session_state.app.mcp_status:
                st.success(st.session_state.app.mcp_status)
            elif "❌" in st.session_state.app.mcp_status:
                st.error(st.session_state.app.mcp_status)
            else:
                st.warning(st.session_state.app.mcp_status)
        
        st.divider()
        
        # Model info
        settings_file_exists = os.path.exists("orion_settings.json")
        settings_info = ""
        if settings_file_exists:
            try:
                import time
                mtime = os.path.getmtime("orion_settings.json")
                settings_info = f"\n- **Settings**: Saved {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}"
            except:
                settings_info = "\n- **Settings**: File exists"
        else:
            settings_info = "\n- **Settings**: Using defaults (not saved yet)"
            
        st.markdown(f"""
        **Current Configuration:**
        - **Model**: {st.session_state.ollama_model} via Ollama
        - **Framework**: Pydantic AI
        - **Ollama URL**: {st.session_state.ollama_url}/v1
        - **MCP Server**: {'Connected' if st.session_state.app.mcp_server else 'Not configured'}
        - **Output Type**: Structured (AgentResponse)
        - **System Prompt**: {len(st.session_state.system_prompt)} characters{settings_info}
        """)
        
        st.divider()
        
        # Controls
        st.header("🎛️ Controls")
        
        if st.button("🔄 Check Connection", use_container_width=True):
            st.session_state.connection_checked = False
            # Also refresh available models when checking connection
            st.session_state.available_models = get_available_ollama_models(st.session_state.ollama_url)
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Statistics
        if st.session_state.messages:
            st.divider()
            st.header("📊 Statistics")
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            st.metric("Total Messages", total_messages)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("User", user_messages)
            with col2:
                st.metric("Assistant", assistant_messages)
    
    # Main chat interface
    if not is_connected:
        st.warning("⚠️ Cannot connect to Ollama. Please check the sidebar for setup instructions.")
        st.stop()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "thinking" in message:
                    # Show thinking UI for assistant messages
                    thinking_content = message.get("thinking", "")
                    
                    # Thinking section
                    thinking_container = st.container()
                    with thinking_container:
                        thinking_expander = st.expander("💭 View thinking process", expanded=False)
                        if thinking_content:
                            thinking_expander.markdown(thinking_content)
                        else:
                            thinking_expander.markdown("*No explicit thinking process captured*")
                    
                    # Response section
                    st.markdown(message["content"])
                else:
                    # Regular message display
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Get assistant response with thinking model UI
        with chat_container:
            with st.chat_message("assistant"):
                # Convert session messages to ChatMessage objects
                conversation_history = [
                    ChatMessage(role=msg["role"], content=msg["content"])
                    for msg in st.session_state.messages[:-1]  # Exclude current message
                ]
                
                # Stream response from agent with thinking UI
                try:
                    # Create containers for thinking and response
                    thinking_container = st.container()
                    response_container = st.container()
                    
                    thinking_content = ""
                    response_content = ""
                    thinking_active = True
                    
                    # Spinner characters for animated effect
                    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    spinner_index = 0
                    
                    # Create containers for thinking and response
                    with thinking_container:
                        thinking_expander_container = st.empty()
                    
                    with response_container:
                        response_placeholder = st.empty()
                    
                    # Current expander state tracking
                    current_expander_title = f"{spinner_chars[0]} Thinking..."
                    thinking_placeholder = None
                    update_frequency = 0  # Update title every few chunks to reduce flicker
                    
                    # Get streaming response with thinking separation
                    async def stream_response():
                        nonlocal thinking_content, response_content, thinking_active, thinking_expander_container, spinner_index, current_expander_title, thinking_placeholder, update_frequency
                        
                        async for chunk_data in st.session_state.app.get_streaming_response(prompt, conversation_history):
                            chunk_type = chunk_data.get('type', 'response')
                            chunk_content = chunk_data.get('content', '')
                            chunk_status = chunk_data.get('status', 'streaming')
                            
                            if chunk_type == 'thinking':
                                thinking_content += chunk_content
                                # Update spinner animation in expander title, but not too frequently
                                update_frequency += 1
                                
                                if update_frequency % 5 == 0:  # Update title every 5th chunk to reduce flicker
                                    spinner_index = (spinner_index + 1) % len(spinner_chars)
                                    current_spinner = spinner_chars[spinner_index]
                                    current_expander_title = f"{current_spinner} Thinking..."
                                    
                                    # Update expander with new title and content
                                    with thinking_expander_container.container():
                                        thinking_expander = st.expander(current_expander_title, expanded=True)
                                        thinking_placeholder = thinking_expander.empty()
                                        thinking_placeholder.markdown(thinking_content + "▌")
                                elif thinking_placeholder:
                                    # Just update content without changing title
                                    thinking_placeholder.markdown(thinking_content + "▌")
                                elif thinking_content.strip():
                                    # Initialize thinking expander if it doesn't exist yet
                                    with thinking_expander_container.container():
                                        thinking_expander = st.expander(current_expander_title, expanded=True)
                                        thinking_placeholder = thinking_expander.empty()
                                        thinking_placeholder.markdown(thinking_content + "▌")
                                
                            elif chunk_type == 'thinking_complete':
                                thinking_active = False
                                current_expander_title = "💭 View thinking process"
                                
                                # Update to completion title and collapse automatically
                                with thinking_expander_container.container():
                                    thinking_expander = st.expander(current_expander_title, expanded=False)
                                    thinking_placeholder = thinking_expander.empty()
                                    if thinking_content.strip():
                                        thinking_placeholder.markdown(thinking_content)
                                    else:
                                        thinking_placeholder.markdown("*The AI processed your request directly without explicit thinking steps*")
                                
                            elif chunk_type == 'response':
                                # Ensure thinking is properly closed before starting response
                                if thinking_active:
                                    thinking_active = False
                                    current_expander_title = "💭 View thinking process"
                                    
                                    with thinking_expander_container.container():
                                        thinking_expander = st.expander(current_expander_title, expanded=False)
                                        thinking_placeholder = thinking_expander.empty()
                                        if thinking_content.strip():
                                            thinking_placeholder.markdown(thinking_content)
                                        else:
                                            thinking_placeholder.markdown("*Processing complete*")
                                
                                response_content += chunk_content
                                response_placeholder.markdown(response_content + "▌")
                                
                            elif chunk_type == 'response_complete':
                                # Remove the cursor from final response
                                if response_content:
                                    response_placeholder.markdown(response_content)
                                else:
                                    response_placeholder.markdown("*No response generated*")
                                
                            elif chunk_type == 'error':
                                response_placeholder.error(chunk_content)
                                return chunk_content
                        
                        return response_content
                    
                    # Run the streaming function
                    final_response = asyncio.run(stream_response())
                    
                    # Fallback to non-streaming if no response received or if response seems incomplete
                    if not final_response or not final_response.strip() or len(final_response.strip()) < 10:
                        st.warning("⚠️ Streaming response was incomplete, using fallback method...")
                        fallback_response = asyncio.run(
                            st.session_state.app.get_agent_response(prompt, conversation_history)
                        )
                        
                        # Update expander title and content for fallback method
                        with thinking_expander_container.container():
                            thinking_expander = st.expander("⚡ Used fallback method", expanded=False)
                            thinking_placeholder = thinking_expander.empty()
                            thinking_placeholder.markdown("*Streaming was incomplete, used non-streaming fallback*")
                        
                        response_placeholder.markdown(fallback_response)
                        final_response = fallback_response
                    
                    # Store both thinking and response in message history
                    if final_response.strip():
                        message_data = {
                            "role": "assistant", 
                            "content": final_response,
                            "thinking": thinking_content if thinking_content.strip() else None
                        }
                        st.session_state.messages.append(message_data)
                        
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })

    # Auto-focus the chat input after all UI elements are rendered
    auto_focus_chat_input()


if __name__ == "__main__":
    main()
