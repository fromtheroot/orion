import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import streamlit as st
import httpx

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class ChatMessage(BaseModel):
    """Structure for chat messages"""
    role: str = Field(description="Role of the message sender")
    content: str = Field(description="Content of the message")


class AgentResponse(BaseModel):
    """Structure for agent responses"""
    response: str = Field(description="The agent's response")


class PydanticAIWebApp:
    """Main application class for the Pydantic AI Web UI"""
    
    def __init__(self, system_prompt: Optional[str] = None):
        # Initialize Ollama model with OpenAI compatibility
        self.ollama_model = OpenAIModel(
            model_name='qwen3:14b',
            provider=OpenAIProvider(base_url='http://localhost:11434/v1')
        )
        
        # Default system prompt if none provided
        default_system_prompt = """You are a helpful AI assistant powered by Pydantic AI and running on Ollama.
            You provide accurate, helpful, and engaging responses to user questions.
            Always be polite, informative, and concise in your responses.
            If you're uncertain about something, acknowledge the uncertainty."""
        
        # Create the agent with custom or default system prompt
        self.agent = Agent(
            model=self.ollama_model,
            system_prompt=system_prompt or default_system_prompt
        )
        
        # Store the system prompt for reference
        self.system_prompt = system_prompt or default_system_prompt
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt and recreate the agent"""
        self.system_prompt = new_system_prompt
        self.agent = Agent(
            model=self.ollama_model,
            system_prompt=new_system_prompt
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
            # Get response from agent - no message_history for simpler approach
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
            # Use stream for streaming responses - no message_history for now
            async with self.agent.run_stream(user_prompt=message) as result:
                previous_content = ""
                thinking_content = ""
                response_content = ""
                in_thinking_mode = True
                thinking_complete = False
                
                async for chunk in result.stream():
                    # Get the current content
                    if hasattr(chunk, 'response'):
                        current_content = chunk.response
                    else:
                        current_content = str(chunk)
                    
                    # Only process new content
                    if current_content and len(current_content) > len(previous_content):
                        new_chunk = current_content[len(previous_content):]
                        previous_content = current_content
                        
                        if new_chunk.strip():
                            # Check for explicit thinking tags first
                            chunk_lower = new_chunk.lower()
                            
                            # Track if we have explicit thinking tags in the content
                            has_thinking_start = any(tag in (thinking_content + new_chunk).lower() for tag in ['<think>', '<thinking>'])
                            has_thinking_end = any(tag in chunk_lower for tag in ['</think>', '</thinking>'])
                            
                            if in_thinking_mode:
                                # Check if we've hit the end of thinking first
                                if has_thinking_end:
                                    # Found explicit end tag - need to split the content
                                    end_tag_patterns = ['</think>', '</thinking>']
                                    thinking_part = new_chunk
                                    response_part = ""
                                    
                                    # Find where the thinking ends
                                    for pattern in end_tag_patterns:
                                        if pattern in chunk_lower:
                                            split_index = chunk_lower.find(pattern) + len(pattern)
                                            thinking_part = new_chunk[:split_index]
                                            response_part = new_chunk[split_index:]
                                            break
                                    
                                    # Add thinking part
                                    thinking_content += thinking_part
                                    yield {
                                        'type': 'thinking',
                                        'content': thinking_part,
                                        'status': 'streaming'
                                    }
                                    
                                    # Switch to response mode
                                    in_thinking_mode = False
                                    thinking_complete = True
                                    yield {
                                        'type': 'thinking_complete',
                                        'content': '',
                                        'status': 'complete'
                                    }
                                    
                                    # Add response part if any
                                    if response_part.strip():
                                        response_content += response_part
                                        yield {
                                            'type': 'response',
                                            'content': response_part,
                                            'status': 'streaming'
                                        }
                                else:
                                    # Add content to thinking
                                    thinking_content += new_chunk
                                    
                                if has_thinking_start and not thinking_complete and not has_thinking_end:
                                    # We're definitely in thinking mode with explicit tags
                                    yield {
                                        'type': 'thinking',
                                        'content': new_chunk,
                                        'status': 'streaming'
                                    }
                                elif not has_thinking_end and not has_thinking_start:
                                    # Use heuristic detection only if no explicit tags found
                                    thinking_patterns = [
                                        "let me think", "i need to", "i should", "analyzing", "considering",
                                        "hmm", "well", "so", "first", "next", "then", "however", "but",
                                        "the user", "they want", "they're asking", "this question"
                                    ]
                                    
                                    response_patterns = [
                                        "hello!", "hi there", "sure!", "certainly", "of course",
                                        "i can help", "here's", "i'll help", "absolutely",
                                        "great question", "i'd be happy", "let me explain",
                                        "here are", "to answer", "based on"
                                    ]
                                    
                                    # Check if we're transitioning to response mode
                                    starts_with_response = any(
                                        chunk_lower.strip().startswith(pattern.split()[0]) and pattern in chunk_lower 
                                        for pattern in response_patterns
                                    )
                                    
                                    has_response_pattern = any(pattern in chunk_lower for pattern in response_patterns)
                                    
                                    # Length-based heuristic for formal responses
                                    is_formal_response = (
                                        len(thinking_content) > 100 and 
                                        (chunk_lower.strip().startswith(('i', 'to', 'the', 'this', 'that', 'based', 'according')) or
                                         '.' in new_chunk and len(new_chunk.split('.')) > 1)
                                    )
                                    
                                    if starts_with_response or (has_response_pattern and len(thinking_content) > 50) or is_formal_response:
                                        in_thinking_mode = False
                                        thinking_complete = True
                                        yield {
                                            'type': 'thinking_complete',
                                            'content': '',
                                            'status': 'complete'
                                        }
                                        # This chunk becomes part of response
                                        response_content += new_chunk
                                        yield {
                                            'type': 'response',
                                            'content': new_chunk,
                                            'status': 'streaming'
                                        }
                                    else:
                                        # Still thinking
                                        yield {
                                            'type': 'thinking',
                                            'content': new_chunk,
                                            'status': 'streaming'
                                        }
                            else:
                                # We're in response mode
                                response_content += new_chunk
                                yield {
                                    'type': 'response',
                                    'content': new_chunk,
                                    'status': 'streaming'
                                }
                
                # Mark completion
                if not thinking_complete:
                    yield {
                        'type': 'thinking_complete',
                        'content': '',
                        'status': 'complete'
                    }
                    
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


def check_ollama_connection():
    """Check if Ollama server is running and model is available"""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            qwen_models = [m for m in models if "qwen3" in m.get("name", "").lower()]
            
            if qwen_models:
                return True, "✅ Ollama server is running and qwen3 model is available!"
            else:
                return False, "⚠️ Ollama server is running but qwen3:14b model not found. Run: `ollama pull qwen3:14b`"
        else:
            return False, "❌ Ollama server is not responding properly"
            
    except Exception as e:
        return False, f"❌ Cannot connect to Ollama server: {e}. Make sure Ollama is running: `ollama serve`"


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        # Default system prompt
        st.session_state.system_prompt = """You are a helpful AI assistant powered by Pydantic AI and running on Ollama.
You provide accurate, helpful, and engaging responses to user questions.
Always be polite, informative, and concise in your responses.
If you're uncertain about something, acknowledge the uncertainty."""
    if "app" not in st.session_state:
        st.session_state.app = PydanticAIWebApp(st.session_state.system_prompt)
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
        initial_sidebar_state="expanded"
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
    st.markdown("""
    **Powered by [Synthetic](https://syntheticlabs.xyz)**
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Information")
        
        # Connection status
        if not st.session_state.connection_checked:
            with st.spinner("Checking Ollama connection..."):
                is_connected, status_message = check_ollama_connection()
                st.session_state.connection_status = (is_connected, status_message)
                st.session_state.connection_checked = True
        
        is_connected, status_message = st.session_state.connection_status
        
        if is_connected:
            st.success(status_message)
        else:
            st.error(status_message)
            st.markdown("""
            **Setup Steps:**
            1. Install Ollama: [ollama.com](https://ollama.com)
            2. Start server: `ollama serve`
            3. Pull model: `ollama pull qwen3:14b`
            """)
        
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
                    # Use the update method to ensure proper agent recreation
                    st.session_state.app.update_system_prompt(system_prompt_draft.strip())
                    st.success("✅ System prompt updated and applied!")
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
If you're uncertain about something, acknowledge the uncertainty."""
            
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.system_prompt = default_prompt
                st.session_state.app.update_system_prompt(default_prompt)
                st.session_state.messages = []  # Clear history when resetting
                st.rerun()
        
        st.divider()
        
        # Model info
        st.markdown(f"""
        **Configuration:**
        - **Model**: qwen3:14b via Ollama
        - **Framework**: Pydantic AI
        - **Base URL**: http://localhost:11434/v1
        - **Output Type**: Structured (AgentResponse)
        - **System Prompt**: {len(st.session_state.system_prompt)} characters
        """)
        
        st.divider()
        
        # Controls
        st.header("🎛️ Controls")
        
        if st.button("🔄 Check Connection", use_container_width=True):
            st.session_state.connection_checked = False
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
                                
                                if update_frequency % 3 == 0:  # Update title every 3rd chunk
                                    spinner_index = (spinner_index + 1) % len(spinner_chars)
                                    current_spinner = spinner_chars[spinner_index]
                                    current_expander_title = f"{current_spinner} Thinking..."
                                    
                                    # Update expander with new title
                                    with thinking_expander_container.container():
                                        thinking_expander = st.expander(current_expander_title, expanded=True)
                                        thinking_placeholder = thinking_expander.empty()
                                        thinking_placeholder.markdown(thinking_content + "▌")
                                else:
                                    # Just update content without changing title
                                    if thinking_placeholder:
                                        thinking_placeholder.markdown(thinking_content + "▌")
                                
                            elif chunk_type == 'thinking_complete':
                                thinking_active = False
                                current_expander_title = "✅ Thinking complete - View process"
                                
                                # Update to completion title
                                with thinking_expander_container.container():
                                    thinking_expander = st.expander(current_expander_title, expanded=True)
                                    thinking_placeholder = thinking_expander.empty()
                                    if thinking_content:
                                        thinking_placeholder.markdown(thinking_content)
                                    else:
                                        thinking_placeholder.markdown("*No explicit thinking process captured*")
                                
                            elif chunk_type == 'response':
                                if thinking_active:
                                    # If we get response content while still thinking, mark thinking as complete
                                    thinking_active = False
                                    current_expander_title = "✅ Thinking complete - View process"
                                    
                                    with thinking_expander_container.container():
                                        thinking_expander = st.expander(current_expander_title, expanded=True)
                                        thinking_placeholder = thinking_expander.empty()
                                        if thinking_content:
                                            thinking_placeholder.markdown(thinking_content)
                                        else:
                                            thinking_placeholder.markdown("*Processing complete*")
                                
                                response_content += chunk_content
                                response_placeholder.markdown(response_content + "▌")
                                
                            elif chunk_type == 'response_complete':
                                if response_content:
                                    response_placeholder.markdown(response_content)
                                
                            elif chunk_type == 'error':
                                response_placeholder.error(chunk_content)
                                return chunk_content
                        
                        return response_content
                    
                    # Run the streaming function
                    final_response = asyncio.run(stream_response())
                    
                    # Fallback to non-streaming if no response received
                    if not final_response.strip():
                        fallback_response = asyncio.run(
                            st.session_state.app.get_agent_response(prompt, conversation_history)
                        )
                        
                        # Update expander title and content for fallback method
                        with thinking_expander_container.container():
                            thinking_expander = st.expander("⚡ Used fallback method", expanded=True)
                            thinking_placeholder = thinking_expander.empty()
                            thinking_placeholder.markdown("*Used fallback response method*")
                        
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


if __name__ == "__main__":
    main()
