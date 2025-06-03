#                  O√S                   
#
#  .d88b.  888d888 888  .d88b.  88888b.  
# d88""88b 888P"   888 d88""88b 888 "88b 
# 888  888 888     888 888  888 888  888 
# Y88..88P 888     888 Y88..88P 888  888 
#  "Y88P"  888     888  "Y88P"  888  888

# Orion - Hybrid Reasoning
## Pydantic AI Web UI Agent App

A modern web-based chat assistant built with **Pydantic AI**, **Ollama**, and **Streamlit**. This application provides a clean, user-friendly interface to interact with AI models running locally via Ollama.

## ✨ Features

- 🦙 **Local AI Model**: Uses Ollama with qwen3:14b model
- 🔧 **Pydantic AI Framework**: Type-safe agent interactions
- 🌊 **Web Interface**: Beautiful Streamlit-based chat UI
- 💬 **Multi-turn Conversations**: Maintains conversation history
- 📊 **Structured Outputs**: Uses Pydantic models for responses
- 🔍 **Connection Checking**: Automatic Ollama server validation
- 🎨 **Modern UI**: Clean, responsive design with sidebar controls
- 📈 **Real-time Statistics**: Track conversation metrics

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running
3. **qwen3:14b** model downloaded

### Installation

1. **Clone/Navigate to the project directory**
   ```bash
   cd orion
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**
   ```bash
   # Install Ollama (if not already installed)
   # Visit: https://ollama.com/download
   
   # Start Ollama server
   ollama serve
   
   # Pull the qwen3:14b model
   ollama pull qwen3:14b
   ```

### Running the Application

```bash
streamlit run app.py
```

The web interface will be available at: **http://localhost:8501**

## 🔧 Configuration

### Model Configuration
You can modify the model settings in `app.py`:

```python
self.ollama_model = OpenAIModel(
    model_name='qwen3:14b',  # Change model here
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
```

### System Prompt
Customize the agent's behavior by modifying the system prompt:

```python
system_prompt="""Your custom system prompt here..."""
```

### Web Interface Settings
Adjust Streamlit settings in the page config:

```python
st.set_page_config(
    page_title="🤖 Pydantic AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

You can also customize the app by modifying the Streamlit configuration in `.streamlit/config.toml`.

## 📁 Project Structure

```
orion/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Dependencies

- **pydantic>=2.0.0**: Data validation and settings management
- **pydantic-ai**: Agent framework
- **streamlit>=1.28.0**: Web UI framework
- **httpx**: HTTP client for connection checking

## 🔍 Troubleshooting

### Ollama Connection Issues

1. **Server not running**
   ```bash
   ollama serve
   ```

2. **Model not found**
   ```bash
   ollama pull qwen3:14b
   ```

3. **Check available models**
   ```bash
   ollama list
   ```

### Common Errors

- **Port 8501 in use**: Use `streamlit run app.py --server.port XXXX` to specify a different port
- **Python version**: Ensure you're using Python 3.10+
- **Dependencies**: Run `pip install -r requirements.txt` again
- **Session state issues**: Clear browser cache or use incognito mode

## 🎯 Usage Examples

### Basic Chat
Simply type your message in the chat input at the bottom and press Enter.

### Conversation History
The app maintains conversation context across multiple messages and displays them in a clean chat interface.

### Sidebar Features
- **Connection Status**: Real-time Ollama connection monitoring
- **Clear Chat**: Start a fresh conversation
- **Statistics**: View message counts and conversation metrics
- **System Info**: Check model configuration and settings

## 🔮 Advanced Features

### Structured Responses
The agent uses Pydantic models for structured outputs:

```python
class AgentResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
```

### Conversation Tracking
Messages are stored as structured objects:

```python
class ChatMessage(BaseModel):
    role: str
    content: str
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Pydantic AI](https://ai.pydantic.dev/) - The agent framework
- [Ollama](https://ollama.com/) - Local AI model server
- [Streamlit](https://streamlit.io/) - Web UI framework
- [Qwen](https://qwenlm.github.io/) - The AI model

---

## 📝 TODO

- [ ] Re-do UI
- [ ] Add dynamic reasoning using qwen3
- [ ] Ollama fetch and pick models in sidebar, rather than pre-defined
- [ ] Add other model providers like OpenRouter
- [ ] Add MCP tools via n8n
- [ ] Add user authentication and session management
- [ ] Implement conversation persistence (database storage)
- [ ] Add export functionality (PDF, markdown, JSON)
- [ ] Create comprehensive test suite
- [ ] Add conversation search and filtering
- [ ] Implement rate limiting and usage quotas
- [ ] Add dark/light theme toggle
- [ ] Create mobile-responsive design
- [ ] Add file upload and document processing capabilities
- [ ] Implement conversation templates and prompts library
- [ ] Add real-time typing indicators
- [ ] Create REST API endpoints
- [ ] Add conversation sharing and collaboration features
- [ ] Implement plugin/extension system
- [ ] Add performance monitoring and analytics
- [ ] Create Docker containerization
- [ ] Add environment-specific configuration management
- [ ] Implement backup and restore functionality
- [ ] Add multi-language support (i18n) 