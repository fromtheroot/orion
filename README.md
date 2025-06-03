# Orion - Pydantic AI Chat Assistant with MCP Integration

A powerful Streamlit-based chat interface powered by Pydantic AI and Ollama, with support for Model Context Protocol (MCP) tools.

## Features

- 🤖 **Local AI Assistant**: Runs on Ollama with qwen3:14b model
- 🔗 **MCP Integration**: Connect to external tools and services via Model Context Protocol
- 💭 **Thinking Process**: Visualize the AI's thinking process during responses
- 🎯 **Customizable System Prompts**: Easily modify the AI's behavior
- 📊 **Real-time Status**: Monitor Ollama and MCP server connections
- 🎨 **Modern UI**: Beautiful Streamlit interface with responsive design

## Prerequisites

1. **Ollama**: Install and run Ollama with the qwen3:14b model
   ```bash
   # Install Ollama from https://ollama.com
   ollama serve
   ollama pull qwen3:14b
   ```

2. **Python 3.10+**: Required for MCP support

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd orion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional for MCP):
   ```bash
   cp .env.example .env
   # Edit .env with your MCP server details
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# MCP Server Configuration (optional)
MCP_SERVER_URL=https://your-mcp-server.com/sse
MCP_BEARER_TOKEN=your_bearer_token

# Optional: LLM API Keys
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
```

### MCP Integration

The app supports HTTP Server-Sent Events (SSE) MCP servers with bearer authentication. When configured:

- Tools from your MCP server are automatically available to the AI
- The AI can call external APIs, query databases, or interact with other services
- All tool usage is transparent and shown in the chat interface

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Start chatting! The AI will automatically use available MCP tools when appropriate.

## Features Overview

### System Prompt Customization
- Modify the AI's behavior through the sidebar
- Apply changes instantly without restarting
- Option to clear chat history when changing prompts

### MCP Tool Integration
- Automatic discovery of tools from configured MCP servers
- Seamless integration with Pydantic AI's function calling
- Real-time status monitoring of MCP connections

### Thinking Process Visualization
- See how the AI processes your requests
- Collapsible thinking sections to reduce clutter
- Clear separation between thinking and final responses

### Connection Monitoring
- Real-time Ollama server status
- MCP server connection status
- Automatic retry and error handling

## Troubleshooting

### Ollama Issues
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`
- Pull model if missing: `ollama pull qwen3:14b`

### MCP Issues
- Check your MCP server URL and authentication
- Verify the server supports SSE transport
- Check network connectivity to the MCP server

### General Issues
- Check the sidebar for connection status
- View browser console for detailed error messages
- Ensure all dependencies are installed correctly

## Development

The application is built with:
- **Streamlit**: Web interface framework
- **Pydantic AI**: AI agent framework with built-in MCP support
- **Ollama**: Local LLM runtime
- **httpx**: HTTP client for API calls

Key files:
- `app.py`: Main application logic
- `requirements.txt`: Python dependencies
- `.env`: Environment configuration
- `.gitignore`: Protects sensitive files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your License Here]

---

**Note**: This application is designed to work with your specific MCP server setup. Adjust the configuration as needed for your environment. 