# 🤖 LangGraph Chatbot Project

A sophisticated chatbot built with LangGraph that features tool integration, conversation memory, and a Streamlit web interface.

## 📋 Features

### 🛠️ Core Functionality
- **Tool Integration**: Built-in tools for time queries and mathematical calculations
- **Conversation Memory**: Thread-based conversation persistence using SQLite
- **Conditional Routing**: Smart routing between chat and tool execution
- **Streamlit Interface**: Beautiful web UI for chatting

### 🧰 Available Tools
- **`get_current_time`**: Returns current date and time
- **`calculate`**: Performs mathematical operations (add, subtract, multiply, divide)

### 🏗️ Architecture
- **LangGraph**: State management and workflow orchestration
- **Groq LLM**: Fast, efficient language model integration
- **SQLite**: Persistent conversation storage
- **Streamlit**: Modern web interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API Key
- Optional: LangSmith API Key (for tracing)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd LangGraph
```

2. **Create virtual environment**
```bash
python -m venv ENV
.\ENV\Scripts\activate  # On Windows
source ENV/bin/activate  # On Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=your_project_name
LANGSMITH_TRACING=true
```

### Running the Applications

#### 🖥️ Command Line Interface
```bash
python chatbot\bot1.py
```

#### 🌐 Web Interface
```bash
streamlit run chatbot\streamlit.py
```

## 📁 Project Structure

```
LangGraph/
├── chatbot/
│   ├── bot1.py              # Main chatbot with tools and SQLite persistence
│   ├── chatbot_tools.py     # Tool definitions (time, calculator)
│   ├── streamlit.py         # Streamlit web interface
│   └── bot.py              # Simple joke bot (example)
├── graphs/
│   ├── basic_graph.py       # Basic LangGraph example
│   ├── conditional_graph.py # Conditional routing example
│   ├── essay_evaluation.py # Essay evaluation with structured output
│   ├── iterative_workflows.py # Iterative workflow examples
│   └── parallel_workflows.py  # Parallel processing examples
├── .env                     # Environment variables
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for Groq LLM access
- `LANGSMITH_API_KEY`: Optional, for LangSmith tracing
- `LANGSMITH_PROJECT`: Optional, LangSmith project name
- `LANGSMITH_TRACING`: Enable/disable LangSmith tracing

### Database
- Conversations are automatically stored in `chatbot.db` (SQLite)
- Each conversation thread is uniquely identified
- Data persists across application restarts

## 🛠️ Development

### Adding New Tools

1. **Create a new tool in `chatbot_tools.py`:**
```python
@tool
def your_tool(param1: type1, param2: type2):
    """Tool description"""
    # Your tool logic here
    return result
```

2. **Add to tools list:**
```python
tools = [get_current_time, calculate, your_tool]
```

### Customizing the Chatbot

1. **Modify the system message** in `bot1.py` to change bot behavior
2. **Update the graph structure** to add new nodes or routing logic
3. **Extend the state schema** to store additional conversation data

## 🎯 Usage Examples

### Basic Chat
```
You: What time is it?
Bot: The current time is 2024-04-29 15:30:45

You: Calculate 25 * 4
Bot: 25 * 4 = 100
```

### Mathematical Operations
```
You: What's 100 divided by 5?
Bot: 100 / 5 = 20.0

You: Add 50 and 75
Bot: 50 + 75 = 125
```

### Web Interface Features
- **Chat History**: View and switch between conversation threads
- **New Chat**: Start fresh conversations
- **Persistent Storage**: Conversations saved automatically
- **Real-time Responses**: Streaming bot responses

## 🔍 Troubleshooting

### Common Issues

1. **"No module named 'langgraph'"**
   - Solution: `pip install langgraph`

2. **"Could not import ddgs package"**
   - Solution: `pip install -U ddgs`

3. **"tool call validation failed"**
   - Solution: Ensure all tools in `chatbot_tools.py` are properly defined

4. **SQLite database errors**
   - Solution: Delete `chatbot.db` and restart the application

5. **Streamlit not found**
   - Solution: `pip install streamlit`

### Debug Mode
Enable detailed logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 Learn More

### LangGraph Documentation
- [Official LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [State Management](https://python.langchain.com/docs/langgraph/concepts/state)
- [Tool Usage](https://python.langchain.com/docs/langgraph/concepts/tools)

### Related Technologies
- [LangChain](https://python.langchain.com/) - LLM framework
- [Groq](https://groq.com/) - Fast inference platform
- [Streamlit](https://streamlit.io/) - ML app framework

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain team for the excellent LangGraph framework
- Groq for providing fast LLM inference
- Streamlit community for the amazing UI framework

---

**Happy Chatting! 🚀**
