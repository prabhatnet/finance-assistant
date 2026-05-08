---
title: AI Finance Assistant
emoji: 💰
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 💰 AI Finance Assistant

**Production-grade multi-agent AI finance assistant built with LangGraph, RAG, and real-time market APIs.**

Democratizing financial literacy through intelligent conversational AI — helping beginners take their first steps toward financial security with personalized, accessible education and guidance.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🏗️ Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Web App                         │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │   Chat   │  │    Portfolio     │  │      Market          │  │
│  │   Page   │  │    Dashboard     │  │      Overview        │  │
│  └────┬─────┘  └────────┬─────────┘  └──────────┬───────────┘  │
└───────┼─────────────────┼────────────────────────┼──────────────┘
        │                 │                        │
        └─────────────────┼────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    LangGraph Workflow Engine                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Query Router (LLM)                      │   │
│  └───┬──────┬──────┬──────┬──────┬──────┬───────────────────┘   │
│      │      │      │      │      │      │                        │
│  ┌───▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐                  │
│  │Fin QA││Port.││Mkt. ││Goal ││News ││Tax  │  Specialized      │
│  │Agent ││Agent││Agent││Agent││Agent││Agent│  Agents            │
│  └───┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘                  │
└──────┼──────┼──────┼──────┼──────┼──────┼───────────────────────┘
       │      │      │      │      │      │
┌──────▼──────▼──────┼──────▼──────▼──────▼───────────────────────┐
│            RAG Pipeline            │     Market Data APIs         │
│  ┌─────────────────────────────┐   │  ┌──────────────────────┐   │
│  │  Knowledge Base (50+ docs)  │   │  │  yFinance / Alpha    │   │
│  │  FAISS Vector Store         │   │  │  Vantage API         │   │
│  │  Semantic Search            │   │  │  + TTL Cache         │   │
│  └─────────────────────────────┘   │  └──────────────────────┘   │
└────────────────────────────────────┴─────────────────────────────┘
```

### Data Flow

```
User Query → Workflow Router → Appropriate Agent(s) → RAG Retrieval → LLM Processing → Response → UI
```

### Core Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Multi-Agent System | LangGraph | Orchestrates specialized agents with state management |
| Language Model | OpenAI GPT / Google Gemini / Claude | Natural language understanding and generation |
| Vector Database | FAISS / ChromaDB | Semantic search over financial knowledge base |
| Market Data | yFinance / Alpha Vantage | Real-time stock quotes and financial data |
| Web Interface | Streamlit | Interactive multi-tab user experience |
| Embeddings | sentence-transformers | Local embedding generation for RAG |

---

## ✨ Features

### 6 Specialized Agents

1. **Finance Q&A Agent** — General financial education (stocks, bonds, ETFs, diversification)
2. **Portfolio Analysis Agent** — Portfolio review, diversification assessment, risk analysis
3. **Market Analysis Agent** — Real-time stock data, market trends, company information
4. **Goal Planning Agent** — SMART goal setting, savings plans, retirement projections
5. **News Synthesizer Agent** — Financial news summarization with educational context
6. **Tax Education Agent** — Tax-advantaged accounts, capital gains, tax strategies

### Key Capabilities

- **RAG-Powered Responses** — Grounded in a curated financial knowledge base with source citations
- **Real-Time Market Data** — Live stock quotes, historical charts, company information
- **Intelligent Routing** — LLM-based query classification routes to the optimal agent
- **Conversation Context** — Multi-turn conversations with history preservation
- **Portfolio Visualization** — Interactive charts showing allocation and performance
- **Proper Disclaimers** — Clear separation between education and financial advice

---

## 📁 Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/                    # Specialized financial agents
│   │   ├── base_agent.py         # Abstract base class for all agents
│   │   ├── finance_qa_agent.py   # General financial education
│   │   ├── portfolio_agent.py    # Portfolio analysis
│   │   ├── market_agent.py       # Real-time market insights
│   │   ├── goal_planning_agent.py# Financial goal planning
│   │   ├── news_agent.py         # News synthesis
│   │   └── tax_agent.py          # Tax education
│   ├── core/                      # Core infrastructure
│   │   ├── config.py             # Configuration management (Pydantic + YAML)
│   │   ├── llm.py                # LLM factory (OpenAI/Google/Anthropic)
│   │   ├── state.py              # LangGraph state schema
│   │   └── prompts.py            # System prompts and templates
│   ├── data/                      # Data layer
│   │   ├── market_data.py        # Market data provider (yFinance/Alpha Vantage)
│   │   ├── cache.py              # TTL cache for API responses
│   │   └── knowledge_base/       # Financial education articles (RAG source)
│   ├── rag/                       # RAG pipeline
│   │   ├── embeddings.py         # Embedding model factory
│   │   ├── vector_store.py       # FAISS/Chroma vector store management
│   │   ├── retriever.py          # Document retrieval with filtering
│   │   └── indexer.py            # Knowledge base indexing pipeline
│   ├── web_app/                   # Streamlit application
│   │   ├── app.py                # Main app entry point
│   │   ├── pages/                # Multi-tab pages
│   │   │   ├── chat.py           # Conversational interface
│   │   │   ├── portfolio.py      # Portfolio dashboard
│   │   │   └── market.py         # Market overview
│   │   └── components/           # Reusable UI components
│   │       ├── sidebar.py        # Navigation sidebar
│   │       └── charts.py         # Plotly chart builders
│   ├── utils/                     # Shared utilities
│   │   ├── logger.py             # Structured logging (structlog)
│   │   ├── exceptions.py         # Custom exception hierarchy
│   │   └── validators.py         # Input validation
│   └── workflow/                  # LangGraph orchestration
│       ├── graph.py              # Workflow graph definition
│       ├── router.py             # LLM-based query router
│       └── nodes.py              # Graph node implementations
├── tests/                         # Test suite
│   ├── conftest.py               # Shared fixtures
│   ├── unit/                     # Unit tests
│   │   ├── test_agents.py
│   │   ├── test_rag.py
│   │   └── test_workflow.py
│   └── integration/              # Integration tests
│       ├── test_end_to_end.py
│       └── test_market_data.py
├── config.yaml                    # Application configuration
├── main.py                        # Application initialization
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata and tool config
├── Dockerfile                     # Container configuration
├── docker-compose.yml             # Container orchestration
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- pip or conda
- Git
- API key for at least one LLM provider (OpenAI, Google, or Anthropic)
- (Optional) Alpha Vantage API key for market data

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai_finance_assistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate

# Windows/Python
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
# At minimum, set one LLM provider key:
#   OPENAI_API_KEY=sk-...
#   or GOOGLE_API_KEY=...
#   or ANTHROPIC_API_KEY=...
```

### 5. Index the Knowledge Base

```bash
python -c "from src.rag.indexer import KnowledgeBaseIndexer; KnowledgeBaseIndexer().index()"
```

### 6. Run the Application

```bash
streamlit run src/web_app/app.py
```

The app will be available at `http://localhost:8501`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t ai-finance-assistant .
docker run -p 8501:8501 --env-file .env ai-finance-assistant
```

---

## 💡 Usage Examples

### Chat Interface

```
User: "What is dollar-cost averaging and why should I use it?"
Agent: [Finance Q&A Agent routes and responds with educational content + RAG sources]

User: "Analyze my portfolio: 50 shares AAPL, 30 shares MSFT, 100 shares VOO"
Agent: [Portfolio Agent analyzes diversification, concentration, and risk]

User: "What's the current price of Tesla?"
Agent: [Market Agent fetches live data from yFinance API]

User: "How should I save $50,000 for a house down payment in 3 years?"
Agent: [Goal Planning Agent creates a structured savings plan]

User: "What happened in the market today?"
Agent: [News Agent synthesizes recent financial news]

User: "How do Roth IRA conversions work?"
Agent: [Tax Agent explains with appropriate disclaimers]
```

### Programmatic Usage

```python
from main import initialize_app
from src.workflow.graph import create_workflow_graph

# Initialize the application
initialize_app()

# Create and invoke the workflow
graph = create_workflow_graph()
result = await graph.ainvoke({
    "query": "What is compound interest?",
    "chat_history": [],
})
print(result["response"])
```

---

## 📖 API Documentation

### Agent Interface

All agents implement the `BaseAgent` abstract class:

```python
class BaseAgent(ABC):
    async def process(self, state: AgentState) -> AgentState:
        """Process state and return updated state with response."""
        ...
```

### State Schema

```python
class AgentState(TypedDict):
    query: str                         # User's input query
    chat_history: list[dict]           # Conversation history
    route: str                         # Determined agent route
    response: str                      # Generated response
    sources: list[dict]                # RAG source citations
    portfolio_data: dict               # Portfolio holdings
    market_data: dict                  # Market quotes
    symbols: list[str]                 # Extracted ticker symbols
    error: str | None                  # Error information
```

### Configuration

Configuration is loaded from `config.yaml` with environment variable overrides:

```yaml
llm:
  provider: "openai"       # openai | google | anthropic
  model: "gpt-4o-mini"
  temperature: 0.1

vector_store:
  type: "faiss"            # faiss | chroma
  chunk_size: 1000

market_data:
  provider: "yfinance"     # yfinance | alpha_vantage
  cache_ttl_seconds: 300
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/ -m integration

# Specific agent tests
pytest tests/unit/test_agents.py -v
```

### Test Structure

- **Unit Tests**: Test individual agents, router, cache, and RAG components in isolation
- **Integration Tests**: Test end-to-end workflow execution and API interactions

---

## 🐳 Deployment

### Docker

```bash
docker-compose up --build -d
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `GOOGLE_API_KEY` | Yes* | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key |
| `ALPHA_VANTAGE_API_KEY` | No | Alpha Vantage key (defaults to yFinance) |
| `LLM_PROVIDER` | No | LLM provider to use (default: openai) |
| `LLM_MODEL` | No | Model name (default: gpt-4o-mini) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

*At least one LLM provider key is required.

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual env is activated and dependencies installed |
| `OPENAI_API_KEY not set` | Copy `.env.example` to `.env` and add your API key |
| `Vector store not initialized` | Run the knowledge base indexer (see Setup step 5) |
| `Rate limit exceeded` | Reduce request frequency or increase `CACHE_TTL_SECONDS` |
| `yfinance data unavailable` | Check network connection; some symbols may not be available |
| `Streamlit port in use` | Use `streamlit run src/web_app/app.py --server.port=8502` |

### Debug Mode

Enable debug mode for verbose logging:

```bash
# In .env
APP_DEBUG=true
LOG_LEVEL=DEBUG
```

### Resetting the Vector Store

```bash
# Delete persisted store and re-index
rm -rf data/vector_store/
python -c "from src.rag.indexer import KnowledgeBaseIndexer; KnowledgeBaseIndexer().index()"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Format
ruff format src/ tests/
```

---

## ⚠️ Disclaimer

This application is designed for **educational purposes only** and does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions. The creators of this tool are not responsible for any financial decisions made based on its output.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
