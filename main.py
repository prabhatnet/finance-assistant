"""Application entry point - Initializes and runs the AI Finance Assistant."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.config import get_settings
from src.core.llm import create_llm
from src.rag.embeddings import create_embeddings
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import RAGRetriever
from src.data.market_data import MarketDataProvider
from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.tax_agent import TaxEducationAgent
from src.workflow.router import QueryRouter
from src.workflow.nodes import initialize_nodes
from src.utils.logger import setup_logging, get_logger


def initialize_app() -> None:
    """Initialize all application components.

    Sets up logging, LLM, RAG pipeline, agents, and workflow routing.
    """
    settings = get_settings()
    setup_logging(level=settings.log_level)
    logger = get_logger(__name__)
    logger.info("Initializing AI Finance Assistant v%s", settings.app_version)

    # Create LLM instance
    llm = create_llm(settings)

    # Initialize RAG pipeline
    embeddings = create_embeddings(settings)
    vector_store = VectorStoreManager(embeddings, settings)
    vector_store.initialize()
    retriever = RAGRetriever(vector_store, settings)

    # Initialize market data provider
    market_data = MarketDataProvider(settings)

    # Create specialized agents
    agents = {
        "finance_qa": FinanceQAAgent(llm=llm, retriever=retriever),
        "portfolio": PortfolioAnalysisAgent(llm=llm),
        "market": MarketAnalysisAgent(llm=llm, market_data_provider=market_data),
        "goal_planning": GoalPlanningAgent(llm=llm),
        "news": NewsSynthesizerAgent(llm=llm, market_data_provider=market_data),
        "tax": TaxEducationAgent(llm=llm, retriever=retriever),
    }

    # Create router and initialize workflow nodes
    router = QueryRouter(llm=llm)
    initialize_nodes(agents=agents, router=router)

    logger.info("Application initialized successfully with %d agents", len(agents))


if __name__ == "__main__":
    initialize_app()
