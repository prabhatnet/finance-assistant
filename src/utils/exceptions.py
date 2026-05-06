"""Custom exception hierarchy for the AI Finance Assistant."""


class FinanceAssistantError(Exception):
    """Base exception for all finance assistant errors."""

    def __init__(self, message: str = "An error occurred in the finance assistant") -> None:
        self.message = message
        super().__init__(self.message)


class LLMError(FinanceAssistantError):
    """Raised when LLM invocation fails."""

    def __init__(self, message: str = "LLM invocation failed", provider: str = "") -> None:
        self.provider = provider
        super().__init__(f"{message} (provider: {provider})" if provider else message)


class RAGError(FinanceAssistantError):
    """Raised when RAG retrieval or indexing fails."""

    def __init__(self, message: str = "RAG operation failed") -> None:
        super().__init__(message)


class MarketDataError(FinanceAssistantError):
    """Raised when market data API calls fail."""

    def __init__(self, message: str = "Market data fetch failed", symbol: str = "") -> None:
        self.symbol = symbol
        super().__init__(f"{message} (symbol: {symbol})" if symbol else message)


class ConfigurationError(FinanceAssistantError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str = "Configuration error") -> None:
        super().__init__(message)


class WorkflowError(FinanceAssistantError):
    """Raised when the LangGraph workflow encounters an error."""

    def __init__(self, message: str = "Workflow execution failed", node: str = "") -> None:
        self.node = node
        super().__init__(f"{message} (node: {node})" if node else message)
