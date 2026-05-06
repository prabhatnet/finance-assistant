# Knowledge Base

This directory contains the financial education articles and documents used by the RAG system.

## Directory Structure

```
knowledge_base/
├── investing_basics/       # Fundamental investing concepts
├── portfolio_management/   # Portfolio construction and management
├── market_analysis/        # Market dynamics and analysis
├── retirement_planning/    # Retirement accounts and strategies
├── tax_education/          # Tax concepts for investors
└── financial_planning/     # General financial planning topics
```

## Adding New Articles

1. Create a `.md` or `.txt` file in the appropriate category subdirectory
2. Include a clear title as the first heading
3. Write in clear, jargon-free language suitable for beginners
4. Include source citations where applicable
5. Re-run the indexer to update the vector store:

```python
from src.rag.indexer import KnowledgeBaseIndexer
indexer = KnowledgeBaseIndexer()
indexer.index()
```

## Article Guidelines

- Target length: 500-2000 words per article
- Use headers to organize content
- Include practical examples and analogies
- Define technical terms when first introduced
- Add metadata tags in the filename (e.g., `basics_compound_interest.md`)
