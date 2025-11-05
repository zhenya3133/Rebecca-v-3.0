# Rebecca Platform - Offline Mode Documentation

## Overview

Rebecca Platform supports **Offline Mode** for deterministic, dependency-light test execution. This mode enables:

- ✅ **Deterministic testing** without random/external factors
- ✅ **Fast CI/CD pipelines** without network dependencies
- ✅ **No external service requirements** (databases, APIs, models)
- ✅ **Reproducible results** for debugging and validation

## Activation

### Environment Variables

Set one of these environment variables:

```bash
# Primary offline mode flag
export REBECCA_OFFLINE_MODE=1

# Alternative flag (for test suites)
export REBECCA_TEST_MODE=1
```

Accepted values: `1`, `true`, `yes`, `on` (case-insensitive)

### Automatic Test Mode

Tests automatically enable offline mode via `tests/conftest.py`:

```python
@pytest.fixture(scope="session", autouse=True)
def enable_offline_mode():
    """Automatically enable offline mode for all tests."""
    os.environ["REBECCA_OFFLINE_MODE"] = "1"
    os.environ["REBECCA_TEST_MODE"] = "1"
```

### Checking Offline Mode in Code

```python
from configuration import is_offline_mode

if is_offline_mode():
    # Use offline stubs
    print("Running in offline mode")
else:
    # Use real services
    print("Running in online mode")
```

## Components Affected

### 1. Vector Store (`memory_manager/vector_store_client.py`)

**Offline Behavior:**
- Forces `MemoryVectorStore` (in-memory) instead of Qdrant/ChromaDB/Weaviate
- Skips lazy imports of external vector store packages
- Uses deterministic hash-based embeddings (SHA-256)

**Example:**
```python
from memory_manager.vector_store_client import VectorStoreClient, VectorStoreConfig

config = VectorStoreConfig(provider='qdrant')  # Will use 'memory' in offline mode
client = VectorStoreClient(config)

# Embeddings are deterministic
text = "test document"
emb1 = await client.vectorize_text(text)
emb2 = await client.vectorize_text(text)
assert emb1 == emb2  # Always true in offline mode
```

**Key Changes:**
- No network calls to OpenAI/Ollama embedding APIs
- 384-dimensional deterministic embeddings via SHA-256 hash
- All data stored in-memory only

### 2. LLM Evaluator (`retrieval/llm_evaluator.py`)

**Offline Behavior:**
- Uses deterministic relevancy scoring based on word overlap
- No LLM API calls (OpenAI, Ollama, etc.)
- Scores are reproducible for same inputs

**Example:**
```python
from retrieval.llm_evaluator import llm_judge_relevancy

query = "multi-agent system"
doc = "Rebecca implements a multi-agent system"

score1 = llm_judge_relevancy(query, doc)
score2 = llm_judge_relevancy(query, doc)
assert score1 == score2  # Always true
```

**Scoring Algorithm:**
- Word overlap ratio: `len(query_words ∩ doc_words) / len(query_words)`
- Normalized to [0.3, 1.0] range
- Small deterministic hash component for differentiation

### 3. Hybrid Retriever (`retrieval/hybrid_retriever.py`)

**Offline Behavior:**
- Works with in-memory BM25, Vector, and Graph indexes
- LLM evaluation uses deterministic scoring
- No network dependencies

**Example:**
```python
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex

# All indexes work in-memory
bm25 = InMemoryBM25Index()
vec = InMemoryVectorIndex()
graph = InMemoryGraphIndex()

retriever = HybridRetriever(dao, graph, bm25, vec)
results = retriever.retrieve("query", k=10, use_llm_eval=True)
# Results are deterministic
```

### 4. Concept Extractor (`knowledge_graph/concept_extractor.py`)

**Offline Behavior:**
- Skips spaCy model downloads
- Skips SentenceTransformer model loading
- Falls back to rule-based NLP methods
- No network calls for models

**Example:**
```python
from knowledge_graph.concept_extractor import ConceptExtractor

extractor = ConceptExtractor(memory_manager=None)
# In offline mode: nlp_model will be None
# Uses basic text extraction instead of spaCy NER

result = await extractor.extract_from_text("Sample text")
# Works without downloading any models
```

**Fallbacks:**
- Uses regex-based concept extraction
- Simple noun phrase detection
- Frequency-based importance scoring

### 5. LLM Stubs (`rebecca/utils.py`)

**Offline Behavior:**
- Provides `OfflineLLMStub` class for deterministic LLM responses
- Hash-based response selection
- Deterministic embeddings

**Example:**
```python
from rebecca.utils import OfflineLLMStub

stub = OfflineLLMStub()

# Deterministic response
response = stub.generate_response("Analyze this code")
# Returns template-based response

# Deterministic embedding
embedding = stub.generate_embedding("text")
# Returns 384-dim deterministic vector

# Deterministic relevance
score = stub.score_relevance("query", "document")
# Returns deterministic score
```

## Testing with Offline Mode

### Running Tests

```bash
# All tests (conftest.py enables offline mode automatically)
pytest tests/

# Specific test file
REBECCA_OFFLINE_MODE=1 pytest tests/test_memory_manager.py -v

# With coverage
REBECCA_OFFLINE_MODE=1 pytest --cov=src --cov-report=html
```

### Test Fixtures

The `tests/conftest.py` provides fixtures:

```python
# Vector store fixtures
def test_vector_operations(dummy_vector_client):
    # Uses in-memory store automatically
    pass

# Index fixtures
def test_hybrid_retrieval(dummy_bm25_index, dummy_vector_index, dummy_graph_index):
    # All in-memory indexes
    pass

# Concept extractor
def test_concept_extraction(dummy_concept_extractor):
    # No spaCy models loaded
    pass
```

### Writing Offline-Compatible Tests

```python
import pytest
from configuration import is_offline_mode

def test_my_feature():
    # This runs in offline mode automatically
    assert is_offline_mode()
    
    # All components use stubs/in-memory versions
    # Tests are deterministic and fast
```

## Limitations

### When NOT to Use Offline Mode

- **Production deployments** - Use real services
- **Quality assessment** - Real embeddings are more accurate
- **Feature testing** - May need actual LLM responses
- **Performance benchmarks** - Network latency is part of reality

### Quality Trade-offs

| Component | Online (Production) | Offline (Testing) |
|-----------|-------------------|-------------------|
| Embeddings | Semantic, 768-dim | Hash-based, 384-dim |
| NLP/NER | spaCy models | Regex patterns |
| LLM Responses | GPT/Claude/etc. | Template-based |
| Vector Store | Persistent, indexed | In-memory, simple |
| Retrieval Quality | High precision | Moderate precision |

### What Works in Offline Mode

✅ Unit tests
✅ Integration tests
✅ CI/CD pipelines
✅ Local development
✅ Smoke tests
✅ Deterministic validation

### What Doesn't Work

❌ Actual semantic search quality
❌ Real LLM intelligence
❌ Model fine-tuning
❌ Production workloads
❌ Quality benchmarks

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r src/requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests (offline mode)
        env:
          REBECCA_OFFLINE_MODE: "1"
        run: |
          pytest tests/ -v --cov=src
```

### Docker Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r src/requirements.txt

# Enable offline mode for container tests
ENV REBECCA_OFFLINE_MODE=1

CMD ["pytest", "tests/", "-v"]
```

## Verification

### Check Offline Mode Status

```bash
# Run verification script
python test_offline_mode.py

# Should output:
# ✓ Offline mode enabled
# ✓ In-memory vector store
# ✓ Deterministic embeddings
# ✓ Deterministic LLM scoring
```

### Manual Verification

```python
import os
os.environ['REBECCA_OFFLINE_MODE'] = '1'

from configuration import is_offline_mode
from memory_manager.vector_store_client import VectorStoreClient

# Check mode
print(f"Offline: {is_offline_mode()}")  # Should be True

# Check vector store
client = VectorStoreClient()
info = client.get_store_info()
print(f"Provider: {info['provider']}")  # Should be 'memory'
```

## Troubleshooting

### Issue: Tests still trying to connect to external services

**Solution:** Ensure `REBECCA_OFFLINE_MODE=1` is set before imports:

```python
import os
os.environ['REBECCA_OFFLINE_MODE'] = '1'

# Now safe to import
from src.module import something
```

### Issue: Embeddings are random/non-deterministic

**Solution:** Check that offline mode is actually enabled:

```python
from configuration import is_offline_mode
assert is_offline_mode(), "Offline mode not enabled!"
```

### Issue: spaCy trying to download models

**Solution:** The concept extractor should skip models in offline mode. If not:

```python
from knowledge_graph.concept_extractor import ConceptExtractor

# Disable semantic grouping to avoid model loading
extractor = ConceptExtractor(
    memory_manager=None,
    enable_semantic_grouping=False
)
```

## Additional Resources

- `README.md` - Main documentation with offline mode section
- `RUN_GUIDE.md` - Testing guide with offline instructions
- `config/README.md` - Configuration guide with offline setup
- `tests/conftest.py` - Pytest fixtures for offline testing
- `test_offline_mode.py` - Verification script
- `test_offline_integration.py` - Integration test example

## Support

For issues or questions about offline mode:
1. Check that `REBECCA_OFFLINE_MODE=1` is set
2. Run `python test_offline_mode.py` to verify setup
3. Review logs for "Offline mode" messages
4. Ensure imports happen after environment variable is set
