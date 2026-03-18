# Similarity Search Unit Tests

This directory contains comprehensive unit tests for the similarity search API.

## Installation

First, install the required testing dependencies:

```bash
pip install pytest pytest-asyncio httpx
```

Or if you have a requirements file:

```bash
pip install -r requirements-test.txt
```

## Running the Tests

### Run all tests:
```bash
cd spyre-rag/src/similarity_search
pytest test_similarity_search.py -v
```

### Run specific test classes:
```bash
# Test only health endpoints
pytest test_similarity_search.py::TestHealthEndpoints -v

# Test only similarity search endpoint
pytest test_similarity_search.py::TestSimilaritySearchEndpoint -v

# Test only performance metrics
pytest test_similarity_search.py::TestPerfMetricsEndpoint -v
```

### Run specific test methods:
```bash
pytest test_similarity_search.py::TestSimilaritySearchEndpoint::test_similarity_search_without_rerank -v
```

### Run with coverage:
```bash
pip install pytest-cov
pytest test_similarity_search.py --cov=. --cov-report=html
```

### Run with detailed output:
```bash
pytest test_similarity_search.py -vv -s
```

## Test Structure

The test suite is organized into the following classes:

### 1. **TestHealthEndpoints**
- `test_health_check()` - Tests the `/health` endpoint
- `test_db_status_ready()` - Tests `/db-status` when database is ready
- `test_db_status_not_ready()` - Tests `/db-status` when database is not ready

### 2. **TestSimilaritySearchEndpoint**
- `test_similarity_search_without_rerank()` - Tests basic similarity search
- `test_similarity_search_with_rerank()` - Tests search with reranking enabled
- `test_similarity_search_empty_query()` - Tests empty query validation
- `test_similarity_search_whitespace_query()` - Tests whitespace-only query validation
- `test_similarity_search_query_too_long()` - Tests query length validation
- `test_similarity_search_invalid_top_k()` - Tests invalid top_k parameter
- `test_similarity_search_db_not_ready()` - Tests behavior when DB is not ready

### 3. **TestModelsEndpoint**
- `test_list_models()` - Tests the `/v1/models` endpoint

### 4. **TestPerfMetricsEndpoint**
- `test_get_all_metrics()` - Tests retrieving all performance metrics
- `test_get_metric_by_request_id()` - Tests retrieving specific metric
- `test_get_metric_not_found()` - Tests handling of non-existent metrics

### 5. **TestRequestModels**
- `test_similarity_search_request_defaults()` - Tests default values
- `test_similarity_search_request_custom()` - Tests custom values

### 6. **TestIntegration**
- `test_full_search_pipeline_with_rerank()` - Tests complete end-to-end flow

## What the Tests Cover

✅ **API Endpoints**: All REST endpoints are tested
✅ **Request Validation**: Empty queries, invalid parameters, query length
✅ **Reranking Logic**: Tests both with and without reranking
✅ **Error Handling**: Database errors, validation errors, not found errors
✅ **Response Structure**: Validates all response fields and types
✅ **Performance Metrics**: Tests metric collection and retrieval
✅ **Health Checks**: Database status and service health

## Mocking Strategy

The tests use mocking to isolate the API layer from dependencies:

- **Vector Store**: Mocked to avoid needing a real OpenSearch instance
- **Retrieval Functions**: Mocked to return predictable test data
- **Reranking Functions**: Mocked to simulate reranking behavior
- **LLM Endpoints**: Mocked to avoid external API calls

This allows tests to run quickly and reliably without external dependencies.

## Example Test Output

```
test_similarity_search.py::TestHealthEndpoints::test_health_check PASSED
test_similarity_search.py::TestHealthEndpoints::test_db_status_ready PASSED
test_similarity_search.py::TestSimilaritySearchEndpoint::test_similarity_search_without_rerank PASSED
test_similarity_search.py::TestSimilaritySearchEndpoint::test_similarity_search_with_rerank PASSED
test_similarity_search.py::TestSimilaritySearchEndpoint::test_similarity_search_empty_query PASSED
...

========================= 20 passed in 2.34s =========================
```

## Continuous Integration

To run these tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-asyncio httpx
    cd spyre-rag/src/similarity_search
    pytest test_similarity_search.py -v --tb=short
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the correct directory:
```bash
cd spyre-rag/src/similarity_search
python -m pytest test_similarity_search.py -v
```

### Missing Dependencies
Install all test dependencies:
```bash
pip install pytest pytest-asyncio httpx fastapi
```

### Mock Issues
If mocks aren't working, ensure you're patching the correct path:
- Use `similarity_search.app.vectorstore` not just `vectorstore`
- Patch where the object is used, not where it's defined

## Adding New Tests

To add new tests:

1. Create a new test class or add to an existing one
2. Use descriptive test names: `test_<what>_<condition>_<expected>`
3. Follow the AAA pattern: Arrange, Act, Assert
4. Use fixtures for common setup
5. Mock external dependencies

Example:
```python
def test_new_feature_with_valid_input_returns_success(self, client):
    # Arrange
    payload = {"query": "test", "top_k": 5}
    
    # Act
    response = client.post("/v1/similarity-search", json=payload)
    
    # Assert
    assert response.status_code == 200
```

## Questions?

For questions or issues with the tests, please check:
1. The test output for specific error messages
2. The main app.py file for API implementation details
3. The response_utils.py file for request/response models