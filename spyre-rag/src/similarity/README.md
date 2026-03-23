# Similarity Search API

Vector similarity search service with optional reranking.

## Overview

This service provides a REST API endpoint for performing vector similarity search using dense k-NN (cosine similarity) with optional Cohere-based reranking.

## Features

- **Dense k-NN Search**: Pure vector similarity search using cosine similarity
- **Optional Reranking**: Apply Cohere reranker for improved relevance
- **Configurable Results**: Control the number of results returned (1-100)
- **Fast Performance**: Optimized for low-latency retrieval

## API Endpoints

### POST /v1/similarity-search

Perform vector similarity search.

**Request Body:**
```json
{
  "query": "How do I configure network settings?",
  "top_k": 5,
  "rerank": false
}
```

**Response (without reranking):**
```json
{
  "score_type": "cosine",
  "results": [
    {
      "page_content": "To configure network settings...",
      "filename": "admin-guide.pdf",
      "type": "text",
      "source": "admin-guide.pdf",
      "chunk_id": "8374619250",
      "score": 0.8742,
      "rank": 1
    }
  ],
  "meta": {
    "processing_time_ms": 245,
    "result_count": 1
  }
}
```

**Response (with reranking):**
```json
{
  "score_type": "relevance",
  "results": [
    {
      "page_content": "To configure network settings...",
      "filename": "admin-guide.pdf",
      "type": "text",
      "source": "admin-guide.pdf",
      "chunk_id": "8374619250",
      "score": 0.9215,
      "rank": 1
    }
  ],
  "meta": {
    "processing_time_ms": 512,
    "result_count": 1
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### GET /db-status

Check vector database status.

**Response:**
```json
{
  "ready": true
}
```

## Running the Service

### Prerequisites

- Python 3.9+
- Required dependencies installed
- Vector database (OpenSearch) running and populated
- Embedding model endpoint configured
- (Optional) Reranker model endpoint for reranking

### Start the Server

```bash
cd spyre-rag/src
python -m similarity.app
```

The service will start on port 7000 by default. You can override this with the `PORT` environment variable:

```bash
PORT=8000 python -m similarity.app
```

## Testing

### Basic Search (Cosine Similarity)

```bash
curl -X POST http://localhost:7000/v1/similarity-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search text",
    "top_k": 5
  }'
```

### Search with Reranking

```bash
curl -X POST http://localhost:7000/v1/similarity-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search text",
    "top_k": 5,
    "rerank": true
  }'
```

### Health Check

```bash
curl http://localhost:7000/health
```

## Configuration

The service uses settings from `common.settings`. Key configuration options:

- `max_concurrent_requests`: Maximum concurrent requests (default: from settings)
- `max_query_token_length`: Maximum query length in tokens
- `score_threshold`: Minimum relevance score for reranked results

## Architecture

The service follows the same architectural pattern as the summarize service:

- **FastAPI** for the web framework
- **Async/await** for concurrent request handling
- **Pydantic** for request/response validation
- **Concurrency limiting** to prevent overload
- **Request ID tracking** for debugging

## Error Handling

The service returns structured error responses:

- **400**: Bad request (missing query, invalid parameters)
- **429**: Server busy (too many concurrent requests)
- **500**: Internal server error
- **503**: Vector database not ready or empty

## Comparison with Other Endpoints

| Aspect | `/v1/similarity-search` | `/reference` | `/v1/chat/completions` |
|--------|------------------------|--------------|------------------------|
| Search mode | Dense only (k-NN) | Hybrid (dense + BM25) | Hybrid (dense + BM25) |
| Reranking | Optional (default: No) | Yes (always) | Yes (always) |
| LLM generation | No | No | Yes |
| Score type | Cosine or relevance | Relevance | Relevance |
| Latency | Low (default) | Medium | High |