"""
Unit tests for the similarity search API.

Run with: pytest test_similarity_search.py -v
Or: python -m pytest test_similarity_search.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity_search.app import app
from similarity_search.response_utils import (
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    Document,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for testing."""
    mock_vs = Mock()
    mock_vs.check_db_populated.return_value = True
    return mock_vs


@pytest.fixture
def mock_search_results():
    """Mock search results."""
    return [
        {
            "page_content": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "filename": "ai_basics.pdf",
            "type": "text",
            "source": "/docs/ai_basics.pdf",
            "chunk_id": 1001
        },
        {
            "page_content": "Machine learning is a subset of artificial intelligence.",
            "filename": "ml_intro.pdf",
            "type": "text",
            "source": "/docs/ml_intro.pdf",
            "chunk_id": 1002
        }
    ]


@pytest.fixture
def mock_scores():
    """Mock similarity scores."""
    return [0.95, 0.87]


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self, client):
        """Test the /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    @patch('similarity_search.app.vectorstore')
    def test_db_status_ready(self, mock_vs, client):
        """Test /db-status when database is ready."""
        mock_vs.check_db_populated.return_value = True
        
        response = client.get("/db-status")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
    
    @patch('similarity_search.app.vectorstore')
    def test_db_status_not_ready(self, mock_vs, client):
        """Test /db-status when database is not ready."""
        mock_vs.check_db_populated.return_value = False
        
        response = client.get("/db-status")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert "message" in data


class TestSimilaritySearchEndpoint:
    """Test the main similarity search endpoint."""
    
    @patch('similarity_search.app.vectorstore')
    @patch('similarity_search.backend_utils.retrieve_documents')
    @patch('similarity_search.backend_utils.validate_query_length')
    def test_similarity_search_without_rerank(
        self, mock_validate, mock_retrieve, mock_vs, client, mock_search_results, mock_scores
    ):
        """Test similarity search without reranking."""
        # Setup mocks
        mock_validate.return_value = (True, None)
        mock_retrieve.return_value = (mock_search_results, mock_scores)
        mock_vs.check_db_populated.return_value = True
        
        # Make request
        payload = {
            "query": "What is artificial intelligence?",
            "top_k": 2,
            "rerank": False
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["score_type"] == "cosine"
        assert "perf_metrics" in data
        
        # Check first result
        first_result = data["results"][0]
        assert first_result["rank"] == 1
        assert first_result["score"] == 0.95
        assert "document" in first_result
        assert first_result["document"]["page_content"] == mock_search_results[0]["page_content"]
    
    @patch('similarity_search.app.vectorstore')
    @patch('similarity_search.backend_utils.retrieve_documents')
    @patch('similarity_search.backend_utils.rerank_documents')
    @patch('similarity_search.backend_utils.validate_query_length')
    def test_similarity_search_with_rerank(
        self, mock_validate, mock_rerank, mock_retrieve, mock_vs, 
        client, mock_search_results, mock_scores
    ):
        """Test similarity search with reranking."""
        # Setup mocks
        mock_validate.return_value = (True, None)
        mock_retrieve.return_value = (mock_search_results, mock_scores)
        
        # Mock reranking - returns list of (doc, score) tuples
        reranked_results = [
            (mock_search_results[1], 0.92),  # Second doc ranked higher
            (mock_search_results[0], 0.88)   # First doc ranked lower
        ]
        mock_rerank.return_value = reranked_results
        mock_vs.check_db_populated.return_value = True
        
        # Make request
        payload = {
            "query": "What is machine learning?",
            "top_k": 2,
            "rerank": True
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["score_type"] == "relevance"  # Changed from cosine
        assert len(data["results"]) == 2
        
        # Verify reranking changed the order
        first_result = data["results"][0]
        assert first_result["score"] == 0.92
        assert "machine learning" in first_result["document"]["page_content"].lower()
    
    def test_similarity_search_empty_query(self, client):
        """Test that empty queries are rejected."""
        payload = {
            "query": "",
            "top_k": 5,
            "rerank": False
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_similarity_search_whitespace_query(self, client):
        """Test that whitespace-only queries are rejected."""
        payload = {
            "query": "   ",
            "top_k": 5,
            "rerank": False
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        assert response.status_code == 400
    
    @patch('similarity_search.app.vectorstore')
    @patch('similarity_search.backend_utils.validate_query_length')
    def test_similarity_search_query_too_long(self, mock_validate, mock_vs, client):
        """Test that overly long queries are rejected."""
        mock_validate.return_value = (False, "Query too long")
        mock_vs.check_db_populated.return_value = True
        
        payload = {
            "query": "a" * 10000,  # Very long query
            "top_k": 5,
            "rerank": False
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        assert response.status_code == 400
        assert "too long" in response.json()["detail"].lower()
    
    def test_similarity_search_invalid_top_k(self, client):
        """Test validation of top_k parameter."""
        payload = {
            "query": "test query",
            "top_k": -1,  # Invalid
            "rerank": False
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        assert response.status_code == 422  # Validation error
    
    @patch('similarity_search.app.vectorstore', None)
    def test_similarity_search_db_not_ready(self, client):
        """Test behavior when vector store is not initialized."""
        payload = {
            "query": "test query",
            "top_k": 5,
            "rerank": False
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        # Should handle gracefully - might be 503 or 500 depending on implementation
        assert response.status_code in [500, 503]


class TestModelsEndpoint:
    """Test the models listing endpoint."""
    
    @patch('similarity_search.app.llm_model_dict', {'llm_endpoint': 'http://mock-llm:8000'})
    @patch('similarity_search.app.query_vllm_models')
    def test_list_models(self, mock_query, client):
        """Test listing available models."""
        mock_query.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "ibm-granite/granite-3.3-8b-instruct",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "ibm"
                }
            ]
        }
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "granite" in data["data"][0]["id"]


class TestPerfMetricsEndpoint:
    """Test performance metrics endpoint."""
    
    @patch('similarity_search.app.perf_registry')
    def test_get_all_metrics(self, mock_registry, client):
        """Test retrieving all performance metrics."""
        mock_metrics = [
            {
                "timestamp": 1678901234.567,
                "readable_timestamp": "2023-03-15 14:30:34",
                "request_id": "test-123",
                "retrieve_time": 0.15,
                "rerank_time": 0.12
            }
        ]
        mock_registry.get_metrics.return_value = mock_metrics
        
        response = client.get("/v1/perf_metrics")
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["request_id"] == "test-123"
    
    @patch('similarity_search.app.perf_registry')
    def test_get_metric_by_request_id(self, mock_registry, client):
        """Test retrieving a specific metric by request ID."""
        mock_metric = {
            "timestamp": 1678901234.567,
            "readable_timestamp": "2023-03-15 14:30:34",
            "request_id": "test-456",
            "retrieve_time": 0.20,
            "rerank_time": 0.15
        }
        mock_registry.get_metric_by_request_id.return_value = mock_metric
        
        response = client.get("/v1/perf_metrics?request_id=test-456")
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["request_id"] == "test-456"
    
    @patch('similarity_search.app.perf_registry')
    def test_get_metric_not_found(self, mock_registry, client):
        """Test retrieving a non-existent metric."""
        mock_registry.get_metric_by_request_id.return_value = None
        
        response = client.get("/v1/perf_metrics?request_id=nonexistent")
        assert response.status_code == 404


class TestRequestModels:
    """Test Pydantic request models."""
    
    def test_similarity_search_request_defaults(self):
        """Test default values in SimilaritySearchRequest."""
        request = SimilaritySearchRequest(query="test")
        assert request.query == "test"
        assert request.top_k == 10  # Default
        assert request.rerank is False  # Default
    
    def test_similarity_search_request_custom(self):
        """Test custom values in SimilaritySearchRequest."""
        request = SimilaritySearchRequest(
            query="custom query",
            top_k=5,
            rerank=True
        )
        assert request.query == "custom query"
        assert request.top_k == 5
        assert request.rerank is True


class TestIntegration:
    """Integration tests that test multiple components together."""
    
    @patch('similarity_search.app.vectorstore')
    @patch('similarity_search.backend_utils.retrieve_documents')
    @patch('similarity_search.backend_utils.rerank_documents')
    @patch('similarity_search.backend_utils.validate_query_length')
    def test_full_search_pipeline_with_rerank(
        self, mock_validate, mock_rerank, mock_retrieve, mock_vs, 
        client, mock_search_results, mock_scores
    ):
        """Test the complete search pipeline from request to response."""
        # Setup all mocks
        mock_validate.return_value = (True, None)
        mock_retrieve.return_value = (mock_search_results, mock_scores)
        mock_rerank.return_value = [
            (mock_search_results[0], 0.95),
            (mock_search_results[1], 0.87)
        ]
        mock_vs.check_db_populated.return_value = True
        
        # Test the full flow
        payload = {
            "query": "Explain artificial intelligence",
            "top_k": 2,
            "rerank": True
        }
        
        response = client.post("/v1/similarity-search", json=payload)
        
        # Verify complete response structure
        assert response.status_code == 200
        data = response.json()
        
        # Check all expected fields
        assert "results" in data
        assert "score_type" in data
        assert "perf_metrics" in data
        
        # Verify results structure
        for result in data["results"]:
            assert "document" in result
            assert "score" in result
            assert "rank" in result
            
            doc = result["document"]
            assert "page_content" in doc
            assert "filename" in doc
            assert "type" in doc
            assert "source" in doc
            assert "chunk_id" in doc
        
        # Verify performance metrics
        assert "retrieve_time" in data["perf_metrics"]
        assert "rerank_time" in data["perf_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# Made with Bob
