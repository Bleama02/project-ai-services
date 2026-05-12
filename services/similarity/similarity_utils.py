from typing import Optional

from pydantic import BaseModel, Field

from common.retrieval_utils import retrieve_documents
from common.reranker_utils import rerank_documents
from common.error_utils import http_error_responses
from similarity.settings import settings


class SimilaritySearchRequest(BaseModel):
    """Request body for POST /v1/similarity-search"""
    query: str = Field(..., description="Natural language search query")
    mode: str = Field(
        default="dense",
        description="A mode parameter to select the search strategy: dense  K-NN, sparse BM25, or hybrid (both)"
    )
    top_k: int = Field(
        default=settings.similarity.num_chunks_post_search,
        ge=1,
        description="Number of results to return (minimum 1). Defaults to num_chunks_post_search from settings."
    )
    rerank: bool = Field(
        default=False,
        description="When true, applies Cohere reranker to re-score and re-order results."
    )


class SimilaritySearchResult(BaseModel):
    """A single document result with its score."""
    page_content: str = Field(..., description="Text content of the chunk")
    filename: str = Field(..., description="Source filename")
    type: str = Field(..., description="Document type: text, image, or table")
    source: str = Field(..., description="Source path or HTML content")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Cosine similarity (rerank=false) or relevance score (rerank=true)")


class SimilaritySearchResponse(BaseModel):
    """Response from POST /v1/similarity-search"""
    score_type: str = Field(
        ...,
        description="'cosine' for dense-only results, 'relevance' when reranked"
    )
    results: list[SimilaritySearchResult] = Field(
        ...,
        description="Documents ranked by descending score"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "score_type": "cosine",
                "results": [
                    {
                        "page_content": "To configure network settings, navigate to system preferences...",
                        "filename": "admin-guide.pdf",
                        "type": "text",
                        "source": "admin-guide.pdf",
                        "chunk_id": "8374619250",
                        "score": 0.8742
                    },
                    {
                        "page_content": "Network troubleshooting can be performed by checking connection status...",
                        "filename": "troubleshooting.pdf",
                        "type": "text",
                        "source": "troubleshooting.pdf",
                        "chunk_id": "5091837264",
                        "score": 0.7518
                    }
                ]
            }
        }
    }



def perform_similarity_search(
    query: str,
    emb_model: str,
    emb_endpoint: str,
    emb_max_tokens: int,
    vectorstore,
    top_k: int,
    rerank: bool,
    mode: str,
    reranker_model: Optional[str] = None,
    reranker_endpoint: Optional[str] = None,
    top_r: Optional[int] = None,
    score_threshold: Optional[float] = None,
    track_performance: bool = False,
) -> tuple[list[dict], list[float], str, Optional[dict]]:
    """
    Run vector similarity search using the specified mode, with optional Cohere reranking.

    Args:
        query: Search query string
        emb_model: Embedding model name
        emb_endpoint: Embedding service endpoint
        emb_max_tokens: Maximum tokens for embedding
        vectorstore: Vector database instance
        top_k: Number of documents to retrieve initially
        rerank: Whether to apply reranking
        mode: Search mode ("dense", "sparse", or "hybrid")
        reranker_model: Reranker model name (required if rerank=True)
        reranker_endpoint: Reranker service endpoint (required if rerank=True)
        top_r: Optional limit on number of results after reranking (e.g., 3 for chatbot)
        score_threshold: Optional minimum score threshold for filtering results
        track_performance: Whether to track and return performance metrics

    Returns:
        docs       - list of document dicts (page_content, filename, type, source, chunk_id)
        scores     - parallel list of float scores
        score_type - "cosine", "bm25", "hybrid", or "relevance" (when reranked)
        perf_stats - dict with timing metrics if track_performance=True, else None
    """
    import time
    
    perf_stats: Optional[dict] = {} if track_performance else None
    start_time: float = 0.0
    
    # Retrieve documents with optional timing
    if track_performance and perf_stats is not None:
        start_time = time.time()
    
    docs, scores = retrieve_documents(
        query,
        emb_model,
        emb_endpoint,
        emb_max_tokens,
        vectorstore,
        top_k,
        mode=mode,
    )
    
    if track_performance and perf_stats is not None:
        perf_stats["retrieve_time"] = time.time() - start_time

    score_type_map = {
        "dense": "cosine",
        "hybrid": "hybrid",
        "sparse": "bm25"
    }
    score_type = score_type_map.get(mode, "cosine")

    if rerank:
        # Ensure reranker parameters are provided when reranking is requested
        if not reranker_model or not reranker_endpoint:
            raise ValueError("reranker_model and reranker_endpoint are required when rerank=True")
        
        if track_performance and perf_stats is not None:
            start_time = time.time()
        
        reranked = rerank_documents(query, docs, reranker_model, reranker_endpoint)
        
        if track_performance and perf_stats is not None:
            perf_stats["rerank_time"] = time.time() - start_time
        
        # Apply top_r limit if specified
        if top_r is not None and top_r > 0:
            reranked = reranked[:top_r]
        
        docs = [d for d, _ in reranked]
        scores = [s for _, s in reranked]
        score_type = "relevance"

    # Apply score threshold filter if specified
    if score_threshold is not None:
        filtered = [(d, s) for d, s in zip(docs, scores) if s >= score_threshold]
        docs = [d for d, _ in filtered]
        scores = [s for _, s in filtered]

    return docs, scores, score_type, perf_stats
