import logging
import os
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from common.settings import get_settings
from common.misc_utils import set_log_level, get_logger
from common.emb_utils import get_embedder
from common.llm_utils import tokenize_with_llm

log_level = logging.INFO
level = os.getenv("LOG_LEVEL", "").removeprefix("--").lower()
if level != "":
    if "debug" in level:
        log_level = logging.DEBUG
    elif not "info" in level:
        logging.warning("Unknown LOG_LEVEL passed: '%s'", level)
set_log_level(log_level)
logger = get_logger("similarity")

settings = get_settings()


class SimilarityException(Exception):
    """Custom exception for similarity search errors."""
    def __init__(self, code: int, status: str, message: str):
        self.code = code
        self.message = message
        self.status = status


def validate_query_length(query: str, emb_endpoint: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the query length does not exceed the maximum allowed tokens.
    
    Args:
        query: The search query string
        emb_endpoint: The embedding model endpoint
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        tokens = tokenize_with_llm(query, emb_endpoint)
        token_count = len(tokens)
        
        if token_count > settings.max_query_token_length:
            error_msg = f"Query length ({token_count} tokens) exceeds maximum allowed length of {settings.max_query_token_length} tokens"
            logger.warning(error_msg)
            return False, error_msg
        
        return True, None
    except Exception as e:
        logger.error(f"Error validating query length: {e}")
        # If tokenization fails, allow the request to proceed
        return True, None


def retrieve_documents(
    query: str,          # The search query
    emb_model: str,      # Embedding model name
    emb_endpoint: str,   # Embedding endpoint URL
    max_tokens: int,     # Maximum tokens for embedding
    vectorstore,        # Vector store instance    
    top_k: int,        # Number of results to return
    mode: str = "dense"  # Search mode ("dense" for k-NN similarity)
) -> Tuple[List[dict], List[float]]:
    
    embedding = get_embedder(emb_model, emb_endpoint, max_tokens)
    results = vectorstore.search(query, embedding=embedding, top_k=top_k, mode=mode)

    retrieved_documents = []
    scores = []

    for hit in results:
        doc = {
            "page_content": hit.get("page_content", ""),
            "filename": hit.get("filename", ""),
            "type": hit.get("type", ""),
            "source": hit.get("source", ""),
            "chunk_id": hit.get("chunk_id", "")
        }
        retrieved_documents.append(doc)
        
        # For dense hits from OpenSearch, we expect cosine similarity score
        score = hit.get("score") or hit.get("distance") or 0.0
        scores.append(score)

    return retrieved_documents, scores #Returns retrieved documents with there similarity scores


def rerank_documents(
    query: str,
    documents: List[dict], #List the documents that need to be reranked
    reranker_model: str, #Name of the reranker model
    reranker_endpoint: str #Endpoint URL for the reranker model
) -> List[Tuple[dict, float]]:

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from cohere import ClientV2
    
    co2 = ClientV2(api_key="sk-fake-key", base_url=reranker_endpoint)
    reranked: List[Tuple[dict, float]] = []
    
    def rerank_helper(doc: dict) -> Tuple[dict, float]:
        try:
            page_content = doc.get("page_content", "")
            if not page_content:
                logger.warning("Document has no page_content, assigning score 0.0")
                return doc, 0.0
            
            result = co2.rerank(
                model=reranker_model,
                query=query,
                documents=[page_content],
                max_tokens_per_doc=512,
            )
            score = result.results[0].relevance_score
            return doc, score
        except Exception as e:
            logger.error(f"Rerank Error: {e}")
            return doc, 0.0
    
    max_workers = min(8, len(documents))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(rerank_helper, doc): doc for doc in documents}
        
        for future in as_completed(futures):
            try:
                reranked.append(future.result())
            except Exception as e:
                doc = futures[future]
                logger.error(f"Thread error: {e}")
                reranked.append((doc, 0.0))
    
    return sorted(reranked, key=lambda x: x[1], reverse=True) #Returns a list of documents that are sorted by score.(descending order)


def build_success_response(
    results: List[dict],  #A list of retrieved documents
    scores: List[float],  # A list of the similairty/relevance scrores
    score_type: str,      #If the score is a cosine similarity or a relevance score
    processing_time_ms: int #The time it took to process the request in milliseconds
) -> dict:
   
    formatted_results = []
    for i, (doc, score) in enumerate(zip(results, scores), 1):
        formatted_results.append({
            "page_content": doc.get("page_content", ""),
            "filename": doc.get("filename", ""),
            "type": doc.get("type", ""),
            "source": doc.get("source", ""),
            "chunk_id": doc.get("chunk_id", ""),
            "score": score,
            "rank": i
        })
    
    return {
        "score_type": score_type,
        "results": formatted_results,
        "meta": {
            "processing_time_ms": processing_time_ms,
            "result_count": len(formatted_results)
        }
    }


# Pydantic Models for API

class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    rerank: bool = Field(default=False, description="Apply Cohere reranker to re-score results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How do I configure network settings?",
                "top_k": 5,
                "rerank": False
            }
        }
    }


class SimilaritySearchResult(BaseModel):
    page_content: str = Field(..., description="Document content")
    filename: str = Field(..., description="Source filename")
    type: str = Field(..., description="Document type (text, image, table)")
    source: str = Field(..., description="Source path or identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Similarity or relevance score")
    rank: int = Field(..., description="Result ranking position")


class SimilaritySearchMeta(BaseModel):
    processing_time_ms: int = Field(..., description="Request processing time in milliseconds")
    result_count: int = Field(..., description="Number of results returned")


class SimilaritySearchResponse(BaseModel):
    score_type: str = Field(..., description="Type of score: 'cosine' or 'relevance'")
    results: List[SimilaritySearchResult]
    meta: SimilaritySearchMeta

    model_config = {
        "json_schema_extra": {
            "example": {
                "score_type": "cosine",
                "results": [
                    {
                        "page_content": "To configure network settings, navigate to the system preferences...",
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
        }
    }


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    status: int = Field(..., description="HTTP status code")


class SimilarityErrorResponse(BaseModel):
    error: ErrorDetail

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "QUERY_REQUIRED",
                    "message": "query is required",
                    "status": 400
                }
            }
        }
    }


error_responses: Dict[int | str, Dict[str, Any]] = {
    400: {"description": "Bad request (missing query, invalid parameters)", "model": SimilarityErrorResponse},
    503: {"description": "Index is empty. Ingest documents first.", "model": SimilarityErrorResponse},
    500: {"description": "Internal server error", "model": SimilarityErrorResponse},
}


