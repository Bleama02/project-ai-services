import asyncio
import time
import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse

from common.misc_utils import set_log_level, get_logger
log_level = logging.INFO
level = os.getenv("LOG_LEVEL", "").removeprefix("--").lower()
if level != "":
    if "debug" in level:
        log_level = logging.DEBUG
    elif not "info" in level:
        logging.warning(f"Unknown LOG_LEVEL passed: '{level}'")

set_log_level(log_level)

from common.misc_utils import get_model_endpoints, set_request_id
from common.settings import get_settings
import common.db_utils as db
from similarity.similarity_utils import (
    SimilarityException,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    SimilaritySearchResult,
    SimilaritySearchMeta,
    error_responses,
    validate_query_length,
    retrieve_documents,
    rerank_documents,
    build_success_response,
)

logger = get_logger("app")

settings = get_settings()
concurrency_limiter = asyncio.BoundedSemaphore(settings.max_concurrent_requests)

# Global variables for models and vectorstore
vectorstore = None
emb_model_dict = {}
reranker_model_dict = {}


def initialize_models():
    """Initialize model endpoints at startup."""
    global emb_model_dict, reranker_model_dict
    emb_model_dict, _, reranker_model_dict = get_model_endpoints()


def initialize_vectorstore():
    """Initialize vector store at startup."""
    global vectorstore
    vectorstore = db.get_vector_store()


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup and shutdown."""
    initialize_models()
    initialize_vectorstore()
    yield


# OpenAPI tags metadata for endpoint organization
tags_metadata = [
    {
        "name": "similarity",
        "description": "Vector similarity search operations with optional reranking"
    },
    {
        "name": "health",
        "description": "Health check and service status"
    }
]

app = FastAPI(
    lifespan=lifespan,
    title="AI-Services Similarity Search API",
    description="Performs vector similarity search (dense k-NN) with optional Cohere-based reranking.",
    version="1.0.0",
    openapi_tags=tags_metadata
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests and responses."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", include_in_schema=False)
def swagger_root():
    """Expose Swagger UI at the root path (/)"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AI-Services Similarity Search API - Swagger UI",
    )


@app.exception_handler(SimilarityException)
async def similarity_exception_handler(request: Request, exc: SimilarityException):
    """Handle custom SimilarityException errors."""
    return JSONResponse(
        status_code=exc.code,
        content={
            "error": {
                "code": exc.status,
                "message": exc.message,
                "status": exc.code,
            }
        },
    )


async def handle_similarity_search(
    query: str,
    top_k: int,
    rerank: bool = False,
):
    """
    Core similarity search logic.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        rerank: Whether to apply reranking
        
    Returns:
        SimilaritySearchResponse with results
    """
    # Get model configurations
    emb_model = emb_model_dict['emb_model']
    emb_endpoint = emb_model_dict['emb_endpoint']
    emb_max_tokens = emb_model_dict['max_tokens']
    reranker_model = reranker_model_dict.get('reranker_model', '')
    reranker_endpoint = reranker_model_dict.get('reranker_endpoint', '')

    # Validate query length
    is_valid, error_msg = await asyncio.to_thread(
        validate_query_length, query, emb_endpoint
    )
    if not is_valid:
        raise SimilarityException(400, "QUERY_TOO_LONG", error_msg or "Query is too long")

    logger.info(f"Received similarity search request: query length={len(query)}, top_k={top_k}, rerank={rerank}")

    start_time = time.time()

    # Retrieve documents using dense k-NN search
    try:
        docs, scores = await asyncio.to_thread(
            retrieve_documents,
            query,
            emb_model,
            emb_endpoint,
            emb_max_tokens,
            vectorstore,
            top_k,
            mode="dense"
        )
    except db.VectorStoreNotReadyError as e:
        raise SimilarityException(503, "INDEX_EMPTY", "Index is empty. Ingest documents first.")
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        raise SimilarityException(500, "RETRIEVAL_ERROR", f"Failed to retrieve documents: {str(e)}")

    # Apply reranking if requested
    if rerank and docs:
        if not reranker_model or not reranker_endpoint:
            raise SimilarityException(500, "RERANKER_NOT_CONFIGURED", "Reranker model is not configured")
        try:
            reranked = await asyncio.to_thread(
                rerank_documents,
                query,
                docs,
                reranker_model,
                reranker_endpoint
            )
            # Extract documents and scores from reranked results
            docs = [doc for doc, _ in reranked[:top_k]]
            scores = [score for _, score in reranked[:top_k]]
            score_type = "relevance"
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise SimilarityException(500, "RERANKING_ERROR", f"Failed to rerank documents: {str(e)}")
    else:
        score_type = "cosine"

    elapsed_ms = int((time.time() - start_time) * 1000)

    logger.info(f"Similarity search completed in {elapsed_ms}ms, returned {len(docs)} results")

    # Build response
    response_data = build_success_response(
        results=docs,
        scores=scores,
        score_type=score_type,
        processing_time_ms=elapsed_ms
    )

    return SimilaritySearchResponse(**response_data)


@app.post(
    "/v1/similarity-search",
    response_model=SimilaritySearchResponse,
    responses=error_responses,
    summary="Similarity Search",
    description=(
        "Performs vector similarity search using dense k-NN (cosine similarity) "
        "with optional Cohere-based reranking.\n\n"
        "### Request Parameters\n\n"
        "| Field | Type | Required | Default | Description |\n"
        "|-------|------|----------|---------|-------------|\n"
        "| `query` | string | Yes | - | Natural language search query |\n"
        "| `top_k` | integer | No | 10 | Number of results to return (1-100) |\n"
        "| `rerank` | boolean | No | false | Apply Cohere reranker to re-score results |\n\n"
        "### Response\n\n"
        "Returns documents ranked by similarity score (cosine) or relevance score (if reranked).\n\n"
        "**Example Request:**\n"
        "```bash\n"
        'curl -X POST /v1/similarity-search -H "Content-Type: application/json" -d \'{\n'
        '  "query": "How do I configure network settings?",\n'
        '  "top_k": 5,\n'
        '  "rerank": false\n'
        "}\'\n"
        "```"
    ),
    response_description="Similarity search results with scores and metadata.",
    tags=["similarity"],
)
async def similarity_search(request: Request, req: SimilaritySearchRequest):
    """
    Perform vector similarity search with optional reranking.
    
    - **query**: Natural language search query (required)
    - **top_k**: Number of results to return (default: 10, range: 1-100)
    - **rerank**: Apply reranking for better relevance (default: false)
    """
    try:
        # Check if server is busy
        if concurrency_limiter.locked():
            raise SimilarityException(429, "SERVER_BUSY", "Server is busy. Please try again later.")

        # Validate query
        if not req.query or not req.query.strip():
            raise SimilarityException(400, "QUERY_REQUIRED", "query is required")

        async with concurrency_limiter:
            return await handle_similarity_search(
                query=req.query.strip(),
                top_k=req.top_k,
                rerank=req.rerank
            )

    except SimilarityException as se:
        raise se
    except Exception as e:
        logger.error(f"Unexpected error in similarity search: {e}")
        raise SimilarityException(500, "INTERNAL_SERVER_ERROR", f"An unexpected error occurred: {str(e)}")


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the service is running and healthy.",
    response_description="Service health status"
)
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get(
    "/db-status",
    tags=["health"],
    summary="Database status",
    description="Check if the vector database is initialized and populated.",
    response_description="Database status"
)
async def db_status():
    """Check vector database status."""
    try:
        if vectorstore is None:
            return {"ready": False, "message": "Vector store not initialized"}

        status = await asyncio.to_thread(vectorstore.check_db_populated)
        if status:
            return {"ready": True}
        else:
            return {"ready": False, "message": "No data ingested"}

    except Exception as e:
        return {"ready": False, "message": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Made with Bob
