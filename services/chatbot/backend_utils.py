from common.misc_utils import get_logger
from common.validation_utils import validate_query_length as _validate_query_length
from similarity.similarity_utils import perform_similarity_search
from chatbot.settings import settings

logger = get_logger("backend_utils")


def validate_query_length(query, emb_endpoint):
    return _validate_query_length(query, emb_endpoint, settings.chatbot.max_query_token_length)

def search_only(question, emb_model, emb_endpoint, max_tokens, reranker_model, reranker_endpoint, top_k, top_r, vectorstore):
    """
    Perform document retrieval with reranking, filtering, and performance tracking.
    
    This function uses the shared perform_similarity_search() from similarity service
    to ensure consistent retrieval behavior across the application.
    
    Args:
        question: Search query
        emb_model: Embedding model name
        emb_endpoint: Embedding service endpoint
        max_tokens: Maximum tokens for embedding
        reranker_model: Reranker model name
        reranker_endpoint: Reranker service endpoint
        top_k: Number of documents to retrieve initially
        top_r: Number of documents to keep after reranking
        vectorstore: Vector database instance
        
    Returns:
        filtered_docs: List of documents that pass score threshold
        perf_stat_dict: Performance metrics (retrieve_time, rerank_time)
    """
    # Use shared retrieval function with chatbot-specific parameters
    docs, scores, _, perf_stat_dict = perform_similarity_search(
        query=question,
        emb_model=emb_model,
        emb_endpoint=emb_endpoint,
        emb_max_tokens=max_tokens,
        vectorstore=vectorstore,
        top_k=top_k,
        rerank=True,
        mode='hybrid',
        reranker_model=reranker_model,
        reranker_endpoint=reranker_endpoint,
        top_r=top_r,
        score_threshold=settings.chatbot.score_threshold,
        track_performance=True,
    )
    
    logger.debug(f"Retrieved documents: {docs}")
    logger.debug(f"Score threshold: {settings.chatbot.score_threshold}")
    logger.info(f"Document search completed, scores: {scores}")
    
    # perf_stat_dict is guaranteed to be a dict when track_performance=True
    return docs, perf_stat_dict or {}
