"""
Retrieval Module for Document-QA System
Handles vector-based retrieval from FAISS index.
"""

from typing import List, Dict, Any, Optional

from config import config
from logger import get_logger
from embeddings import VectorStore, create_embeddings


logger = get_logger(__name__)


class RetrievalSystem:
    """Vector retrieval system."""
    
    def __init__(self):
        """Initialize retrieval system."""
        self.vector_store = None
        self._initialize()
    
    def _initialize(self):
        """Initialize vector store."""
        index_path = config.get("retrieval.index_path", "storage/faiss/document_index.faiss")
        metadata_path = config.get("retrieval.metadata_path", "storage/faiss/metadata.pkl")
        
        self.vector_store = VectorStore(index_path, metadata_path)
        
        if self.vector_store.index is None:
            logger.warning("Vector store not initialized. Please process documents first.")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of results (default from config)
            min_score: Minimum similarity score (default from config)
            document_id: Filter by document ID
            
        Returns:
            List of relevant chunks with scores
        """
        if self.vector_store is None or self.vector_store.index is None:
            logger.error("Vector store not available")
            return []
        
        # Get config values
        if top_k is None:
            top_k = config.get("retrieval.top_k", 5)
        if min_score is None:
            min_score = config.get("retrieval.min_score", 0.0)
        
        logger.info(f"Retrieving top-{top_k} chunks for query: {query[:50]}...")
        
        # Pass document_id instead of video_id if supported by the embeddings module
        results = self.vector_store.search(query, top_k, min_score, video_id=document_id)
        
        logger.info(f"\n[RETRIEVAL DEBUG] Total chunks retrieved: {len(results)}")
        for i, chunk in enumerate(results, 1):
            score = chunk.get('score', 0)
            doc = chunk.get('document_id', chunk.get('video_id', 'Unknown'))
            page = chunk.get('page', 0)
            text = chunk.get('text', '')[:80]
            # normalize document_id
            chunk['document_id'] = doc
            logger.info(
                f"  [Chunk {i}] Score: {score:.4f} | "
                f"Document: {doc} (Page {page}) | "
                f"Text: {text}..."
            )
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        return results
    
    def is_available(self) -> bool:
        """Check if retrieval system is ready."""
        return self.vector_store is not None and self.vector_store.index is not None


def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve chunks.
    
    Args:
        query: User question
        top_k: Number of results
        document_id: Filter by document ID
        
    Returns:
        List of relevant chunks
    """
    retriever = RetrievalSystem()
    return retriever.retrieve(query, top_k, document_id=document_id)
