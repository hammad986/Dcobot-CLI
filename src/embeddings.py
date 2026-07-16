"""
Embeddings Module for Document-QA System
Uses retrieval-optimized BGE/E5 embeddings instead of MiniLM.
"""

import os
import json
import numpy as np
import faiss
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import config
from logger import get_logger


logger = get_logger(__name__)

# Global model cache
_embedding_model = None


class OllamaEmbeddingWrapper:
    """Wrapper to use Ollama for embeddings seamlessly."""
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        
    def encode(self, texts: list, **kwargs):
        import requests
        import numpy as np
        
        try:
            response = requests.post(f"{self.host}/api/embed", json={
                "model": self.model_name,
                "input": texts
            }, timeout=300)
            response.raise_for_status()
            data = response.json()
            return np.array(data["embeddings"], dtype=np.float32)
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise
            
    def get_sentence_embedding_dimension(self) -> int:
        """Dynamically get dimension by running a test embedding."""
        if not hasattr(self, '_dim'):
            test_emb = self.encode(["test"])
            self._dim = test_emb.shape[1]
        return self._dim

def load_embedding_model() -> Any:
    """
    Load retrieval-optimized embedding model (BGE, E5, or Ollama).
    
    Returns:
        SentenceTransformer model or OllamaEmbeddingWrapper
    """
    global _embedding_model
    
    if _embedding_model is None:
        embed_config = config.get_section("embeddings")
        model_name = embed_config.get("model_name", "BAAI/bge-large-en-v1.5")
        device = embed_config.get("device", "cpu")
        
        try:
            if ":" in model_name or "llama" in model_name.lower() or "nomic" in model_name.lower():
                logger.info(f"Loading Ollama embedding model: {model_name}")
                _embedding_model = OllamaEmbeddingWrapper(model_name)
                logger.info("Ollama embedding model loaded successfully")
            else:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {model_name}")
                _embedding_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
                logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, trying fallback: {e}")
            # Fallback to a smaller model
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
                logger.info("Using fallback model: bge-base-en-v1.5")
            except:
                try:
                    from sentence_transformers import SentenceTransformer
                    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
                    logger.warning("Using fallback model: all-MiniLM-L6-v2")
                except Exception as e2:
                    logger.error(f"Failed to load any embedding model: {e2}")
                    return None
    
    return _embedding_model


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings for cosine similarity.
    
    Args:
        vectors: Raw embedding vectors
        
    Returns:
        L2-normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return vectors / norms


def create_embeddings(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Create embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings
    """
    model = load_embedding_model()
    if model is None:
        return None
    
    embed_config = config.get_section("embeddings")
    normalize = embed_config.get("normalize", True)
    
    logger.info(f"Creating embeddings for {len(texts)} texts...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size or embed_config.get("batch_size", 32),
        show_progress_bar=True,
        normalize_embeddings=normalize
    )
    
    if not normalize:
        embeddings = normalize_embeddings(embeddings)
    
    logger.info(f"Embeddings created: {embeddings.shape}")
    return embeddings


class VectorStore:
    """Vector store for retrieval."""
    
    def __init__(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata pickle
        """
        self.index = None
        self.metadata = []
        self.embedding_dimension = None
        
        # Load from config if not provided
        if index_path is None:
            index_path = config.get("retrieval.index_path", "storage/faiss/document_index.faiss")
        if metadata_path is None:
            metadata_path = config.get("retrieval.metadata_path", "storage/faiss/metadata.pkl")
        
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        self._load()
    
    def _load(self) -> bool:
        """Load index and metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("Index or metadata not found")
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = joblib.load(str(self.metadata_path))
            self.embedding_dimension = self.index.d
            logger.info(f"Loaded index with {len(self.metadata)} segments, dimension={self.embedding_dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _check_dimension_compatibility(self) -> bool:
        """Check if the index dimension matches the embedding model."""
        # Get expected dimension from embedding model
        model = load_embedding_model()
        if model is None:
            return False
        
        # Get dimension from model's embedding size
        expected_dim = model.get_sentence_embedding_dimension()
        
        if self.embedding_dimension != expected_dim:
            logger.warning(f"Dimension mismatch: index={self.embedding_dimension}, model={expected_dim}")
            logger.info("Rebuilding index with new embedding dimension...")
            return False
        
        return True
    
    def _rebuild_index(self) -> bool:
        """Rebuild the index with the correct embedding dimension."""
        if not self.metadata:
            logger.error("No metadata available to rebuild index")
            return False

        # Re-embed all existing texts
        texts = [meta.get("text", "") for meta in self.metadata]
        embeddings = create_embeddings(texts)

        if embeddings is None:
            logger.error("Failed to create embeddings for rebuild")
            return False

        # Create new index with correct dimension
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        joblib.dump(self.metadata, str(self.metadata_path))

        self.embedding_dimension = dimension
        logger.info(f"Index rebuilt successfully with dimension={dimension}, metadata entries={len(self.metadata)}")

        return True
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0,
        video_id: Optional[str] = None  # Aliased to document_id by caller
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query:     Query string
            top_k:     Number of results to return
            min_score: Minimum cosine similarity (0.0 = return everything)
            video_id:  Optional document scope filter

        Returns:
            List of matching chunks with scores
        """
        if self.index is None:
            logger.error("Index not loaded")
            return []

        if not self._check_dimension_compatibility():
            logger.warning("Dimension mismatch detected, rebuilding index…")
            if not self._rebuild_index():
                logger.error("Failed to rebuild index")
                return []

        query_embeddings = create_embeddings([query])
        if query_embeddings is None:
            logger.error("Failed to create query embedding")
            return []

        n_total = self.index.ntotal
        if n_total == 0:
            logger.warning("Index is empty — no vectors indexed yet")
            return []

        fetch_k = min(n_total, max(top_k * 10, 100))
        logger.info(f"FAISS search: fetch_k={fetch_k}, total_vectors={n_total}")

        scores, indices = self.index.search(query_embeddings.astype("float32"), fetch_k)

        def _collect(apply_filter: bool) -> List[Dict[str, Any]]:
            out = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if score < min_score:
                    continue
                if idx >= len(self.metadata):
                    continue

                chunk_meta = self.metadata[idx]

                if apply_filter and video_id:
                    # check both document_id and video_id for backwards compat
                    if chunk_meta.get("document_id") != video_id and chunk_meta.get("video_id") != video_id:
                        continue

                item = chunk_meta.copy()
                item["score"] = float(score)
                out.append(item)

                if len(out) >= top_k:
                    break
            return out

        if video_id:
            results = _collect(apply_filter=True)
            if not results:
                logger.warning(
                    f"document_id filter '{video_id}' matched 0 chunks — "
                    "falling back to un-filtered top-k retrieval."
                )
                results = _collect(apply_filter=False)
        else:
            results = _collect(apply_filter=False)

        logger.info(f"FAISS returned {len(results)} chunks for query: {query[:60]!r}")
        return results
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all indexed chunks."""
        return self.metadata

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add new chunks to the index."""
        if not chunks:
            return True
            
        texts = [c.get("text", "") for c in chunks]
        embeddings = create_embeddings(texts)
        if embeddings is None:
            logger.error("Failed to create embeddings for new chunks")
            return False
            
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.embedding_dimension = dimension
            
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(chunks)
        
        # Save
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        joblib.dump(self.metadata, str(self.metadata_path))
        
        logger.info(f"Successfully added {len(chunks)} chunks to index.")
        return True

    def get_indexed_filenames(self) -> List[str]:
        """Return unique document IDs present in the metadata."""
        seen = set()
        names = []
        for meta in self.metadata:
            doc_id = meta.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                names.append(doc_id)
        return names

    def get_total_chunks(self) -> int:
        """Return the total number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

