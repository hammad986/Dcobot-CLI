import logging
import json
import httpx
import numpy as np
from typing import List, Dict, Any, Tuple

from chroma_store import ChromaStore
from embeddings import EmbeddingGenerator
from config import (
    TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL,
    RRF_K, BM25_WEIGHT, DENSE_WEIGHT,
    MMR_LAMBDA, BOOST_WEIGHT, SCORE_HIGH, SCORE_MEDIUM,
    CHAT_MODEL
)

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    7-Stage Hybrid Retrieval Pipeline:
    1. Dense Retrieval
    2. Sparse Retrieval
    3. RRF Fusion
    4. Metadata Boosting
    5. MMR Diversity
    6. Confidence Estimation
    7. Conditional Query Expansion
    """
    def __init__(self, store: ChromaStore, embed_gen: EmbeddingGenerator):
        self.store = store
        self.embed_gen = embed_gen
        self.api_url = "http://127.0.0.1:11434/api/generate"

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def _apply_mmr(self, doc_embeddings: List[np.ndarray], query_embedding: np.ndarray, 
                   doc_dicts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance to maximize diversity."""
        if not doc_embeddings or not doc_dicts:
            return []

        selected_indices = []
        unselected_indices = list(range(len(doc_embeddings)))
        
        # Select first document (most relevant)
        similarities = [self._cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
        first_idx = np.argmax(similarities)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        while len(selected_indices) < min(top_k, len(doc_embeddings)):
            max_mmr = -np.inf
            best_idx = -1
            
            for idx in unselected_indices:
                sim_to_query = similarities[idx]
                
                # Max similarity to already selected docs
                sim_to_selected = max([self._cosine_similarity(doc_embeddings[idx], doc_embeddings[s]) 
                                       for s in selected_indices])
                
                mmr_score = (MMR_LAMBDA * sim_to_query) - ((1 - MMR_LAMBDA) * sim_to_selected)
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)
            else:
                break

        return [doc_dicts[i] for i in selected_indices]

    def _run_pipeline(self, query: str, concepts: List[str]) -> Tuple[List[Dict[str, Any]], float]:
        """Runs stages 1-6 of the pipeline and returns chunks + confidence."""
        if self.store.get_total_chunks() == 0:
            return [], 0.0

        # Stage 1: Dense Retrieval
        query_emb = self.embed_gen.generate_embeddings([query])[0]
        query_emb_np = np.array(query_emb)
        
        dense_results = self.store.collection.query(
            query_embeddings=[query_emb],
            n_results=TOP_K_DENSE,
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Stage 2: Sparse Retrieval
        bm25_scores = []
        if self.store.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.store.bm25.get_scores(tokenized_query)
            
        # Combine everything into a dictionary for RRF tracking
        chunk_map = {} # chunk_id -> dict
        
        # Process Dense
        if dense_results["ids"] and dense_results["ids"][0]:
            for rank, (c_id, doc, meta, emb) in enumerate(zip(
                dense_results["ids"][0],
                dense_results["documents"][0],
                dense_results["metadatas"][0],
                dense_results["embeddings"][0]
            )):
                chunk_map[c_id] = {
                    "id": c_id, "text": doc, "metadata": meta, "embedding": np.array(emb),
                    "dense_rank": rank + 1, "sparse_rank": float('inf'), "score": 0.0
                }

        # Process Sparse
        if len(bm25_scores) > 0:
            top_sparse_indices = np.argsort(bm25_scores)[::-1][:TOP_K_SPARSE]
            for rank, idx in enumerate(top_sparse_indices):
                score = bm25_scores[idx]
                if score <= 0:
                    continue
                c_id = self.store.bm25_ids[idx]
                if c_id not in chunk_map:
                    # Fetch from store if not in dense
                    res = self.store.collection.get(ids=[c_id], include=["documents", "metadatas", "embeddings"])
                    if res["ids"]:
                        chunk_map[c_id] = {
                            "id": c_id, "text": res["documents"][0], "metadata": res["metadatas"][0],
                            "embedding": np.array(res["embeddings"][0]),
                            "dense_rank": float('inf'), "sparse_rank": rank + 1, "score": 0.0
                        }
                else:
                    chunk_map[c_id]["sparse_rank"] = rank + 1

        # Stage 3: RRF & Stage 4: Metadata Boosting
        query_lower = query.lower()
        top_dense_sim = 0.0
        
        # Build boost terms list (user concepts + split query)
        boost_terms = [t.lower() for t in concepts] + [t for t in query_lower.split() if len(t) > 4]
        
        for c_id, data in chunk_map.items():
            rrf_dense = 1.0 / (RRF_K + data["dense_rank"])
            rrf_sparse = 1.0 / (RRF_K + data["sparse_rank"])
            score = (DENSE_WEIGHT * rrf_dense) + (BM25_WEIGHT * rrf_sparse)
            
            # Boost if keyword/concept matches
            meta_str = str(data["metadata"].get("keywords", "")) + str(data["metadata"].get("concepts", ""))
            meta_str = meta_str.lower()
            if any(term in meta_str for term in boost_terms):
                score *= BOOST_WEIGHT
                
            data["score"] = score
            
            # Track max dense sim for confidence
            sim = self._cosine_similarity(query_emb_np, data["embedding"])
            if sim > top_dense_sim:
                top_dense_sim = sim

        # Sort by RRF score
        sorted_chunks = sorted(list(chunk_map.values()), key=lambda x: x["score"], reverse=True)
        top_fusion = sorted_chunks[:TOP_K_FINAL * 2] # take double for MMR

        # Stage 5: MMR Diversity
        if not top_fusion:
            return [], 0.0
            
        doc_embeddings = [c["embedding"] for c in top_fusion]
        final_chunks = self._apply_mmr(doc_embeddings, query_emb_np, top_fusion, TOP_K_FINAL)
        
        # Sort finally by chunk_index to keep reading order natural if they are from the same doc
        final_chunks.sort(key=lambda x: (x["metadata"].get("document_id", ""), x["metadata"].get("chunk_index", 0)))
        
        return final_chunks, top_dense_sim

    def retrieve(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrates retrieval stages based on Query Analyzer output."""
        query = analysis.get("expanded_query", "")
        concepts = analysis.get("concepts", [])
        
        chunks, confidence = self._run_pipeline(query, concepts)

        # Prepare final response
        conf_level = "HIGH" if confidence >= SCORE_HIGH else "MEDIUM" if confidence >= SCORE_MEDIUM else "LOW"
        
        return {
            "chunks": chunks,
            "confidence_level": conf_level,
            "confidence_score": confidence
        }
