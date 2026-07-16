"""
Knowledge Structuring Module for Document-QA System
Handles semantic segmentation using embedding similarity.
Splits documents into topic-level chunks (NOT fixed word chunking).
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import config
from logger import get_logger

logger = get_logger(__name__)

from embeddings import load_embedding_model

def get_sentence_embeddings(sentences: List[str]) -> np.ndarray:
    """Get embeddings for a list of sentences."""
    model = load_embedding_model()
    if model is None:
        return None
    
    embeddings = model.encode(sentences, show_progress_bar=False)
    return np.array(embeddings)

def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))

def create_chunk_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a chunk dictionary from segments."""
    return {
        "text": " ".join(s["text"] for s in segments),
        "word_count": len(" ".join(s["text"] for s in segments).split()),
        "document_id": segments[0].get("document_id", ""),
        "page": segments[0].get("page", 0)
    }

def split_large_chunk(
    segments: List[Dict[str, Any]],
    max_words: int
) -> List[Dict[str, Any]]:
    """Split a large chunk into smaller ones."""
    chunks = []
    current_chunk = []
    current_words = 0
    
    for seg in segments:
        seg_words = len(seg["text"].split())
        
        if current_words + seg_words <= max_words:
            current_chunk.append(seg)
            current_words += seg_words
        else:
            if current_chunk:
                chunks.append(create_chunk_from_segments(current_chunk))
            current_chunk = [seg]
            current_words = seg_words
    
    if current_chunk:
        chunks.append(create_chunk_from_segments(current_chunk))
    
    return chunks

def fixed_chunking(
    segments: List[Dict[str, Any]],
    chunk_size_words: int = 150,
    overlap_words: int = 40
) -> List[Dict[str, Any]]:
    """
    Fallback method: create chunks of fixed word count with overlap.
    """
    if not segments:
        return []
    
    logger.info(f"Using fixed chunking ({chunk_size_words} words, {overlap_words} overlap)")
    
    # Extract all words with their metadata
    all_words = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
            
        words = text.split()
        for word in words:
            all_words.append({
                "word": word,
                "document_id": seg.get("document_id", ""),
                "page": seg.get("page", 0)
            })
    
    if not all_words:
        return []
    
    chunks = []
    i = 0
    
    while i < len(all_words):
        # Get slice for current chunk
        end_idx = min(i + chunk_size_words, len(all_words))
        chunk_words = all_words[i:end_idx]
        
        # Calculate metadata
        text = " ".join(w["word"] for w in chunk_words)
        
        chunks.append({
            "text": text,
            "word_count": len(chunk_words),
            "document_id": chunk_words[0].get("document_id", ""),
            "page": chunk_words[0].get("page", 0)
        })
        
        # Move forward, accounting for overlap
        step = chunk_size_words - overlap_words
        if step <= 0:
            step = chunk_size_words // 2  # Fallback if overlap is too large
            
        i += step
    
    logger.info(f"Created {len(chunks)} fixed chunks")
    return chunks

def semantic_segmentation(
    segments: List[Dict[str, Any]],
    similarity_threshold: float = 0.3,
    min_chunk_words: int = 50,
    max_chunk_words: int = 300
) -> List[Dict[str, Any]]:
    """
    Perform semantic segmentation on document text segments.
    
    Uses embedding similarity to detect topic changes:
    - If similarity between consecutive sentences drops below threshold → new chunk
    - This preserves topic coherence vs. arbitrary word count splitting
    
    Args:
        segments: List of text segments/sentences (dicts with 'text')
        similarity_threshold: Threshold for starting new chunk
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        
    Returns:
        List of semantic chunks
    """
    if not segments:
        return []
    
    logger.info("Performing semantic segmentation...")
    
    # Extract sentences from segments
    sentences = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            sentences.append({
                "text": text,
                "document_id": seg.get("document_id", ""),
                "page": seg.get("page", 0)
            })
    
    if not sentences:
        return []
    # Get embeddings
    text_list = [s["text"] for s in sentences]
    embeddings = get_sentence_embeddings(text_list)
    
    if embeddings is None:
        logger.warning("Falling back to fixed chunking")
        return fixed_chunking(segments, max_chunk_words, min_chunk_words)
    
    # Find topic boundaries
    chunk_boundaries = [0]  # Start with first sentence
    
    for i in range(1, len(sentences)):
        similarity = calculate_similarity(embeddings[i-1], embeddings[i])
        
        # Check if similarity drops below threshold
        if similarity < similarity_threshold:
            # Check if current chunk has minimum words
            current_start = chunk_boundaries[-1]
            current_text = " ".join(s["text"] for s in sentences[current_start:i])
            word_count = len(current_text.split())
            
            if word_count >= min_chunk_words:
                chunk_boundaries.append(i)
    
    # Always include the last segment
    if chunk_boundaries[-1] < len(sentences):
        chunk_boundaries.append(len(sentences))
    
    # Create chunks from boundaries
    chunks = []
    for i in range(len(chunk_boundaries) - 1):
        start_idx = chunk_boundaries[i]
        end_idx = chunk_boundaries[i + 1]
        
        chunk_segments = sentences[start_idx:end_idx]
        
        # Calculate chunk metadata
        chunk_text = " ".join(s["text"] for s in chunk_segments)
        word_count = len(chunk_text.split())
        
        # Enforce max chunk size
        if word_count > max_chunk_words:
            # Split into smaller chunks
            sub_chunks = split_large_chunk(chunk_segments, max_chunk_words)
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "text": chunk_text,
                "word_count": word_count,
                "document_id": chunk_segments[0].get("document_id", ""),
                "page": chunk_segments[0].get("page", 0)
            })
    
    logger.info(f"Created {len(chunks)} semantic chunks")
    return chunks

def process_document_to_chunks(
    document_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert document data to semantic chunks.
    
    Args:
        document_data: Dictionary containing 'document_id' and 'segments' 
                       where segments are lists of dicts with 'text' and 'page'
        
    Returns:
        List of chunks
    """
    segments = document_data.get("segments", [])
    document_id = document_data.get("document_id", "unknown_document")
    
    logger.info(f"Processing document: {document_id}")
    logger.info(f"Total segments: {len(segments)}")
    
    # Get chunking settings
    chunking_config = config.get_section("chunking")
    method = chunking_config.get("method", "semantic")
    
    if method == "semantic":
        semantic_config = chunking_config.get("semantic", {})
        chunks = semantic_segmentation(
            segments,
            similarity_threshold=semantic_config.get("similarity_threshold", 0.3),
            min_chunk_words=semantic_config.get("min_chunk_words", 50),
            max_chunk_words=semantic_config.get("max_chunk_words", 300)
        )
    else:
        fixed_config = chunking_config.get("fixed", {})
        chunks = fixed_chunking(
            segments,
            chunk_size_words=fixed_config.get("chunk_size_words", 150),
            overlap_words=fixed_config.get("overlap_words", 40)
        )
    
    # Add metadata
    final_chunks = []
    for i, chunk in enumerate(chunks):
        final_chunks.append({
            "chunk_id": f"{document_id}_{i}",
            "document_id": chunk.get("document_id", document_id),
            "page": chunk.get("page", 0),
            "word_count": chunk.get("word_count", len(chunk["text"].split())),
            "text": chunk["text"]
        })
    
    logger.info(f"Created {len(final_chunks)} chunks for {document_id}")
    return final_chunks
