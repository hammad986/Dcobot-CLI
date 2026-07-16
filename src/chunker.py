import re
import logging
from typing import List, Dict, Any
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, filepath: Path, document_id: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits text into chunks by paragraphs, falling back to sentences/characters if too long.
        Preserves metadata.
        """
        if not text.strip():
            return []

        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        raw_chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If a single paragraph is too large, split it further by sentence
            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?]) +', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                        raw_chunks.append(current_chunk.strip())
                        # Keep overlap
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:] + " " + sentence
                    else:
                        current_chunk += " " + sentence
            else:
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    raw_chunks.append(current_chunk.strip())
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            raw_chunks.append(current_chunk.strip())

        total_chunks = len(raw_chunks)
        chunks_with_metadata = []

        for i, chunk_text in enumerate(raw_chunks):
            # Base metadata
            metadata = {
                "filename": filepath.name,
                "absolute_file_path": str(filepath.resolve()),
                "document_id": document_id,
                "chunk_id": f"{document_id}_{i}",
                "chunk_index": i,
                "total_chunks": total_chunks,
                "file_type": filepath.suffix.lower()
            }
            
            # Merge with extracted generic document metadata
            if doc_metadata:
                metadata.update(doc_metadata)

            chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": metadata
            })

        logger.info(f"Chunked {filepath.name} into {total_chunks} chunks.")
        return chunks_with_metadata
