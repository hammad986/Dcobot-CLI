import os
import json
import logging
import chromadb
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from config import CHROMA_DIR

logger = logging.getLogger(__name__)

class ChromaStore:
    def __init__(self, collection_name: str = "document_chunks"):
        # Ensure directory exists
        if not CHROMA_DIR.exists():
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        self.bm25 = None
        self.bm25_corpus: List[str] = []
        self.bm25_ids: List[str] = []
        
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Rebuilds the BM25 index from ChromaDB documents."""
        try:
            results = self.collection.get(include=['documents', 'metadatas'])
            self.bm25_corpus = results.get('documents', [])
            self.bm25_ids = results.get('ids', [])
            
            if self.bm25_corpus:
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
        except Exception as e:
            logger.error(f"Failed to rebuild BM25: {e}")
            self.bm25 = None

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds chunks, metadata, and embeddings to ChromaDB."""
        if not chunks or len(chunks) == 0:
            return

        ids = []
        documents = []
        metadatas = []
        embeds = []

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            c_idx = metadata.get("chunk_index", 0)
            
            # Chroma requires metadata values to be str, int, float, or bool. 
            # We must serialize lists/dicts (like keywords, entities) to strings.
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    clean_metadata[k] = json.dumps(v)
                else:
                    clean_metadata[k] = v

            chunk_id = f"{doc_id}_{c_idx}"
            ids.append(chunk_id)
            documents.append(chunk.get("text", ""))
            metadatas.append(clean_metadata)
            embeds.append(embeddings[i].tolist())

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeds
            )
            logger.info(f"Added {len(ids)} chunks to ChromaDB.")
            self._rebuild_bm25()
        except Exception as e:
            logger.error(f"Failed to add to ChromaDB: {e}")

    def remove_document(self, filename: str) -> int:
        """Removes a document from ChromaDB by filename and rebuilds BM25."""
        try:
            results = self.collection.get(where={"filename": filename}, include=[])
            ids_to_delete = results.get("ids", [])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for {filename}.")
                self._rebuild_bm25()
                return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Failed to remove document {filename}: {e}")
        return 0

    def clear(self):
        """Deletes all data in the collection and physically deletes storage."""
        try:
            # Delete all documents by fetching all IDs and deleting them
            results = self.collection.get(include=[])
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            self._rebuild_bm25()
            
            # Physically wipe DOCS_DIR
            import shutil
            from config import DOCS_DIR
            if DOCS_DIR.exists():
                shutil.rmtree(DOCS_DIR)
                DOCS_DIR.mkdir(parents=True, exist_ok=True)
                
            logger.info("ChromaDB and storage wiped completely.")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB: {e}")

    def get_indexed_filenames(self) -> List[str]:
        """Returns unique filenames currently indexed."""
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            filenames = set(m.get("filename", "Unknown") for m in metadatas if m)
            return list(filenames)
        except Exception as e:
            logger.error(f"Failed to fetch filenames: {e}")
            return []

    def get_total_chunks(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0
