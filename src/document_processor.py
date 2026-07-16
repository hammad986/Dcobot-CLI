import csv
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from pypdf import PdfReader
from docx import Document

from logger import get_logger
from knowledge_structuring import process_document_to_chunks
from query_router import generate_document_summary

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.processed_hashes = set()

    def _get_file_hash(self, filepath: Path) -> str:
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {filepath}: {e}")
            return ""

    def process_file(self, filepath: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Process a document: extract text by page, structure into chunks, and generate a summary.
        Returns a list of chunks, or None if failed or duplicate.
        """
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        file_hash = self._get_file_hash(filepath)
        if file_hash in self.processed_hashes:
            logger.info(f"Skipping duplicate file: {filepath.name}")
            return None

        self.processed_hashes.add(file_hash)
        
        document_id = filepath.stem
        segments = self._extract_segments(filepath)
        
        if not segments:
            logger.warning(f"No text extracted from {filepath.name}")
            return None
            
        document_data = {
            "document_id": document_id,
            "segments": segments
        }
        
        # 1. Chunking (Semantic or Fixed)
        chunks = process_document_to_chunks(document_data)
        
        # 2. Summary Generation
        full_text = " ".join(s.get("text", "") for s in segments)
        generate_document_summary(full_text, document_id)
        
        return chunks

    def _extract_segments(self, filepath: Path) -> List[Dict[str, Any]]:
        """Extracts text segments by page/paragraph."""
        suffix = filepath.suffix.lower()
        try:
            if suffix == '.pdf':
                return self._extract_pdf(filepath)
            elif suffix == '.txt' or suffix == '.md':
                return self._extract_txt(filepath)
            elif suffix == '.docx':
                return self._extract_docx(filepath)
            elif suffix == '.csv':
                return self._extract_csv(filepath)
            else:
                logger.error(f"Unsupported file format: {suffix} for {filepath}")
                return []
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            return []

    def _extract_pdf(self, filepath: Path) -> List[Dict[str, Any]]:
        segments = []
        reader = PdfReader(str(filepath))
        document_id = filepath.stem
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Split page into smaller natural segments (e.g. paragraphs or sentences)
                # to allow semantic chunking to work nicely
                import re
                paragraphs = re.split(r'\n\s*\n', page_text.strip())
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        segments.append({
                            "document_id": document_id,
                            "page": i + 1,
                            "text": para
                        })
        return segments

    def _extract_txt(self, filepath: Path) -> List[Dict[str, Any]]:
        segments = []
        document_id = filepath.stem
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            import re
            paragraphs = re.split(r'\n\s*\n', content.strip())
            for para in paragraphs:
                para = para.strip()
                if para:
                    segments.append({
                        "document_id": document_id,
                        "page": 1,
                        "text": para
                    })
        return segments

    def _extract_docx(self, filepath: Path) -> List[Dict[str, Any]]:
        segments = []
        document_id = filepath.stem
        doc = Document(str(filepath))
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                segments.append({
                    "document_id": document_id,
                    "page": 1,
                    "text": text
                })
        return segments

    def _extract_csv(self, filepath: Path) -> List[Dict[str, Any]]:
        segments = []
        document_id = filepath.stem
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                text = " ".join(row).strip()
                if text:
                    segments.append({
                        "document_id": document_id,
                        "page": 1,
                        "text": text
                    })
        return segments
