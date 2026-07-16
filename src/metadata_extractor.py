import logging
import httpx
import json
from typing import Dict, Any

from config import CHAT_MODEL

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Dynamically analyzes document content using the local LLM to extract generic
    metadata (summary, entities, keywords, concepts) without relying on hardcoded lists.
    """
    def __init__(self, model_name: str = CHAT_MODEL):
        self.model_name = model_name
        self.api_url = "http://127.0.0.1:11434/api/generate"
        self.system_prompt = """You are a document metadata extraction engine. 
Given the document text, output a JSON object with the following generic keys:
{
  "summary": "a short 2 sentence summary of what the document is about",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "entities": ["Named Entity 1", "Named Entity 2"],
  "concepts": ["Concept 1", "Concept 2"],
  "document_type": "predicted type (e.g. resume, contract, technical manual)"
}
Output strictly valid JSON and nothing else. No markdown wrapping.
"""

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Takes raw document text (or a sample of it) and returns a metadata dictionary.
        """
        # Truncate text to avoid blowing up the context window for extraction
        sample_text = text[:4000]
        
        prompt = f"{self.system_prompt}\n\nDocument Text:\n{sample_text}"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        default_meta = {
            "summary": "Unknown",
            "keywords": [],
            "entities": [],
            "concepts": [],
            "document_type": "Unknown"
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                response_text = data.get("response", "").strip()
                
                if response_text:
                    parsed = json.loads(response_text)
                    # Merge with default to ensure keys exist
                    return {**default_meta, **parsed}
                    
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            
        return default_meta
