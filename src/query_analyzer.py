import httpx
import json
import logging
from typing import Dict, Any

from config import CHAT_MODEL

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self, model_name: str = CHAT_MODEL):
        self.model_name = model_name
        self.api_url = "http://127.0.0.1:11434/api/generate"
        self.system_prompt = """You are a Query Analyzer for a Document Intelligence Platform.
Your goal is to parse the user's input and output a strict JSON object.

If the user is just saying a conversational greeting (e.g., "Hi", "Hello", "How are you?"), output:
{
  "is_greeting": true,
  "greeting_response": "A polite, friendly response as an AI assistant ready to help with their documents.",
  "intent": "greeting",
  "entities": [],
  "concepts": []
}

If the user is asking a question about their documents, output:
{
  "is_greeting": false,
  "intent": "Determine intent: e.g. overview, factual, comparison, summary",
  "entities": ["Extract named entities like people, organizations, locations"],
  "concepts": ["Extract core concepts, semantic synonyms, and aliases"],
  "expanded_query": "A rewritten version of the query that includes synonyms for better vector search"
}

Ensure the output is ONLY valid JSON.
"""

    def analyze(self, query: str) -> Dict[str, Any]:
        prompt = f"{self.system_prompt}\n\nUser Input: {query}"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        default_meta = {
            "is_greeting": False,
            "intent": "unknown",
            "entities": [],
            "concepts": [],
            "expanded_query": query
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(self.api_url, json=payload)
                res.raise_for_status()
                response_text = res.json().get("response", "").strip()
                
                if response_text:
                    parsed = json.loads(response_text)
                    return {**default_meta, **parsed}
                    
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            
        return default_meta
