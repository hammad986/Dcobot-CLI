import json
from typing import List, Dict, Any

class PromptBuilder:
    def __init__(self):
        self.system_prompt = """You are an intelligent document understanding assistant.

You answer questions only from the uploaded document collection.
You may summarize, explain, compare, infer relationships between retrieved facts, combine information from multiple retrieved sections, and produce natural responses.
You must never introduce information that is unsupported by the retrieved context.

If the available evidence is insufficient to answer confidently, reply exactly:
Answer not found in uploaded document.

Formatting Instructions:
- Answer naturally like a professional expert.
- Use headings and bullet lists where appropriate to improve readability.
- Do not expose internal retrieval reasoning, scores, or chunk IDs to the user.
- Automatically detect if the user wants a summary, table, list, or timeline, and format the output appropriately.
"""

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Dynamically combines chunk metadata and exact text into a clean context prompt.
        Includes chat history for conversational awareness.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            
            # Combine dynamically, keeping text exact
            title = meta.get("title", meta.get("filename", "Unknown Document"))
            concepts = meta.get("concepts", "")
            
            chunk_block = f"--- Document: {title} ---\n"
            if concepts:
                chunk_block += f"Concepts: {concepts}\n"
            chunk_block += f"Content:\n{text}\n"
            
            context_parts.append(chunk_block)
            
        context_str = "\n".join(context_parts)
        
        history_str = ""
        if chat_history:
            history_parts = []
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_str = "Conversation History:\n" + "\n".join(history_parts) + "\n\n"

        full_prompt = f"""{self.system_prompt}

{history_str}Retrieved Context:
{context_str}

User Question: {query}
"""
        return full_prompt

    def build_validation_prompt(self, answer: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Builds a prompt for the secondary Response Validation pass.
        """
        context_parts = [c.get("text", "") for c in chunks]
        context_str = "\n\n---\n\n".join(context_parts)
        
        return f"""You are a strict grounding evaluator.
Review the following generated answer against the provided context.

Context:
{context_str}

Generated Answer:
{answer}

Determine if the generated answer contains any claims, facts, or entities NOT strictly supported by the context.
If the answer is fully supported, reply with ONLY the word: APPROVED
If the answer contains unsupported claims, rewrite the answer to remove them. If the entire answer is unsupported, reply exactly: Answer not found in uploaded document.
"""
