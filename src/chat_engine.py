import sys
import logging
import httpx
from collections import deque
from typing import Callable, Optional

from query_engine import QueryEngine
from prompt_builder import PromptBuilder
from query_analyzer import QueryAnalyzer
from config import CHAT_MODEL, DEBUG_MODE, MAX_CONVERSATION_TURNS, VALIDATION_MODE, SCORE_HIGH

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self, query_engine: QueryEngine, prompt_builder: PromptBuilder, document_pipeline_callback: Optional[Callable] = None):
        self.query_engine = query_engine
        self.prompt_builder = prompt_builder
        self.query_analyzer = QueryAnalyzer()
        self.document_pipeline_callback = document_pipeline_callback
        self.api_url = "http://127.0.0.1:11434/api/generate"
        
        # Lightweight conversational context
        self.chat_history = deque(maxlen=MAX_CONVERSATION_TURNS)

    def _generate(self, prompt: str) -> str:
        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(self.api_url, json={
                    "model": CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False
                })
                res.raise_for_status()
                return res.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Error: Could not generate response."

    def _validate_response(self, answer: str, chunks: list) -> str:
        """Runs the strict validation prompt to catch hallucinations."""
        if DEBUG_MODE:
            print("[DEBUG] Running Response Validation...")
            
        val_prompt = self.prompt_builder.build_validation_prompt(answer, chunks)
        val_result = self._generate(val_prompt)
        
        if val_result.strip() == "APPROVED":
            return answer
        else:
            if DEBUG_MODE:
                print("[DEBUG] Hallucination caught! Rewriting answer...")
            return val_result.strip()

    def handle_slash_command(self, cmd: str) -> bool:
        cmd = cmd.lower()
        if cmd == "/help":
            print("\nAvailable Commands:")
            print("  /help   - Show this message")
            print("  /files  - List indexed files")
            print("  /stats  - Show index statistics")
            print("  /add    - Add new documents to the index")
            print("  /clear  - Wipe all knowledge base data entirely")
            print("  exit    - Quit the application\n")
            return True
            
        elif cmd == "/files":
            files = self.query_engine.store.get_indexed_filenames()
            print("\nIndexed Files:")
            if not files:
                print("  (No files currently indexed)")
            else:
                for f in files:
                    print(f"  - {f}")
            print()
            return True
            
        elif cmd == "/stats":
            total_chunks = self.query_engine.store.get_total_chunks()
            files = self.query_engine.store.get_indexed_filenames()
            print("\n--- Knowledge Base Stats ---")
            print(f"Files Indexed: {len(files)}")
            print(f"Total Chunks:  {total_chunks}")
            print("----------------------------\n")
            return True
            
        elif cmd == "/add":
            if self.document_pipeline_callback:
                self.document_pipeline_callback()
            else:
                print("Add callback not defined.")
            return True
            
        elif cmd == "/clear":
            print("\nWARNING: This will permanently delete all files, embeddings, metadata, and history.")
            confirm = input("Are you sure? (y/n): ").strip().lower()
            if confirm == 'y':
                self.query_engine.store.clear()
                self.chat_history.clear()
                print("Knowledge base cleared completely.\n")
            else:
                print("Clear aborted.\n")
            return True

        return False

    def chat_loop(self):
        print("--- Chat Phase ---")
        print("Ask questions about your documents, or use slash commands.")
        print("Type '/help' for commands, or 'exit'/'quit' to leave.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                    
                if user_input.startswith('/'):
                    self.handle_slash_command(user_input)
                    continue

                # 1. Analyze Query
                if DEBUG_MODE: print("\n[DEBUG] Analyzing query...")
                analysis = self.query_analyzer.analyze(user_input)
                
                if DEBUG_MODE:
                    import json
                    print(f"[DEBUG] Analysis:\n{json.dumps(analysis, indent=2)}")

                # 2. Conversational bypass
                if analysis.get("is_greeting"):
                    greeting_ans = analysis.get("greeting_response", "Hello! How can I help you with your documents?")
                    print(f"\nAssistant:\n{greeting_ans}\n")
                    self.chat_history.append({"role": "user", "content": user_input})
                    self.chat_history.append({"role": "assistant", "content": greeting_ans})
                    continue

                # 3. Retrieve
                if DEBUG_MODE: print("\n[DEBUG] Retrieving context...")
                retrieval_res = self.query_engine.retrieve(analysis)
                chunks = retrieval_res["chunks"]
                conf_level = retrieval_res["confidence_level"]
                conf_score = retrieval_res["confidence_score"]

                if not chunks:
                    print("\nAssistant: Answer not found in uploaded document.\n")
                    continue

                if DEBUG_MODE:
                    print(f"[DEBUG] Found {len(chunks)} chunks. Confidence: {conf_level} ({conf_score:.2f})")

                # 4. Build Prompt
                prompt = self.prompt_builder.build_prompt(user_input, chunks, list(self.chat_history))
                
                # 5. Generate Answer
                answer = self._generate(prompt)
                
                # 6. Optional Response Validation
                should_validate = False
                if VALIDATION_MODE == "STRICT":
                    should_validate = True
                elif VALIDATION_MODE == "AUTO":
                    if conf_score < SCORE_HIGH or len(answer) > 1000:
                        should_validate = True
                        
                if should_validate:
                    answer = self._validate_response(answer, chunks)

                # 7. Append to history
                self.chat_history.append({"role": "user", "content": user_input})
                self.chat_history.append({"role": "assistant", "content": answer})

                print(f"\nAssistant: {answer}\n")

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print("\nAn error occurred while processing your request. Please try again.\n")
