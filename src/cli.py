"""
Document Intelligence Assistant — Command Line Interface
════════════════════════════════════════════════════════
Conversational AI assistant grounded in uploaded documents.

Intent categories (see intent_router.py):
  command              → slash commands (/add, /files, /clear …)
  general_conversation → greetings, small talk, meta-questions, language requests
  farewell             → bye / exit / quit
  document_question    → answered via the RAG pipeline
  mixed                → RAG answer + optional style note (e.g. "summarise in Hinglish")
"""

import sys
import shutil
import hashlib
import json
from pathlib import Path
from typing import List, Set, Optional
from collections import deque

# ── Force UTF-8 on Windows before any other imports ──────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from logger import get_logger
from config import config
from intent_router import classify, IntentResult

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# STORAGE PATHS  (single source of truth)
# ──────────────────────────────────────────────────────────────
STORAGE_ROOT  = Path("storage")
FAISS_DIR     = STORAGE_ROOT / "faiss"
SUMMARIES_DIR = STORAGE_ROOT / "summaries"
CHROMA_DIR    = STORAGE_ROOT / "chroma"
DOCS_DIR      = STORAGE_ROOT / "docs"
HASHES_FILE   = STORAGE_ROOT / ".indexed_hashes.json"


# ──────────────────────────────────────────────────────────────
# FILE-HASH REGISTRY  (deduplication across restarts)
# ──────────────────────────────────────────────────────────────

def _load_hashes() -> dict:
    if HASHES_FILE.exists():
        try:
            return json.loads(HASHES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_hashes(hashes: dict) -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    HASHES_FILE.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ──────────────────────────────────────────────────────────────

def _print_assistant(text: str) -> None:
    print(f"\nAssistant: {text}\n")


def _format_sources(contexts: list, max_sources: int = 3) -> str:
    seen: Set[str] = set()
    lines = []
    for ctx in contexts[:max_sources]:
        doc_id = ctx.get("document_id", "Unknown")
        page   = ctx.get("page", None)
        if doc_id not in seen:
            seen.add(doc_id)
            label = doc_id
            if page and int(page) > 0:
                label += f" (page {page})"
            lines.append(f"  - {label}")
    if not lines:
        return ""
    return "\nSource:\n" + "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Central owner of the vector store, hash registry, and document stats.
    All add/clear operations go through this class so /files, /stats, and the
    no-documents guard are always perfectly synchronised.
    """

    def __init__(self):
        from document_processor import DocumentProcessor
        from retrieval import RetrievalSystem

        self._hash_registry: dict = _load_hashes()
        self.doc_processor    = DocumentProcessor()
        self.retrieval_system = RetrievalSystem()
        self.doc_processor.processed_hashes = set(self._hash_registry.keys())

    # ── Read-only properties ──────────────────────────────────

    @property
    def is_empty(self) -> bool:
        vs = self.retrieval_system.vector_store
        if vs is None or vs.index is None:
            return True
        return vs.index.ntotal == 0

    def get_indexed_filenames(self) -> List[str]:
        vs = self.retrieval_system.vector_store
        if vs is None:
            return []
        seen: Set[str] = set()
        names = []
        for meta in vs.metadata:
            doc_id = meta.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                names.append(doc_id)
        return names

    def get_total_chunks(self) -> int:
        vs = self.retrieval_system.vector_store
        if vs is None or vs.index is None:
            return 0
        return vs.index.ntotal

    # ── Add documents ─────────────────────────────────────────

    def add_files(self, paths: List[Path]) -> dict:
        results = {"indexed": [], "skipped": [], "failed": []}

        for path in paths:
            if not path.exists():
                results["failed"].append((path.name, "File not found"))
                continue

            fhash = _file_hash(path)
            if fhash in self._hash_registry:
                results["skipped"].append(
                    (path.name, f"Already indexed as '{self._hash_registry[fhash]}'")
                )
                continue

            print(f"  Processing '{path.name}'...")
            try:
                chunks = self.doc_processor.process_file(path)
                if not chunks:
                    results["failed"].append((path.name, "No content extracted"))
                    continue

                vs = self.retrieval_system.vector_store
                if vs is None:
                    results["failed"].append((path.name, "Vector store not available"))
                    continue

                vs.add_chunks(chunks)

                self._hash_registry[fhash] = path.name
                _save_hashes(self._hash_registry)
                self.doc_processor.processed_hashes.add(fhash)

                results["indexed"].append((path.name, len(chunks)))

            except Exception as exc:
                logger.error(f"Failed to index '{path.name}': {exc}", exc_info=True)
                results["failed"].append((path.name, str(exc)))

        return results

    # ── Clear everything ──────────────────────────────────────

    def clear_all(self) -> None:
        """Complete wipe: disk + in-memory state. No restart needed."""
        for d in [FAISS_DIR, CHROMA_DIR, SUMMARIES_DIR, DOCS_DIR]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

        for f in STORAGE_ROOT.glob("*.pkl"):
            f.unlink(missing_ok=True)
        for f in STORAGE_ROOT.glob("*.json"):
            f.unlink(missing_ok=True)

        self._hash_registry.clear()

        # Wipe in-memory vector store BEFORE re-initialising
        if self.retrieval_system.vector_store is not None:
            vs = self.retrieval_system.vector_store
            vs.index = None
            vs.metadata = []
            vs.embedding_dimension = None

        from document_processor import DocumentProcessor
        from retrieval import RetrievalSystem

        self.doc_processor = DocumentProcessor()
        self.retrieval_system = RetrievalSystem()
        self.doc_processor.processed_hashes.clear()

        import embeddings as emb_mod
        emb_mod._embedding_model = None

        logger.info("Knowledge base fully cleared.")


# ──────────────────────────────────────────────────────────────
# SLASH COMMAND HANDLERS
# ──────────────────────────────────────────────────────────────

def _cmd_help() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║          Document Intelligence Assistant             ║
║                  Command Reference                   ║
╚══════════════════════════════════════════════════════╝

  /add <file> [file2 ...]   Upload one or more documents
  /files                    List all indexed documents
  /stats                    Show knowledge base statistics
  /clear                    Wipe ALL data (full reset)
  /debug                    Toggle debug mode
  /help                     Show this help message

Tips:
  • Drag-and-drop a file path into the terminal window
  • Ask anything about your uploaded documents
  • Type 'bye' or 'exit' to leave
""")


def _cmd_files(kb: KnowledgeBase) -> None:
    names = kb.get_indexed_filenames()
    print("\nIndexed Documents:")
    if not names:
        print("  (none — use /add <file> to upload one)")
    else:
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")
    print()


def _cmd_stats(kb: KnowledgeBase) -> None:
    names  = kb.get_indexed_filenames()
    chunks = kb.get_total_chunks()
    print(f"""
Knowledge Base Statistics
  ─────────────────────────────────────
  Documents Indexed : {len(names)}
  Total Chunks      : {chunks}
  Embedding Model   : {config.get('embeddings.model_name', 'N/A')}
  LLM Model         : {config.get('answer.local_model', 'N/A')}
  ─────────────────────────────────────
""")


def _cmd_add(raw_args: str, kb: KnowledgeBase) -> None:
    """Parse /add arguments and index each file."""
    import shlex

    raw_args = raw_args.strip()
    if not raw_args:
        print(
            "\n  Usage: /add <path-to-file> [path2] ...\n"
            "  Example: /add C:\\Users\\you\\report.pdf\n"
        )
        return

    try:
        tokens = shlex.split(raw_args, posix=False)
    except ValueError:
        tokens = raw_args.split()

    paths: List[Path] = []
    for tok in tokens:
        tok = tok.strip().strip('"').strip("'")
        if tok:
            paths.append(Path(tok))

    if not paths:
        print("  No valid file paths provided.\n")
        return

    print()
    results = kb.add_files(paths)

    if results["indexed"]:
        print("  Successfully indexed:")
        for name, n in results["indexed"]:
            print(f"    [+] {name}  ({n} chunks)")
    if results["skipped"]:
        print("  Skipped (already indexed):")
        for name, reason in results["skipped"]:
            print(f"    [=] {name}  — {reason}")
    if results["failed"]:
        print("  Failed:")
        for name, reason in results["failed"]:
            print(f"    [!] {name}  — {reason}")

    print(
        f"\n  Knowledge base: {len(kb.get_indexed_filenames())} document(s), "
        f"{kb.get_total_chunks()} total chunks.\n"
    )


def _cmd_clear(kb: KnowledgeBase, chat_history: deque) -> None:
    print("\n  WARNING: This will permanently delete ALL indexed documents,")
    print("  embeddings, summaries, caches, and conversation history.")
    confirm = input("  Are you sure? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("  Cancelled.\n")
        return

    print("  Clearing...")
    kb.clear_all()
    chat_history.clear()
    print("  Done. Knowledge base has been completely reset.")
    print("  Use /add <file> to upload new documents.\n")


def _cmd_debug() -> None:
    import builtins
    builtins._debug_mode = not getattr(builtins, "_debug_mode", False)
    state = "ON" if builtins._debug_mode else "OFF"
    print(f"  Debug mode: {state}\n")


def _handle_command(raw: str, kb: KnowledgeBase, chat_history: deque) -> None:
    parts = raw.strip().split(None, 1)
    cmd   = parts[0].lower()
    args  = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "/help":  lambda: _cmd_help(),
        "/files": lambda: _cmd_files(kb),
        "/stats": lambda: _cmd_stats(kb),
        "/add":   lambda: _cmd_add(args, kb),
        "/clear": lambda: _cmd_clear(kb, chat_history),
        "/debug": lambda: _cmd_debug(),
    }

    handler = dispatch.get(cmd)
    if handler:
        handler()
    else:
        print(f"  Unknown command '{cmd}'. Type /help to see all commands.\n")


# ──────────────────────────────────────────────────────────────
# GENERAL CONVERSATION HANDLER  (no RAG, no verification)
# ──────────────────────────────────────────────────────────────

def _handle_conversation(intent: IntentResult, query: str, has_docs: bool) -> None:
    from answer_generator import generate_conversational_answer

    if intent.detail == "greeting":
        if has_docs:
            _print_assistant(
                "Hello! 👋 I've read all your uploaded documents and I'm ready to help.\n"
                "Ask me anything — summaries, facts, comparisons, explanations. What would you like to know?"
            )
        else:
            _print_assistant(
                "Hello! 👋 I'm your Document Intelligence Assistant.\n"
                "I don't have any documents yet. Upload one with /add <file> and I'll get started!"
            )
        return

    if intent.detail == "thanks":
        _print_assistant(
            "You're welcome! 😊 Let me know if you'd like me to explain, summarise, "
            "or analyse anything else from your documents."
        )
        return

    # For all other general conversation (meta-questions, language requests, jokes …)
    reply = generate_conversational_answer(query, detail=intent.detail)
    _print_assistant(reply)


# ──────────────────────────────────────────────────────────────
# DOCUMENT QUESTION HANDLER  (full RAG pipeline)
# ──────────────────────────────────────────────────────────────

_NO_DOCS_MSG = (
    "No documents are currently indexed.\n"
    "Use /add <path-to-file> to upload one or more documents."
)

_NOT_FOUND_PHRASES = [
    "not found in documents",
    "question cannot be answered",
    "not found in document",
    "answer not found",
]


def _handle_document_question(
    query: str,
    kb: KnowledgeBase,
    chat_history: deque,
    language_note: Optional[str] = None,
) -> None:
    """Retrieve → rerank → generate → display. Verification is ON for this path."""
    from reranker import rerank_passages
    from answer_generator import AnswerGenerator

    import builtins
    debug = getattr(builtins, "_debug_mode", False)

    if kb.is_empty:
        _print_assistant(_NO_DOCS_MSG)
        return

    if debug:
        print("[DEBUG] Retrieving context...")

    chunks = kb.retrieval_system.retrieve(query)
    if not chunks:
        _print_assistant(
            "I wasn't able to find relevant information in your documents.\n"
            "Try rephrasing, or use /add to upload additional documents."
        )
        return

    if debug:
        print(f"[DEBUG] {len(chunks)} chunks retrieved. Reranking...")

    reranked = rerank_passages(query, chunks)

    if debug:
        print("[DEBUG] Generating answer...")

    generator = AnswerGenerator()
    result    = generator.generate(query, reranked)

    answer       = result.get("answer", "")
    contexts     = result.get("contexts", [])
    verified     = result.get("verified", False)
    verification = result.get("verification", {})

    # Normalise "not found" variants to the canonical exact string
    if any(p in answer.lower() for p in _NOT_FOUND_PHRASES):
        if not answer.startswith("General knowledge"):
            answer = "Answer not found in uploaded document."

    # Append language note for mixed intent
    if language_note:
        answer += f"\n\n_{language_note}_"

    # Update conversation history
    chat_history.append({"role": "user",      "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"\nAssistant:\n{answer}")

    source_block = _format_sources(contexts)
    if source_block:
        print(source_block)

    if verified:
        print("\n✓ Grounded in document evidence")
    elif verification.get("status") == "HALLUCINATION":
        print("\n⚠ Answer could not be verified against documents")

    print()


# ──────────────────────────────────────────────────────────────
# WELCOME & UPLOAD PHASE
# ──────────────────────────────────────────────────────────────

def _print_welcome_banner(has_docs: bool) -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║          Document Intelligence Assistant             ║
║     Ask questions about your uploaded documents      ║
╚══════════════════════════════════════════════════════╝
 Type /help for commands.  Type 'bye' to exit.
""")
    if has_docs:
        print("  Documents already indexed — ready to chat!\n")


def _document_upload_phase(kb: KnowledgeBase) -> None:
    print("""
────────────────────────────────────────────────────────
  Document Upload (optional)
  Enter a file path to index it, or press Enter to skip.
  Supported: PDF · TXT · DOCX · Markdown · CSV
────────────────────────────────────────────────────────
""")
    while True:
        raw = input("  Path (or Enter to skip): ").strip().strip('"').strip("'")
        if not raw:
            break
        _cmd_add(raw, kb)


# ──────────────────────────────────────────────────────────────
# MAIN CHAT LOOP
# ──────────────────────────────────────────────────────────────

def chat_loop(kb: KnowledgeBase) -> None:
    _print_welcome_banner(has_docs=not kb.is_empty)
    chat_history: deque = deque(maxlen=10)

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Strip wrapping quotes added by drag-and-drop on Windows
            if user_input.startswith('"') and user_input.endswith('"'):
                user_input = user_input[1:-1]

            intent = classify(user_input)

            # ── Route by intent ─────────────────────────────
            if intent.category == "farewell":
                _print_assistant("Goodbye! 👋 Take care and come back anytime.")
                break

            elif intent.category == "command":
                _handle_command(user_input, kb, chat_history)

            elif intent.category == "general_conversation":
                _handle_conversation(intent, user_input, has_docs=not kb.is_empty)

            elif intent.category == "document_question":
                _handle_document_question(user_input, kb, chat_history)

            elif intent.category == "mixed":
                # Detect the language request portion for a note
                import re
                lang_match = re.search(
                    r"\bin (hindi|urdu|hinglish|french|spanish|german|arabic|punjabi|bengali|telugu|tamil)\b",
                    user_input.lower()
                )
                lang_note = None
                if lang_match:
                    lang = lang_match.group(1).title()
                    lang_note = f"(Requested in {lang})"
                _handle_document_question(user_input, kb, chat_history, language_note=lang_note)

            else:
                # Fallback — treat as document question
                _handle_document_question(user_input, kb, chat_history)

        except KeyboardInterrupt:
            _print_assistant("Goodbye! 👋 Take care.")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            print("\nAssistant: Something went wrong. Please try again.\n")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Document Intelligence Assistant")
    parser.add_argument("--ingest", type=str, help="Index a document and exit")
    parser.add_argument("--chat",   action="store_true", help="Skip upload phase, go straight to chat")
    args = parser.parse_args()

    kb = KnowledgeBase()

    if args.ingest:
        r = kb.add_files([Path(args.ingest)])
        if r["indexed"]:
            print(f"Indexed '{args.ingest}' successfully.")
        elif r["skipped"]:
            print(f"Skipped '{args.ingest}' (already indexed).")
        else:
            print(f"Failed to index '{args.ingest}'.")
        return

    if args.chat:
        chat_loop(kb)
        return

    _document_upload_phase(kb)
    chat_loop(kb)


if __name__ == "__main__":
    main()
