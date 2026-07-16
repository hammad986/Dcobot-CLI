"""
Query Router for Document-QA System
─────────────────────────────────
Implements a two-path hybrid architecture:

  PATH A  — Summary / overview questions
            Detected by keyword matching → returns stored document summary
            (no retrieval required)

  PATH B  — Factual / evidence questions  (default)
            Standard RAG: retrieval → re-ranking → LLM answer generation

The summary is generated once at processing time and stored as
  data/summaries/<document_id>.json
so repeated overview questions are answered instantly without an LLM call.
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import config
from logger import get_logger

logger = get_logger(__name__)

# Phrases that indicate an LLM returned an error string rather than real content.
# Any response containing these is silently discarded.
_ERROR_INDICATORS = [
    "hf_token missing",
    "please provide your free hugging face token",
    "api_key",
    "hf auth login",
    "you must provide",
    "access denied",
    "unauthorized",
]


def _is_error_response(text: str) -> bool:
    """Return True if *text* looks like an API error rather than real content."""
    low = text.lower()
    return any(phrase in low for phrase in _ERROR_INDICATORS)

# ─────────────────────────────────────────────
# Summary storage helpers
# ─────────────────────────────────────────────
SUMMARY_DIR = Path("storage/summaries")

SUMMARY_KEYWORDS = [
    r"\bsummar(y|ize|ise)\b",
    r"\boverview\b",
    r"\bwhat (is|was) (this |the )?document about\b",
    r"\bwhat did (we|i) learn\b",
    r"\bmain (topic|point|idea|concept|theme)\b",
    r"\bwhat (does|did) (this|the) (document|file|report) (cover|discuss|explain|teach)\b",
    r"\bgist\b",
    r"\bbrief(ly)?\b",
    r"\bkey (point|takeaway|lesson)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in SUMMARY_KEYWORDS]


def is_summary_question(query: str) -> bool:
    """
    Return True if the query is asking for a document summary / overview.

    Uses keyword regex matching — fast, zero LLM call needed.
    """
    for pattern in _COMPILED:
        if pattern.search(query):
            logger.info(f"[Router] Summary intent detected for query: {query!r}")
            return True
    logger.info(f"[Router] Factual intent — routing to RAG for query: {query!r}")
    return False


# ─────────────────────────────────────────────
# Summary persistence
# ─────────────────────────────────────────────

def summary_path(document_id: str) -> Path:
    return SUMMARY_DIR / f"{document_id}.json"


def load_summary(document_id: str) -> Optional[Dict[str, Any]]:
    """Load a previously generated summary for *document_id*.

    Validates that the stored bullets don't contain error strings.
    If corrupt, deletes the file so it regenerates on next processing.
    """
    path = summary_path(document_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ── CORRUPTION GUARD ──────────────────────────────────────────────
        bullets = data.get("bullets", [])
        combined = " ".join(bullets).lower()
        if _is_error_response(combined):
            logger.warning(
                f"Stale/corrupt summary detected for {document_id!r} "
                f"(contains error indicators). Deleting and regenerating."
            )
            path.unlink(missing_ok=True)
            return None

        return data
    except Exception as exc:
        logger.warning(f"Failed to load summary for {document_id}: {exc}")
        return None


def save_summary(document_id: str, summary_data: Dict[str, Any]) -> None:
    """Persist a generated summary to disk."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = summary_path(document_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved: {path.name}")
    except Exception as exc:
        logger.error(f"Failed to save summary: {exc}")


# ─────────────────────────────────────────────
# Summary generation
# ─────────────────────────────────────────────

def generate_document_summary(
    full_text: str,
    document_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate a 5-bullet document summary from the full text.

    Tries the same LLM priority chain as answer generation:
        Ollama → Gemini → OpenAI → HuggingFace

    The result is saved to storage/summaries/<document_id>.json and returned.
    Returns None if all LLMs fail.
    """
    if not full_text or not full_text.strip():
        logger.warning("Empty document — cannot generate summary.")
        return None

    # Truncate very long documents to avoid context-length errors
    max_chars = 15_000
    text = full_text[:max_chars]
    if len(full_text) > max_chars:
        text += "\n[document truncated for brevity]"

    prompt = (
        "You are an expert document summarizer.\n"
        "Read the following document and write exactly 5 concise bullet points "
        "that capture the key topics and ideas covered in this text.\n"
        "Format each bullet starting with '• '.\n"
        "Do NOT include anything outside the 5 bullet points.\n\n"
        f"Document:\n{text}\n\n"
        "5-bullet summary:"
    )

    response: Optional[str] = None

    # 1. Ollama (local)
    try:
        from answer_generator import ask_local_llm
        _r = ask_local_llm(prompt, timeout=120)
        if _r and not _is_error_response(_r):
            response = _r
            logger.info("[Summary] Local LLM generated summary.")
    except Exception:
        pass

    # 2. Gemini
    if not response:
        try:
            from answer_generator import ask_gemini_llm
            raw = ask_gemini_llm(prompt, question=prompt, contexts=[])
            if raw and not _is_error_response(raw):
                response = raw.split("Answer:", 1)[-1].strip() if "Answer:" in raw else raw
                logger.info("[Summary] Gemini generated summary.")
        except Exception:
            pass

    # 3. OpenAI
    if not response:
        try:
            from answer_generator import ask_openai_llm
            _r = ask_openai_llm(prompt)
            if _r and not _is_error_response(_r):
                response = _r
                logger.info("[Summary] OpenAI generated summary.")
        except Exception:
            pass

    # 4. HuggingFace Serverless
    if not response:
        try:
            from answer_generator import ask_huggingface_llm
            raw = ask_huggingface_llm(
                prompt,
                question=prompt,
                contexts=[{"text": text, "document_id": document_id}],
            )
            if raw and not _is_error_response(raw):
                response = raw.split("Answer:", 1)[-1].strip() if "Answer:" in raw else raw
                logger.info("[Summary] HuggingFace Serverless generated summary.")
        except Exception:
            pass

    if not response:
        logger.warning("[Summary] All LLMs failed — no summary generated.")
        return None

    # Parse bullets (keep only lines starting with •, -, *, or digits)
    bullets: List[str] = []
    for line in response.splitlines():
        line = line.strip()
        if line and (
            line.startswith("•")
            or line.startswith("-")
            or line.startswith("*")
            or (len(line) > 2 and line[0].isdigit() and line[1] in ".)")
        ):
            # Normalise to "• ..." format
            clean = re.sub(r"^[\•\-\*\d\.\)]+\s*", "", line).strip()
            if clean:
                bullets.append(f"• {clean}")

    if not bullets:
        # Fallback: treat every non-empty line as a bullet
        bullets = [f"• {l.strip()}" for l in response.splitlines() if l.strip()]

    # Cap at 5 bullets
    bullets = bullets[:5]

    summary_data = {
        "document_id":  document_id,
        "bullets":   bullets,
        "raw":       response,
    }
    save_summary(document_id, summary_data)
    return summary_data
