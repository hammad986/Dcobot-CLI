"""
Intent Router for Document Intelligence Assistant
──────────────────────────────────────────────────
Classifies every user message into exactly one of:

  command              → /add, /files, /clear, /stats, /debug, /help
  general_conversation → greetings, meta-questions, small talk, language requests
  farewell             → bye, exit, quit, goodbye
  document_question    → questions answerable from uploaded documents
  mixed                → combination (e.g. "summarise in Hinglish")

Only document_question and mixed go through the RAG pipeline.
General conversation is answered directly by the LLM — never verified.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class IntentResult:
    category: str          # one of the 5 categories above
    confidence: str        # "high" | "medium" | "low"
    detail: Optional[str] = None   # optional sub-category hint


# ──────────────────────────────────────────────────────────────
# Static signal dictionaries
# ──────────────────────────────────────────────────────────────

_FAREWELL_TOKENS = {
    "bye", "goodbye", "good bye", "see you", "see ya", "take care",
    "cya", "later", "farewell", "quit", "exit", "ttyl", "adios",
}

_GREETING_TOKENS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "yo", "sup",
    "good morning", "good afternoon", "good evening", "what's up",
    "assalam", "salam", "namaste", "helo",
}

_THANKS_TOKENS = {
    "thanks", "thank you", "thank you so much", "ty", "thx",
    "appreciate it", "appreciated", "shukriya", "shukria",
}

# Phrases that are definitely general conversation — not doc questions
_GENERAL_CONV_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat (are|can) you( do)?\b",
    r"\bwhat is your name\b",
    r"\byour name\b",
    r"\bintroduce yourself\b",
    r"\btell me about yourself\b",
    r"\bhow are you\b",
    r"\bhow do you (do|feel|work)\b",
    r"\bare you (an? )?(ai|bot|assistant|robot|human)\b",
    r"\bcan you (speak|talk|write|respond) in\b",
    r"\bspeak (in |to me in )?\w+(\s+\w+)?\b",  # "speak Hinglish", "speak Hindi"
    r"\btell me a joke\b",
    r"\bjoke\b",
    r"\bfunny\b",
    r"\bhi+\b",
    r"\byou (are|re) (awesome|great|amazing|helpful)\b",
    r"\bwhat language\b",
    r"\bhinglish\b",
    r"\bhindi\b",
    r"\burdu\b",
    r"\bcan you help me\b",
    r"\bhelp me understand\b",
    r"\bexplain yourself\b",
    r"\bwhat can i ask\b",
    r"\bwhat (do|should) i (ask|say)\b",
]

# Strong signals that a message is about uploaded documents
_DOC_QUESTION_PATTERNS = [
    r"\b(document|file|upload(ed)?|pdf|resume|cv|report|paper)\b",
    r"\bsummar(y|ize|ise|ization)\b",
    r"\bwhat (is|are|was|were) (mentioned|discussed|covered|listed|described|in)\b",
    r"\bwho (is|are|was|were) mentioned\b",
    r"\blist (all|the|any)?\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\baccording to\b",
    r"\bfrom the\b",
    r"\bin the document\b",
    r"\bthe (resume|cv|file|report|paper)\b",
    r"\bcertif(icate|ication|ied)\b",
    r"\bskill(s)?\b",
    r"\bexperience\b",
    r"\bqualification\b",
    r"\beducation\b",
    r"\bproject(s)?\b",
    r"\btechnolog(y|ies)\b",
]


# Language modifier pattern (checked before general conv patterns)
_LANGUAGE_MODIFIER_RE = re.compile(
    r"\bin (hindi|urdu|hinglish|french|spanish|german|arabic|punjabi|bengali|telugu|tamil)\b"
)

# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def classify(text: str) -> IntentResult:
    """
    Classify a user message into one of 5 intent categories.
    Returns an IntentResult with category and confidence.
    """
    stripped = text.strip()
    low      = stripped.lower()
    norm     = re.sub(r"[!.,?;:]+$", "", low).strip()

    # 1. Slash commands — highest priority
    if stripped.startswith("/"):
        return IntentResult(category="command", confidence="high")

    # 2. Farewells
    if norm in _FAREWELL_TOKENS or any(norm.startswith(f) for f in _FAREWELL_TOKENS):
        return IntentResult(category="farewell", confidence="high")

    # 3. Greetings (standalone)
    if norm in _GREETING_TOKENS or any(norm == g for g in _GREETING_TOKENS):
        return IntentResult(category="general_conversation", confidence="high", detail="greeting")

    # 4. Thanks
    if norm in _THANKS_TOKENS or any(norm.startswith(t) for t in _THANKS_TOKENS):
        return IntentResult(category="general_conversation", confidence="high", detail="thanks")

    # 5. Detect language modifier early — used to distinguish mixed intent
    has_language_modifier = bool(_LANGUAGE_MODIFIER_RE.search(low))

    # 6. Score document signals
    doc_score = sum(1 for pat in _DOC_QUESTION_PATTERNS if re.search(pat, low))

    # If there are doc signals + a language modifier → mixed
    if doc_score >= 1 and has_language_modifier:
        return IntentResult(category="mixed", confidence="high", detail="language_modifier")

    # If only a language modifier and no doc signals, check if the verb is doc-like
    # e.g. "Summarize in Hinglish" — 'summarize' is a doc verb
    if has_language_modifier:
        doc_verbs = re.compile(
            r"\b(summarize|summarise|summarize|explain|describe|translate|list|show|tell me about)\b"
        )
        if doc_verbs.search(low):
            return IntentResult(category="mixed", confidence="high", detail="language_modifier")

    # 7. General conversation patterns (checked after doc score so we don't
    #    accidentally classify doc questions as conversation)
    if doc_score == 0:
        for pat in _GENERAL_CONV_PATTERNS:
            if re.search(pat, low):
                return IntentResult(category="general_conversation", confidence="high", detail="meta")

    if doc_score >= 1:
        return IntentResult(category="document_question", confidence="high")

    # 8. Short messages with no doc signals → general conversation
    word_count = len(stripped.split())
    if word_count <= 4:
        return IntentResult(category="general_conversation", confidence="medium", detail="short_message")

    # 9. Default: treat as document question
    return IntentResult(category="document_question", confidence="low")
