"""
Answer Generation Module for Document-QA System
Generates answers using a cascading LLM pipeline with graceful fallback.

LLM Priority Order:
  1. Local Ollama (fastest, fully local)
  2. Grok API (xAI)
  3. Gemini API
  4. OpenAI API
  5. HuggingFace Serverless
  6. Extractive Fallback (when all LLMs unavailable but context exists)
"""


import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from config import config
from logger import get_logger

logger = get_logger(__name__)

def rewrite_query(query: str) -> str:
    """
    Query rewriting stub (expand for advanced use).
    For now, just return the query unchanged.
    """
    return query

# ──────────────────────────────────────────────────────────────
# EXTRACTIVE QA (deepset/roberta-base-squad2) — primary HuggingFace path
# Finds exact answer SPANS in the document — fully grounded, no hallucination
# ──────────────────────────────────────────────────────────────
_qa_pipeline = None

def _load_extractive_qa():
    """Load distilbert-base-cased-distilled-squad extractive QA pipeline (cached)."""
    global _qa_pipeline
    if _qa_pipeline is not None:
        return True
    model_name = config.get("answer.hf_model", "distilbert-base-cased-distilled-squad")
    try:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading extractive QA model: {model_name}")
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model=model_name,
            device=-1,   # CPU
        )
        logger.info("Extractive QA model loaded.")
        return True
    except ImportError:
        logger.warning("transformers not installed – extractive QA unavailable.")
        return False
    except Exception as exc:
        logger.warning(f"Failed to load extractive QA model {model_name}: {exc}")
        return False


# ──────────────────────────────────────────────────────────────
# HUGGING FACE SERVERLESS API (InferenceClient)
# Uses the v2 serverless chat API for fast, zero-local-memory QA
# ──────────────────────────────────────────────────────────────
_hf_client = None
_hf_token_used: Optional[str] = None   # tracks which token built the client


def _get_hf_token() -> Optional[str]:
    import os
    _cfg_token = (config.get("answer.hf_token") or "").strip()
    _env_token = (os.environ.get("HF_TOKEN", "") or "").strip()
    token      = _cfg_token or _env_token or None

    logger.info(f"HF token loaded from config: {bool(_cfg_token)}")
    logger.info(f"HF token loaded from env:    {bool(_env_token)}")
    logger.info(f"HF token available:          {bool(token)}")
    return token


def _load_hf_client() -> bool:
    global _hf_client, _hf_token_used

    try:
        from huggingface_hub import InferenceClient

        hf_token = _get_hf_token()

        if _hf_client is not None and _hf_token_used == hf_token:
            return True

        if not hf_token:
            logger.warning("No HF_TOKEN found")

        _hf_client     = InferenceClient(token=hf_token)
        _hf_token_used = hf_token
        logger.info("HuggingFace InferenceClient (re)initialized (token present: %s).", bool(hf_token))
        return True

    except ImportError:
        logger.warning("huggingface_hub not installed.")
        return False
    except Exception as exc:
        logger.warning(f"Failed to init HF InferenceClient: {exc}")
        return False

def build_answer_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build a conversational, natural-language prompt for the assistant.
    """
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        doc_id = ctx.get("document_id", "Unknown")
        page   = ctx.get("page", 0)
        text   = ctx.get("text", "")
        context_text += f"[Source {i}: {doc_id}, page {page}]\n{text}\n\n"

    return f"""You are a professional Document Intelligence Assistant — warm, clear, and conversational.

Your personality:
- You have carefully read all the uploaded documents.
- You speak naturally, like a knowledgeable human research assistant.
- You never sound robotic or like a database.
- You are concise, precise, and easy to understand.

Your rules:
- Answer ONLY using the document excerpts provided below.
- Do NOT introduce facts, names, or figures that do not appear in the excerpts.
- If the information genuinely does not exist in the excerpts, say: NOT FOUND IN DOCUMENTS
- Never begin your answer with "According to the document..." unless absolutely necessary.
- Use markdown formatting (headings, bullets, bold) when it improves clarity.
- Keep answers concise and useful.

Document Excerpts:
{context_text}
Question: {query}

Answer:"""




def build_semantic_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Second-pass semantic prompt — used when the first pass returns NOT FOUND.
    """
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        doc_id = ctx.get("document_id", "Unknown")
        page   = ctx.get("page", 0)
        text   = ctx.get("text", "")
        context_text += f"[Source {i}: {doc_id}, page {page}]\n{text}\n\n"

    return f"""You are a professional Document Intelligence Assistant.

Using ONLY the document excerpts below, answer the question as naturally and helpfully as possible.
You may infer, summarize, and explain — but only from what is present in the excerpts.
If there is truly no relevant information, reply exactly: NOT FOUND IN DOCUMENTS

Document Excerpts:
{context_text}
Question: {query}

Answer:"""




def ask_local_llm(prompt: str, timeout: Optional[int] = None) -> Optional[str]:
    """Ask local Ollama LLM."""
    try:
        response = requests.post(
            config.get("answer.ollama_url", "http://localhost:11434/api/generate"),
            json={
                "model": config.get("answer.local_model", "phi3:mini"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 800,
                    "temperature": 0.1,
                }
            },
            timeout=timeout or config.get("answer.timeout", 180)
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama – it may not be running.")
        return None
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return None


def ask_huggingface_llm(prompt: str, question: str = "", contexts: Optional[List[Dict]] = None) -> Optional[str]:
    """Use Hugging Face Serverless Inference API (Chat Completion v2) to answer."""
    if not _load_hf_client():
        return None

    try:
        if not contexts:
            return None

        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "HuggingFaceH4/zephyr-7b-beta",
        ]

        primary_model = config.get("answer.hf_model", models[0])
        if primary_model not in models:
            models.insert(0, primary_model)

        q = question or prompt

        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            text  = ctx.get("text", "")
            doc_id = ctx.get("document_id", "Unknown")
            context_text += f"[{doc_id}] {text}\n"

        sys_prompt = (
            "You are a helpful document assistant. "
            "Answer the user's question based ONLY on the provided excerpt. "
            "If the information is not in the excerpt at all, reply exactly: NOT FOUND IN DOCUMENTS."
        )
        user_msg = f"Excerpt:\n{context_text}\n\nQuestion: {q}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ]

        for model in models:
            logger.info(f"Trying HF Serverless API with model: {model}")
            try:
                response = _hf_client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1,
                )

                answer = response.choices[0].message.content.strip()
                logger.info(f"HF API Answer ({model}): {answer!r}")

                if answer and "not found in document" not in answer.lower():
                    return f"Answer: {answer}"
                elif "not found in document" in answer.lower():
                    logger.info(f"Model {model}: answer not in context.")
                    return "Answer: Question cannot be answered from these documents."

            except ValueError as ve:
                err = str(ve).lower()
                if any(k in err for k in ("api_key", "hf auth login", "401", "unauthorized", "invalid username")):
                    logger.warning("HF Serverless API authentication failed.")
                    break
                continue
            except Exception as e:
                err = str(e).lower()
                if "401" in err or "unauthorized" in err or "invalid username" in err:
                    break
                continue
                
        return None
        
    except Exception as exc:
        logger.error(f"HF Serverless execution failed: {exc}")
        return None


def ask_gemini_llm(prompt: str, question: str = "", contexts: Optional[List[Dict]] = None) -> Optional[str]:
    """Use Google Gemini API as secondary LLM fallback."""
    try:
        import google.generativeai as genai
    except ImportError:
        return None

    api_key = config.get("answer.gemini_api_key", "")
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            config.get("answer.gemini_model", "gemini-1.5-flash")
        )

        q = question or prompt
        context_text = ""
        if contexts:
            for i, ctx in enumerate(contexts, 1):
                text  = ctx.get("text", "")
                doc_id = ctx.get("document_id", "Unknown")
                context_text += f"[{doc_id}] {text}\n"

        full_prompt = (
            "You are a DOCUMENT QUESTION ANSWERING SYSTEM.\n"
            "Rules:\n"
            "* Answer ONLY from the provided excerpt.\n"
            "* Do NOT use outside knowledge.\n"
            "* If the answer is not in the excerpt, reply: Question cannot be answered from these documents.\n\n"
            f"Excerpt:\n{context_text}\n"
            f"Question: {q}\n"
            "Answer:"
        )

        response = model.generate_content(full_prompt)
        answer   = response.text.strip() if response.text else None

        if answer:
            logger.info(f"Gemini answer: {answer!r}")
            return f"Answer: {answer}"

    except Exception as exc:
        logger.warning(f"Gemini API failed: {exc}")

    return None


def ask_openai_llm(prompt: str, timeout: Optional[int] = None) -> Optional[str]:
    """Ask OpenAI API."""
    try:
        import openai

        api_key = config.get("answer.openai_api_key", "")
        if not api_key:
            return None

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=config.get("answer.openai_model", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a document question answering system."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            timeout=timeout or config.get("answer.timeout", 180),
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        return None
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return None


def ask_grok_llm(
    prompt: str,
    question: str = "",
    contexts: Optional[List[Dict]] = None,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """Ask xAI Grok API."""
    api_key = config.get("answer.grok_api_key", "")
    if not api_key:
        return None

    try:
        import openai
    except ImportError:
        return None

    try:
        grok_url = config.get("answer.grok_url", "https://api.x.ai/v1")
        model    = config.get("answer.grok_model", "grok-beta")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=grok_url,
        )

        q = question or prompt
        context_text = ""
        if contexts:
            for ctx in contexts:
                text  = ctx.get("text",  "")
                doc_id = ctx.get("document_id", "Unknown")
                context_text += f"[{doc_id}] {text}\n"

        full_prompt = (
            "You are a DOCUMENT QUESTION ANSWERING SYSTEM.\n"
            "Rules:\n"
            "* Answer ONLY from the provided excerpt.\n"
            "* Do NOT use outside knowledge.\n"
            "* If the answer is not in the excerpt, reply: Question cannot be answered from these documents.\n\n"
            f"Excerpt:\n{context_text}\n"
            f"Question: {q}\n"
            "Answer:"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            temperature=0,
            max_tokens=256,
            timeout=timeout or config.get("answer.timeout", 60),
        )

        answer = response.choices[0].message.content.strip()
        if answer:
            logger.info(f"Grok answer: {answer!r}")
            return f"Answer: {answer}"

    except Exception as exc:
        logger.warning(f"Grok API failed: {exc}")

    return None


def parse_answer(response: str) -> Optional[str]:
    """Parse LLM response. Handles both 'Answer: ...' prefix and freeform natural language."""
    if not response:
        return None

    # Strip leading/trailing whitespace
    response = response.strip()

    # If response starts with "Answer:", strip the prefix (legacy prompt compat)
    if response.lower().startswith("answer:"):
        answer = response.split(":", 1)[1].strip()
        if answer:
            return answer

    # Otherwise return the full response as-is (new conversational prompt)
    return response


def _is_not_found(answer: Optional[str]) -> bool:
    if not answer:
        return True
    low = answer.lower().strip()
    return (
        low == "not found in documents"
        or low == "question cannot be answered from these documents"
        or low.startswith("not found in documents.")
        or low == "not found in document"
    )


def _extractive_fallback(
    query: str,
    contexts: List[Dict[str, Any]],
) -> Optional[str]:
    if not contexts:
        return None

    import re

    top_ctx = contexts[0]
    text    = (top_ctx.get("text") or "").strip()

    if not text:
        return None

    sentences = re.split(r'(?<=[.!?])\s+', text)
    excerpt   = " ".join(sentences[:2]).strip()

    if not excerpt:
        return None

    logger.warning("[EXTRACTIVE FALLBACK] Using top context chunk.")
    return f"Answer: {excerpt}"


def _run_llm_cascade(
    prompt: str,
    query: str,
    contexts: List[Dict[str, Any]],
    answer_config: Dict[str, Any],
    use_extractive_fallback: bool = True,
) -> Optional[str]:
    response = None

    if answer_config.get("use_local", True):
        response = ask_local_llm(prompt)

    if response is None:
        response = ask_grok_llm(prompt, question=query, contexts=contexts)

    if response is None:
        response = ask_gemini_llm(prompt, question=query, contexts=contexts)

    if response is None and answer_config.get("use_openai", False):
        response = ask_openai_llm(prompt)

    if response is None and answer_config.get("use_huggingface", True):
        response = ask_huggingface_llm(prompt, question=query, contexts=contexts)

    if response is None and use_extractive_fallback and contexts:
        response = _extractive_fallback(query, contexts)

    return response


def _pre_generation_entity_check(query: str, contexts: List[Dict[str, Any]]) -> bool:
    """
    Verify whether the primary entities from the user's question exist in the retrieved evidence.
    If they do not exist, we should not attempt to generate an answer.
    """
    import re
    # Common stop words that might be capitalized at the start of a sentence
    stop_words = {
        "what", "who", "where", "when", "how", "why", "is", "are", "do", "does", "did",
        "can", "could", "would", "should", "the", "a", "an", "this", "that", "in", "on",
        "at", "to", "for", "with", "by", "about", "as", "of", "and", "or", "which", "give", "tell", "show"
    }
    
    # Extract multi-word capitalized phrases (e.g. "Elon Musk", "Machine Learning")
    multi_word = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b', query)
    
    # Extract single capitalized words
    single_word = re.findall(r'\b[A-Z][a-zA-Z]*\b', query)
    valid_singles = [w for w in single_word if w.lower() not in stop_words]
    
    # We will check all extracted valid named entities.
    entities_to_check = set(multi_word + valid_singles)
    
    if not entities_to_check:
        return True # No obvious entities to verify
        
    combined_context = " ".join(c.get("text", "") for c in contexts).lower()
    
    for ent in entities_to_check:
        if ent.lower() not in combined_context:
            logger.warning(f"[PRE-VALIDATION] Entity '{ent}' not found in retrieved context.")
            return False
            
    return True

def ask_general_knowledge_llm(query: str, answer_config: Dict[str, Any]) -> Optional[str]:
    """Ask LLM using general knowledge (no context)."""
    prompt = f"""You are a helpful assistant.
Please answer the following question using your general knowledge.

Question: {query}

Answer:"""
    
    if answer_config.get("use_local", True):
        return ask_local_llm(prompt)
    elif answer_config.get("use_grok", False):
        return ask_grok_llm(prompt, question=query)
    elif answer_config.get("use_openai", False):
        return ask_openai_llm(prompt)
    else:
        return ask_huggingface_llm(prompt, question=query, contexts=[{"text": "General knowledge prompt", "document_id": "General"}])

def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    use_verification: bool = False
) -> Dict[str, Any]:
    logger.info("="*50)
    logger.info("ANSWER GENERATION")
    logger.info("="*50)

    answer_config = config.get_section("answer")
    hybrid_mode = answer_config.get("hybrid_mode", False)
    
    # Result template for "not found"
    not_found_result = {
        "answer":    "Answer not found in uploaded document.",
        "contexts":  contexts,
        "verified":  False,
        "raw_response": None
    }

    if not contexts:
        if hybrid_mode:
            gk_answer = ask_general_knowledge_llm(query, answer_config)
            if gk_answer:
                return {
                    "answer": f"General knowledge (not from uploaded documents):\n\n{gk_answer}",
                    "contexts": [],
                    "verified": True,
                    "raw_response": gk_answer
                }
        not_found_result["contexts"] = []
        return not_found_result

    # 1. Pre-generation validation (Entity Grounding)
    if not _pre_generation_entity_check(query, contexts):
        if hybrid_mode:
            gk_answer = ask_general_knowledge_llm(query, answer_config)
            if gk_answer:
                return {
                    "answer": f"General knowledge (not from uploaded documents):\n\n{gk_answer}",
                    "contexts": [],
                    "verified": True,
                    "raw_response": gk_answer
                }
        return not_found_result

    rewritten_query = rewrite_query(query)

    logger.info("[PASS 1] Strict extractive mode…")
    strict_prompt = build_answer_prompt(rewritten_query, contexts)
    response = _run_llm_cascade(strict_prompt, rewritten_query, contexts, answer_config)

    answer = parse_answer(response) if response else None

    if _is_not_found(answer) and contexts:
        logger.info("[PASS 2] Strict pass returned NOT FOUND — trying SEMANTIC context mode.")
        semantic_prompt = build_semantic_prompt(query, contexts)
        sem_response    = _run_llm_cascade(
            semantic_prompt, query, contexts, answer_config,
            use_extractive_fallback=False,
        )
        sem_answer = parse_answer(sem_response) if sem_response else None

        if sem_answer and not _is_not_found(sem_answer):
            logger.info("[PASS 2] Semantic context answer ACCEPTED.")
            answer    = sem_answer
            response  = sem_response
        else:
            if contexts:
                fb_response = _extractive_fallback(query, contexts)
                if fb_response:
                    fb_answer = parse_answer(fb_response)
                    if fb_answer and not _is_not_found(fb_answer):
                        answer    = fb_answer
                        response  = fb_response

    if _is_not_found(answer):
        if hybrid_mode:
            gk_answer = ask_general_knowledge_llm(query, answer_config)
            if gk_answer:
                return {
                    "answer": f"General knowledge (not from uploaded documents):\n\n{gk_answer}",
                    "contexts": [],
                    "verified": True,
                    "raw_response": gk_answer
                }
        return not_found_result

    result = {
        "answer":       answer,
        "contexts":     contexts,
        "raw_response": response,
        "verified":     False,
    }

    if use_verification and config.get("verification.enabled", True):
        try:
            from evidence_verifier import verify_answer as _verify
            verification_result = _verify(answer, contexts, query)
            is_valid            = verification_result.get("is_valid", False)
            status              = verification_result.get("status", "UNKNOWN")
            
            result["verification"] = verification_result
            result["trust_score"]  = verification_result.get("trust_score", 0)

            if not is_valid:
                logger.warning(f"[VERIFICATION] Answer REJECTED by verification (status={status})")
                if hybrid_mode:
                    gk_answer = ask_general_knowledge_llm(query, answer_config)
                    if gk_answer:
                        result["answer"] = f"General knowledge (not from uploaded documents):\n\n{gk_answer}"
                        result["contexts"] = []
                        result["verified"] = True
                        result["raw_response"] = gk_answer
                        return result
                
                result["answer"] = "Answer not found in uploaded document."
                result["verified"] = False
            else:
                result["verified"] = True
        except ImportError:
            logger.warning("evidence_verifier module not available")

    return result



class AnswerGenerator:
    """Answer generation wrapper."""
    
    def __init__(self):
        self.use_verification = config.get("verification.enabled", True)
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return generate_answer(query, contexts, self.use_verification)


# ──────────────────────────────────────────────────────────────
# CONVERSATIONAL ANSWER (no RAG, no verification)
# Used for greetings, meta-questions, small talk, language requests
# ──────────────────────────────────────────────────────────────

_CANNED_FALLBACKS = {
    "greeting": "Hello! 👋 How can I help you today?",
    "thanks":   "You're welcome! 😊 Is there anything else I can help with?",
    "farewell": "Take care! See you next time. 👋",
    "default":  (
        "I'm your Document Intelligence Assistant. "
        "Upload a document with /add and I'll answer questions about it. "
        "Type /help to see all available commands."
    ),
}

_CONV_SYSTEM_PROMPT = """\
You are a friendly, warm, and witty AI assistant called DocuBot.
You are helpful, concise, and conversational — like a knowledgeable human colleague.
You can speak multiple languages including Hinglish (a blend of Hindi and English).
If the user asks for Hinglish, respond naturally in Hinglish.
Keep responses short unless the user asks for detail.
Never mention that you are a large language model or reference your training data.
"""


def generate_conversational_answer(query: str, detail: Optional[str] = None) -> str:
    """
    Answer a general conversation message using the LLM.
    No context, no retrieval, no verification.
    Falls back to a short canned reply if the LLM is unavailable.
    """
    # Try Ollama first
    prompt = f"{_CONV_SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
    try:
        response = ask_local_llm(prompt, timeout=30)
        if response and len(response.strip()) > 2:
            return response.strip()
    except Exception:
        pass

    # Canned fallback based on detail hint
    if detail == "greeting":
        return _CANNED_FALLBACKS["greeting"]
    if detail == "thanks":
        return _CANNED_FALLBACKS["thanks"]
    if detail == "farewell":
        return _CANNED_FALLBACKS["farewell"]
    return _CANNED_FALLBACKS["default"]
