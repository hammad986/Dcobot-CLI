# Document Intelligence Assistant

> A production-quality, fully local AI assistant that lets you have natural conversations about your documents — no cloud APIs required.

---

## Features

- 🧠 **Intelligent Conversation** — natural chat powered by a local Ollama LLM
- 📄 **Multi-format Support** — PDF, DOCX, TXT, Markdown, CSV
- 🔍 **Semantic Retrieval** — FAISS vector search + BM25 hybrid retrieval
- 🛡️ **Strict Hallucination Prevention** — entity-level grounding and evidence verification
- 🔄 **Incremental Indexing** — add documents at any time without restarting
- ♻️ **Deduplication** — MD5 hash check skips already-indexed files
- 🔁 **Full Reset** — `/clear` wipes everything without restarting the app
- 🌐 **Multi-language** — can respond in Hinglish, Hindi, Urdu, and more (via Ollama)
- 🏠 **Fully Local** — zero data sent to external APIs (by default)
- 🪟 **Windows Compatible** — UTF-8 terminal encoding handled automatically

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────┐
│       Intent Router         │  ← classify() in intent_router.py
│  command / conversation /   │
│  document_question / mixed  │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
 Slash Command    General Conversation
 Handler          (LLM, no retrieval)
       │
       ▼
 Document Question
       │
       ▼
 ┌─────────────────────┐
 │   RetrievalSystem   │  ← retrieval.py
 │  FAISS + BM25       │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │     Reranker        │  ← reranker.py
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Entity Grounding   │  ← answer_generator.py
 │  (pre-validation)   │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Ollama LLM         │  ← qwen2.5-coder:0.5b (default)
 │  (answer gen)       │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │ Evidence Verifier   │  ← evidence_verifier.py (optional)
 └────────┬────────────┘
          │
          ▼
      Final Answer
```

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/document-intelligence-assistant.git
cd document-intelligence-assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com/), then pull the required models:

```bash
# LLM for chat and answer generation
ollama pull phi3:mini

# Embedding model for semantic search
ollama pull nomic-embed-text
```

Start the Ollama server (if it isn't already running):

```bash
ollama serve
```

### 5. Run the assistant

```bash
python src/main.py
```

---

## Usage

### Launch and upload a document

```
python src/main.py
```

When prompted, paste the path to a document (or press Enter to skip and go straight to chat).

### Command-line flags

| Flag | Description |
|------|-------------|
| `--ingest <file>` | Index a file and exit immediately |
| `--chat` | Skip the upload prompt and go directly to chat |

```bash
# Index a file in one command
python src/main.py --ingest C:\Users\you\report.pdf

# Go straight to chat (documents already indexed)
python src/main.py --chat
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/add <file> [file2 ...]` | Upload and index one or more documents |
| `/files` | List all currently indexed documents |
| `/stats` | Show KB stats (docs, chunks, models) |
| `/clear` | Full reset — deletes all indexes and caches |
| `/debug` | Toggle verbose debug output |
| `/help` | Show the command reference |

### Examples

```
You: /add C:\Users\you\resume.pdf
You: /add "D:\My Docs\Q4 Report.pdf" "D:\My Docs\Q3 Report.pdf"
You: /files
You: /stats
You: /clear
```

---

## Conversation Examples

```
You: Hi
Assistant: Hello! 👋 I've read all your uploaded documents and I'm ready to help.

You: Can you speak Hinglish?
Assistant: Bilkul! Main Hinglish mein baat kar sakta hoon. Kya poochna chahte ho?

You: Summarize the document
Assistant: [detailed summary from document]

You: Who is Elon Musk?
Assistant: Answer not found in uploaded document.

You: Summarize the document in Hinglish
Assistant: [summary in Hinglish, (Requested in Hinglish)]

You: Bye
Assistant: Goodbye! 👋 Take care and come back anytime.
```

---

## Supported File Formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word Document | `.docx` |
| Plain Text | `.txt` |
| Markdown | `.md` |
| CSV | `.csv` |

---

## Project Structure

```
document-intelligence-assistant/
├── src/
│   ├── main.py               # Entry point
│   ├── cli.py                # Chat loop and command handlers
│   ├── intent_router.py      # 5-category intent classifier
│   ├── config.py             # Config loader
│   ├── config.yaml           # All configuration
│   ├── document_processor.py # Text extraction (PDF, DOCX, TXT, CSV)
│   ├── knowledge_structuring.py  # Semantic chunking
│   ├── embeddings.py         # FAISS vector store + Ollama embeddings
│   ├── retrieval.py          # Hybrid retrieval (FAISS + BM25)
│   ├── reranker.py           # Cross-encoder / score-based reranking
│   ├── query_router.py       # Summary vs. factual query routing
│   ├── answer_generator.py   # LLM answer generation + conversational mode
│   ├── evidence_verifier.py  # Grounding & hallucination detection
│   └── logger.py             # Centralised logging
├── storage/                  # (gitignored) runtime indexes and caches
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Configuration

All settings live in `src/config.yaml`.

Key settings:

```yaml
answer:
  local_model: "qwen2.5-coder:0.5b"   # Ollama chat model
  hybrid_mode: false                   # true = use general knowledge as fallback

embeddings:
  model_name: "nomic-embed-text:latest"

retrieval:
  top_k: 10

verification:
  enabled: true                        # false = skip evidence verification
```

### Hybrid Mode

By default the assistant only answers from uploaded documents. To allow general-knowledge fallback, set:

```yaml
answer:
  hybrid_mode: true
```

When hybrid mode is active, answers not found in documents will be answered from the LLM's general knowledge and clearly labelled:

```
General knowledge (not from uploaded documents):
...
```

---

## How Retrieval Works

1. **Chunking** — Documents are split into semantically coherent chunks using embedding similarity.
2. **Embedding** — Each chunk is embedded with `nomic-embed-text` via Ollama.
3. **Indexing** — Embeddings are stored in a FAISS flat index; metadata is persisted as a pickle.
4. **Retrieval** — Incoming queries are embedded and matched against FAISS (cosine similarity) + BM25 keyword search. Results are merged and reranked.
5. **Entity Grounding** — Named entities in the question are verified to exist in the retrieved context before generation.
6. **Generation** — The top-ranked passages are passed to the LLM with a strict document-only prompt.
7. **Verification** — The generated answer is optionally verified against the evidence by a second LLM call.

---

## Known Limitations

- **Small embedding model** — `nomic-embed-text` is fast but may miss nuanced semantic matches.
- **Small LLM** — `qwen2.5-coder:0.5b` is very fast but can produce imprecise answers for complex queries. Upgrade to a larger model (e.g. `llama3:8b`) by changing `answer.local_model` in `config.yaml`.
- **No table/image understanding** — Tabular data in PDFs and embedded images are not parsed.
- **Single collection** — All documents share one FAISS index. Per-document search is not yet supported.

---

## Roadmap

- [ ] Per-document query routing
- [ ] Table and structured data extraction
- [ ] Web UI (Gradio / Streamlit)
- [ ] Conversation memory across restarts
- [ ] `/remove <doc>` to remove individual documents without a full clear
- [ ] Support for `.html` and `.pptx` formats

---

## License

MIT — see [LICENSE](LICENSE).
