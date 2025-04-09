# 🧠 RAG-HW: Retrieval-Augmented Generation for Lecture Understanding

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline that:
- Downloads and transcribes YouTube speeches
- Summarizes content using Google Gemini
- Builds a searchable semantic vector database (FAISS)
- Provides an interactive Gradio web interface for user Q&A
- Tracks and visualizes user query history with session IDs

---

## 📦 Features

- 🎙️ Whisper transcription from YouTube
- ✍️ Summarization with Gemini (Vertex AI)
- 🔍 Semantic search with FAISS and LangChain
- 🧠 Gemini-powered conversational QA
- 📈 Built-in analytics dashboard
- 👤 Anonymous user session tracking

---

## 🚀 Getting Started

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd RAG-HW
python -m venv rag-env
source rag-env/bin/activate
```

### 2. Install dependencies

```bash
make setup
# or
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
.
├── Makefile
├── README.md
├── requirements.txt
├── main.py                 # Entry point: initializes RAG and launches the app
├── app_ui.py              # Gradio web interface
├── utils/                 # Supporting modules and helpers
│   ├── analytics.py
│   ├── downloader.py
│   ├── processor.py
│   ├── rag_system.py
│   ├── summarizer.py
│   └── transcriber.py
```

---

## 🔄 System Flow Overview

```text
           +------------------------+
           |  YouTube URL / File    |
           +-----------+------------+
                       |
                       v
        +--------------+---------------+
        | Transcription with Whisper   |
        +--------------+---------------+
                       |
                       v
        +--------------+---------------+
        | Summarization (Gemini)       |
        +--------------+---------------+
                       |
                       v
        +--------------+---------------+
        |  FAISS Vector Store (Search) |
        +--------------+---------------+
                       |
                       v
        +--------------+---------------+
        | Gradio QA Interface (Gemini) |
        +------------------------------+
```

---

## 🌐 Web Interface

- Launches via `app_ui.py` (called from `main.py`)
- Users can upload audio or paste a YouTube URL
- Transcript and summary shown
- Ask questions about the content
- Each user has a generated anonymous ID
- Logs are saved to `query_log.json` for analysis

```bash
python main.py
```

---

## 📊 Analytics Preview

- Line chart: Query frequency over time
- Bar chart: Query count per user
- Histogram: Answer length distribution

---

## 🔐 API Keys Required

Make sure to export your API key:

```bash
export GOOGLE_API_KEY="your_google_generative_ai_key"
```
Note: Be sure to update the `GOOGLE_API_KEY` in `main.py` with your own API key if you're not using environment variables.

---

## 🧼 Code Quality (Pre-commit Hooks)

This project uses [pre-commit](https://pre-commit.com) to automatically format, lint, and clean code before commits.

To set it up:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Optional: to run on all files once
```

The configuration is in `.pre-commit-config.yaml`.

---

## 📚 References

- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS (Facebook)](https://github.com/facebookresearch/faiss)
- [Gradio](https://gradio.app)

---

## 🙌 Author

Created by Edmond Huang
For Spring 2025 IT Course RAG Project
