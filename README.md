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
├── audio/                  # Stored MP3 files
├── transcripts/            # Transcribed text files
├── query_log.json          # Logs of user questions and answers
├── query_stats.png         # Auto-generated analytics image
├── main.py                 # Full RAG pipeline
├── rag_system.py           # Main RAG logic and QA interface
├── transcriber.py          # Whisper transcription
├── summarizer.py           # Gemini summarization
├── downloader.py           # YouTube downloader
├── analytics.py            # Query log visualizer
├── Makefile                # Automation commands
├── requirements.txt
└── README.md
```

---

## 🛠️ Usage

### Run full pipeline:

```bash
make run
```

### Run specific steps:

```bash
make transcribe     # Transcribe audio with Whisper
make summarize      # Summarize text with Gemini
make analyze        # Generate analytics chart
```

---

## 🌐 Web Interface

- A Gradio app will launch
- Ask questions about the speech content
- Session-based `user_id`s are automatically generated
- Logs saved to `query_log.json`
- Analytics available under the 📈 "Stats" tab

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

## 📚 References

- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS (Facebook)](https://github.com/facebookresearch/faiss)
- [Gradio](https://gradio.app)

---

## 🙌 Author

Created by Edmond Huang
For Spring 2025 IOT Course RAG Project
