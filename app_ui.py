import datetime
import json
import os
import random

import gradio as gr

from utils.downloader import AudioDownloader
from utils.summarizer import summarize_with_palm
from utils.transcriber import Transcriber


def launch_gradio_app(rag):
    query_log = []
    active_user_ids = set()

    def generate_unique_guest_id():
        while True:
            candidate = f"guest_{random.randint(1000, 9999)}"
            if candidate not in active_user_ids:
                active_user_ids.add(candidate)
                return candidate

    def handle_input(youtube_url, uploaded_file):
        if not youtube_url and not uploaded_file:
            return "❗ Please provide a YouTube URL or upload an audio file.", "", ""

        try:
            if youtube_url:
                filepath = AudioDownloader().download_youtube_audio(youtube_url)
            elif uploaded_file:
                filepath = uploaded_file

            transcript = Transcriber().transcribe_audio(filepath)
            summary = summarize_with_palm(transcript)
            return transcript, summary, os.path.basename(filepath)
        except Exception as e:
            return f"⚠️ Error processing input: {e}", "", ""

    def qa_interface(query, history, user_id):
        result = rag.qa_chain.invoke({"query": query})
        answer = result["result"]

        query_log.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "user_id": user_id,
                "question": query,
                "answer": answer,
            }
        )

        with open("query_log.json", "w", encoding="utf-8") as f:
            json.dump(query_log, f, ensure_ascii=False, indent=2)

        return answer

    user_id_state = gr.State(generate_unique_guest_id())

    with gr.Blocks(title="RAG QA System with Gemini") as demo:
        gr.Markdown(
            "## 🎙️ Learning System QA Interface\n"
            "Upload audio or paste a YouTube URL, "
            "then ask questions based on the content."
        )

        with gr.Row():
            youtube_input = gr.Textbox(
                label="📺 YouTube URL", placeholder="Paste a YouTube link here"
            )
            file_input = gr.File(
                label="🎧 Upload Audio File", file_types=[".mp3", ".wav", ".m4a"]
            )

        with gr.Row():
            transcribe_btn = gr.Button("Transcribe & Summarize")

        transcript_output = gr.Textbox(label="📝 Transcript", lines=10)
        summary_output = gr.Markdown(label="🧠 Summary")
        filename_display = gr.Textbox(label="📂 Processed File", interactive=False)

        transcribe_btn.click(
            fn=handle_input,
            inputs=[youtube_input, file_input],
            outputs=[transcript_output, summary_output, filename_display],
        )

        gr.Markdown("## 💬 Ask Questions")
        gr.ChatInterface(
            fn=lambda q, h: qa_interface(q, h, user_id_state.value),
            title="RAG QA Chatbot",
        )

    demo.launch(share=True)
