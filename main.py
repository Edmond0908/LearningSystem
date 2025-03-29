import os

from utils.downloader import AudioDownloader
from utils.processor import TranscriptProcessor
from utils.rag_system import RAGSystem
from utils.summarizer import summarize_with_palm
from utils.transcriber import Transcriber


def main():

    if "GOOGLE_API_KEY" not in os.environ:
        raise OSError(
            "GOOGLE_API_KEY not set. Please export it as an environment variable."
        )

    # 0. Initialize RAGSystem (using Whisper base for transcription)
    rag = RAGSystem(model_size="base")

    # Create directories for audio and transcripts
    audio_dir = "audio"
    transcripts_dir = "transcripts"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # 1. Define multiple data sources
    # Template: (YouTube URL, output audio file path,
    #           transcript file path, label for the source)
    youtube_data = [
        (
            "https://www.youtube.com/watch?v=kHV1g-yHxgQ",
            f"{audio_dir}/Intro2DES.mp3",
            f"{transcripts_dir}/Intro2DES.txt",
            "Video1",
        ),
        (
            "https://www.youtube.com/watch?v=iqyxRyi_mEE",
            f"{audio_dir}/DESOverview.mp3",
            f"{transcripts_dir}/DESOverview.txt",
            "Video2",
        ),
    ]

    texts_and_sources = []
    for url, audio_name, transcript_name, source_label in youtube_data:
        # Download audio if it doesn't exist
        if not os.path.exists(audio_name):
            print(f"\nAudio file '{audio_name}' not found. Downloading from {url}...")
            AudioDownloader().download_youtube_audio(url, audio_name)
        else:
            print(f"\nAudio file '{audio_name}' already exists. Skipping download.")

        # Transcribe if transcript doesn't exist
        if not os.path.exists(transcript_name):
            print(
                f"Transcript '{transcript_name}' not found. "
                f"Transcribing '{audio_name}'..."
            )
            text = Transcriber().transcribe_audio(audio_name)
            TranscriptProcessor().save_transcript(text, transcript_name)

            # Summarize after transcribing using Google PaLM 2
            summary = summarize_with_palm(text)
            print("\n=== Summary of transcript ===")
            print(summary)

        else:
            print(
                f"Transcript '{transcript_name}' already exists. "
                "Skipping transcription."
            )
            with open(transcript_name, encoding="utf-8") as f:
                text = f.read()

        texts_and_sources.append((text, source_label))

    # 2. Build or load FAISS index with local embeddings
    faiss_index_dir = "faiss_index"
    if not os.path.exists(faiss_index_dir):
        print(
            "\nNo FAISS index folder found. "
            "Building index from multiple transcripts..."
        )
        rag.build_vectorstore_from_multiple_texts(
            texts_and_sources=texts_and_sources,
            chunk_size=1000,
            chunk_overlap=200,
            save_path=faiss_index_dir,
        )
    else:
        print(f"\nFAISS index found at '{faiss_index_dir}'. Loading...")
        rag.load_vectorstore(faiss_index_dir)

    # 3. Create QA chain using Vertex AI (PaLM 2)
    rag.create_qa_chain_with_vertexai(k=3, temperature=0.7)

    # 4. Ask a sample question
    question = "請問課程的重點是什麼？"
    answer = rag.answer_question(question)
    print(f"\nQ: {question}\nA: {answer}\n")

    # 5. (Optional) Launch Gradio UI
    rag.launch_gradio_app()


if __name__ == "__main__":
    main()
