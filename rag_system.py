import os
import whisper
import yt_dlp
import shutil
import time

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

class RAGSystem:
    def __init__(self, model_size="base"):
        self.whisper_model = whisper.load_model(model_size)
        self.vectorstore = None
        self.qa_chain = None

    def download_audio(self, youtube_url, output_path="output.mp3"):
        """
        Downloads audio from the specified YouTube URL and saves it to the given output path.

        Arguments
        ========
        youtube_url : str
            The URL of the YouTube video to download audio from.
        output_path : str
            The path where the downloaded audio file will be saved.
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        if os.path.exists('temp_audio.mp3'):
            shutil.move('temp_audio.mp3', output_path)
            print(f"Audio saved to: {output_path}")

    def transcribe(self, audio_path):
        """
        Transcribes the audio file located at the given path using the Whisper model.

        Arguments
        ========
        audio_path : str
            The path to the audio file to be transcribed.

        Returns
        =======
        str
            The transcribed text from the audio file.
        """
        print(f"Transcribing file: {audio_path}")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def save_transcript(self, text, path):
        """
        Saves the provided text to a specified file path.

        Arguments
        ========
        text : str
            The transcript text to be saved.
        path : str
            The file path where the transcript will be saved.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcript saved to {path}")

    def build_vectorstore_from_multiple_texts(self, texts_and_sources, chunk_size=1000, chunk_overlap=200, save_path="faiss_index"):
        """
        Builds a vector store from multiple texts and their corresponding sources.

        Arguments
        ========
        texts_and_sources : list of tuples
            A list of tuples where each tuple contains a text and its source.
        chunk_size : int
            The size of each text chunk to be processed.
        chunk_overlap : int
            The number of overlapping characters between chunks.
        save_path : str
            The path where the vector store will be saved.
        """
        all_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for text, source in texts_and_sources:
            docs = splitter.create_documents([text], metadatas=[{"source": source}] * len([text]))
            all_docs.extend(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        def _safe_embed_documents(docs, batch_size=2, max_retries=3):
            from langchain.docstore.document import Document
            vectorstore = None

            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                for attempt in range(1, max_retries + 1):
                    try:
                        if vectorstore is None:
                            vectorstore = FAISS.from_documents(batch, embeddings)
                        else:
                            vs = FAISS.from_documents(batch, embeddings)
                            vectorstore.merge_from(vs)
                        break
                    except Exception as e:
                        print(f"⚠️ Error embedding batch {i//batch_size+1} (attempt {attempt}): {e}")
                        if attempt == max_retries:
                            print(f"❌ Giving up on batch {i//batch_size+1}")
                        else:
                            time.sleep(2 ** attempt)
            return vectorstore

        try:
            self.vectorstore = _safe_embed_documents(all_docs)
            if self.vectorstore:
                self.vectorstore.save_local(save_path)
                print("✅ Vector store with multiple data sources created!")
            else:
                print("❌ Failed to create vector store.")
        except Exception as e:
            print(f"⚠️ Error creating vector store: {e}")

    def load_vectorstore(self, load_path="faiss_index"):
        """
        Loads a vector store from the specified local path.

        Arguments
        ========
        load_path : str
            The path from which the vector store will be loaded.
        """
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded from disk.")

    def create_qa_chain_with_vertexai(self, k=3, temperature=0.7):
        """
        Initializes a QA chain using the specified parameters.

        Arguments
        ========
        k : int
            The number of documents to retrieve for answering a question.
        temperature : float
            The temperature parameter for controlling randomness in the output.
        """
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro-001",
            temperature=temperature,
            max_output_tokens=512
        )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        print("✅ QA chain initialized with Gemini (models/gemini-1.5-pro-001).")

    def answer_question(self, question):
        """
        Answers a question using the initialized QA chain.

        Arguments
        ========
        question : str
            The question to be answered.

        Returns
        =======
        str
            The answer to the question.
        """
        if not self.qa_chain:
            raise RuntimeError("QA chain not initialized.")
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

    def launch_gradio_app(self):
        """
        Launches a Gradio interface for the QA system, allowing users to interact with it.

        This method sets up the chat interface and handles user queries and responses.
        """
        import gradio as gr
        import datetime
        import json
        import random

        query_log = []
        active_user_ids = set()

        def generate_unique_guest_id():
            while True:
                candidate = f"guest_{random.randint(1000, 9999)}"
                if candidate not in active_user_ids:
                    active_user_ids.add(candidate)
                    return candidate

        def qa_interface(query, history, user_id):
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"]

            query_log.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "user_id": user_id,
                "question": query,
                "answer": answer
            })

            with open("query_log.json", "w", encoding="utf-8") as f:
                json.dump(query_log, f, ensure_ascii=False, indent=2)

            return answer

        user_id_state = gr.State(generate_unique_guest_id())

        gr.ChatInterface(
            fn=lambda q, h: qa_interface(q, h, user_id_state.value),
            title="RAG QA System with Gemini"
        ).launch(share=True)