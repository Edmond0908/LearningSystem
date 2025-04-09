import os

from app_ui import launch_gradio_app
from utils.rag_system import RAGSystem


def main():
    if "GOOGLE_API_KEY" not in os.environ:
        raise OSError(
            "GOOGLE_API_KEY not set. Please export it as an environment variable.\n"
            'Usage: export GOOGLE_API_KEY="your_api_key" '
            "or set it in your IDE's environment variables."
        )

    rag = RAGSystem(model_size="base")

    faiss_index_dir = "faiss_index"
    if not os.path.exists(faiss_index_dir):
        print("No FAISS index found. Please upload or generate content via the UI.")
    else:
        rag.load_vectorstore(faiss_index_dir)

    rag.create_qa_chain_with_vertexai(k=3, temperature=0.7)
    launch_gradio_app(rag)


if __name__ == "__main__":
    main()
