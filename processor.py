import os

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


class TranscriptProcessor:
    """
    Handles loading and chunking transcript text.
    """

    @staticmethod
    def load_transcript(path: str) -> str:
        """Loads the transcript text from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def chunk_text(text: str, chunk_size=1000, chunk_overlap=200):
        """
        Splits text into overlapping chunks to aid retrieval.
        """
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        return docs
