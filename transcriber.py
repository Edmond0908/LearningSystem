import os
import whisper

class Transcriber:
    """
    Handles transcription using Whisper.
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None

    def load_model(self):
        """Lazily load the Whisper model."""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_size}'...")
            self.model = whisper.load_model(self.model_size)

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes an audio file using the Whisper model.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        self.load_model()
        print(f"Transcribing file: {file_path}")

        result = self.model.transcribe(file_path)
        text = result.get("text", "").strip()
        if not text:
            raise ValueError("Transcription returned empty text. Check if the audio file is valid or not silent.")
        return text