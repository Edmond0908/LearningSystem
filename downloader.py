import os
import subprocess

class AudioDownloader:
    """
    Handles downloading of YouTube audio using yt-dlp.
    """
    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """Check if ffmpeg is installed on the system."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def download_youtube_audio(url: str, output_path: str = "speech.mp3") -> str:
        """
        Downloads the audio from a YouTube video using yt-dlp and returns the saved file path.
        """
        if not AudioDownloader.check_ffmpeg_installed():
            raise EnvironmentError("ffmpeg is not installed. Please install ffmpeg to extract audio.")

        command = f'yt-dlp -x --audio-format mp3 -o "{output_path}" {url}'
        print(f"Downloading audio from: {url}")
        os.system(command)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Failed to download or extract audio to {output_path}")

        print(f"Audio saved to: {output_path}")
        return output_path