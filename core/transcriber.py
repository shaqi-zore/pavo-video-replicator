import os
import whisper

class VideoTranscriber:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def extract_transcript(self, video_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        result = self.model.transcribe(video_path, verbose=False)
        transcript = result.get("text", "").strip()
        with open(os.path.join(output_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)
        return transcript
