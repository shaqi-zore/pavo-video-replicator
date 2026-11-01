import os
import whisper

class VideoTranscriber:
    def __init__(self, model_size="base"):
        print(f"正在加载 Whisper 模型 ({model_size})...")
        self.model = whisper.load_model(model_size)

    def extract_transcript(self, video_path: str, output_dir: str) -> str:
        """提取完整口播文本并保存"""
        os.makedirs(output_dir, exist_ok=True)
        print("开始提取口播音频文本...")
        result = self.model.transcribe(video_path, verbose=False)
        transcript = result.get("text", "").strip()

        save_path = os.path.join(output_dir, "transcript.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        return transcript
