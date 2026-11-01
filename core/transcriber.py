import os
import whisper

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def transcribe_and_split(self, audio_or_video_path: str, interval_sec: int = 10) -> list:
        """提取台词并按设定秒数切分字幕列表"""
        if not os.path.exists(audio_or_video_path):
            return []
        
        result = self.model.transcribe(audio_or_video_path, language="zh")
        segments = result.get("segments", [])
        
        if not segments:
            text = result.get("text", "").strip()
            return [text] if text else ["自然解说展示产品。"]

        # 获取总时长
        total_duration = segments[-1]["end"] if segments else 10.0
        num_intervals = max(1, int(total_duration // interval_sec) + 1)
        split_texts = ["" for _ in range(num_intervals)]

        for seg in segments:
            idx = min(int(seg["start"] // interval_sec), num_intervals - 1)
            split_texts[idx] += seg["text"].strip() + " "

        return [t.strip() if t.strip() else "自然解说展示产品。" for t in split_texts]
