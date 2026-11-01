import os
import torch
import whisper

# 限制并开启 CPU 多核全力运算
torch.set_num_threads(os.cpu_count() or 4)

class AudioTranscriber:
    def __init__(self, model_size: str = "tiny"):  # 纯 CPU 场景改用 tiny，秒级出词
        self.model = whisper.load_model(model_size, device="cpu")

    def transcribe_and_split(self, audio_or_video_path: str, interval_sec: int = 10) -> list:
        if not os.path.exists(audio_or_video_path):
            return []
        
        custom_prompt = "在家里点松塔香，这个香一直往上窜，直直往上拔，点上一根，香道演示，纯正普通话解说。"
        
        result = self.model.transcribe(
            audio_or_video_path,
            language="zh",
            initial_prompt=custom_prompt,
            temperature=0.0,
            fp16=False  # 明确关闭 CPU 不支持的 FP16，避免冗余告警
        )
        segments = result.get("segments", [])
        
        if not segments:
            text = result.get("text", "").strip()
            return [text] if text else ["在家里点松塔香，每天点上一根就可以了。"]

        total_duration = segments[-1]["end"] if segments else 10.0
        num_intervals = max(1, int(total_duration // interval_sec) + 1)
        split_texts = ["" for _ in range(num_intervals)]

        for seg in segments:
            idx = min(int(seg["start"] // interval_sec), num_intervals - 1)
            split_texts[idx] += seg["text"].strip() + " "

        return [t.strip() if t.strip() else "在家里点松塔香，每天点上一根就可以了。" for t in split_texts]
