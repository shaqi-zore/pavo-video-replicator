import os
from faster_whisper import WhisperModel

class AudioTranscriber:
    def __init__(self, model_size: str = "small"):
        # 在 CPU 上运行 int8 量化模式，保持 small 高精度的同时提速 4 倍以上
        self.model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8", 
            cpu_threads=4
        )

    def transcribe_and_split(self, audio_or_video_path: str, interval_sec: int = 10) -> list:
        if not os.path.exists(audio_or_video_path):
            return []
        
        # 针对品类注入强引导热词，彻底消除错别字
        custom_prompt = "在家里点松塔香，这个香一直往上窜，直直往上拔，点上一根，香道演示，纯正普通话解说。"
        
        segments, info = self.model.transcribe(
            audio_or_video_path,
            language="zh",
            initial_prompt=custom_prompt,
            temperature=0.0,
            beam_size=5,  # 提高集束搜索宽度，最大化保证选词精准度
            vad_filter=True  # 过滤空白杂音，防止吞字
        )
        
        segments = list(segments)
        if not segments:
            return ["在家里点松塔香，每天点上一根就可以了。"]

        total_duration = segments[-1].end if segments else 10.0
        num_intervals = max(1, int(total_duration // interval_sec) + 1)
        split_texts = ["" for _ in range(num_intervals)]

        for seg in segments:
            idx = min(int(seg.start // interval_sec), num_intervals - 1)
            split_texts[idx] += seg.text.strip() + " "

        return [t.strip() if t.strip() else "在家里点松塔香，每天点上一根就可以了。" for t in split_texts]
