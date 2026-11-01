import os
import whisper
import google.generativeai as genai

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size, device="cpu")
        # 支持传入免费的 GEMINI_API_KEY 实现 0 耗时高精度纠错
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.llm = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.llm = None

    def _correct_with_llm(self, raw_text: str) -> str:
        """利用语言模型凭借上下文自动修正所有同音错字，无需维护任何字典"""
        if not self.llm or not raw_text.strip():
            return raw_text
        
        prompt = f"""你是一个专业的短视频口语台词纠错专家。以下是一段由轻量语音识别模型提取的电商带货视频台词，里面包含因口音或同音导致的错别字（例如把“烟雾”听成“腌物/淹物”，把“点燃”听成“点然”等）。
请结合上下文语境进行智能纠错，保持原话口语风格、字数节奏和原意，只输出纠错后的最终中文台词，不要输出任何多余的解释说明或开场白。

原始台词：
{raw_text}"""
        try:
            response = self.llm.generate_content(prompt)
            corrected = response.text.strip()
            return corrected if corrected else raw_text
        except Exception:
            return raw_text

    def transcribe_and_split(self, audio_or_video_path: str, interval_sec: int = 10) -> list:
        if not os.path.exists(audio_or_video_path):
            return []

        # 1. 本地 CPU 快速抓取基础发音文本
        result = self.model.transcribe(
            audio_or_video_path,
            language="zh",
            temperature=0.0,
            fp16=False
        )
        segments = result.get("segments", [])
        
        if not segments:
            raw_text = result.get("text", "").strip()
            fixed_text = self._correct_with_llm(raw_text)
            return [fixed_text] if fixed_text else ["台词识别为空。"]

        total_duration = segments[-1]["end"] if segments else 10.0
        num_intervals = max(1, int(total_duration // interval_sec) + 1)
        split_texts = ["" for _ in range(num_intervals)]

        for seg in segments:
            idx = min(int(seg["start"] // interval_sec), num_intervals - 1)
            split_texts[idx] += seg["text"].strip() + " "

        # 2. 对每个分段经过 LLM 语义校对，输出完全符合语言逻辑的正文
        final_splits = []
        for text_chunk in split_texts:
            if text_chunk.strip():
                final_splits.append(self._correct_with_llm(text_chunk.strip()))
            else:
                final_splits.append("台词演示。")

        return final_splits
