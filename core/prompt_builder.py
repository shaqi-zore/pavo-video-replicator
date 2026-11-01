from typing import List, Dict

class PromptBuilder:
    def __init__(self, template_config: dict):
        self.character = template_config.get(
            "default_character",
            "25-year-old Chinese influencer, natural Asian face"
        )
        self.suffix = template_config.get(
            "style_suffix",
            "8k resolution, cinematic lighting --ar 9:16"
        )

    def build_prompts(self, frames_info: List[Dict], transcript: str) -> str:
        """根据抽帧时间轴和口播生成分段分镜提示词"""
        prompt_blocks = []
        
        # 将文本切分为若干简短片段配合分镜
        lines = [line.strip() for line in transcript.replace("。", "\n").split("\n") if line.strip()]

        for idx, item in enumerate(frames_info):
            t = int(item["timestamp"])
            tag = item["tag"]
            
            associated_text = lines[idx % len(lines)] if lines else "主体自然讲述与互动"
            
            prompt = (
                f"【分镜 {idx+1} | 时间戳: {t:02d}s ({tag})】\n"
                f"对应台词: \"{associated_text}\"\n"
                f"提示词: {self.character}, looking at camera, speaking naturally, scene at {t} seconds, "
                f"{associated_text[:25]}, {self.suffix}\n"
            )
            prompt_blocks.append(prompt)

        return "\n".join(prompt_blocks)
