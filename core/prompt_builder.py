from typing import List, Dict

class PromptBuilder:
    def __init__(self, template_config: dict):
        self.character = template_config.get("default_character", "25-year-old Chinese model, realistic face")
        self.suffix = template_config.get("style_suffix", "cinematic lighting --ar 9:16")

    def build_prompts(self, frames_info: List[Dict], transcript: str) -> str:
        prompt_blocks = []
        lines = [line.strip() for line in transcript.replace("。", "\n").split("\n") if line.strip()]

        for idx, item in enumerate(frames_info):
            t = int(item["timestamp"])
            tag = item["tag"]
            line_text = lines[idx % len(lines)] if lines else "主体自然讲述"
            prompt = (
                f"【分镜 {idx+1} | {t:02d}s ({tag})】\n"
                f"台词: \"{line_text}\"\n"
                f"提示词: {self.character}, looking at camera, scene at {t}s, {line_text[:20]}, {self.suffix}\n"
            )
            prompt_blocks.append(prompt)
        return "\n".join(prompt_blocks)
