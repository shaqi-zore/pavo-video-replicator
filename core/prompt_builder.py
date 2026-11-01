import pypinyin

class PromptBuilder:
    def __init__(self):
        # 对应不同性别/年龄的全新人物外貌与服饰设定（彻底替换原人物特征）
        self.character_styles = {
            ("female", "young"): {
                "appearance": "A young Chinese woman in her early 20s, clean ponytail hair, wearing modern minimalist cream-white knitted cardigan",
                "voice_desc": "one native mainland Chinese young female speaker, clear and natural tone, youthful cadence",
                "voice_label": "young female"
            },
            ("female", "mid"): {
                "appearance": "An elegant Chinese woman in her 30s-40s, stylish shoulder-length hair, wearing a premium dark green modern tea-art linen blouse",
                "voice_desc": "one native mainland Chinese mature female speaker, warm, intellectual and confident tone",
                "voice_label": "mid-age female"
            },
            ("female", "older"): {
                "appearance": "A kind-looking Chinese senior woman in her 60s, neat silver-black bun hair, wearing an embroidered traditional cotton jacket",
                "voice_desc": "one native mainland Chinese older female speaker, gentle and steady tone",
                "voice_label": "senior female"
            },
            ("male", "young"): {
                "appearance": "A clean-shaven young Chinese man in his 20s, stylish modern short haircut, wearing a casual gray crewneck sweater",
                "voice_desc": "one native mainland Chinese young male speaker, energetic, clear and authentic",
                "voice_label": "young male"
            },
            ("male", "mid"): {
                "appearance": "A refined middle-aged Chinese gentleman in his 40s, neat classic haircut, wearing a charcoal-gray Zen style linen shirt",
                "voice_desc": "one native mainland Chinese middle-aged male speaker, calm, grounded and professional delivery",
                "voice_label": "middle-aged male"
            },
            ("male", "older"): {
                "appearance": "A distinguished senior Chinese gentleman in his 60s, natural salt-and-pepper hair, wearing a traditional dark navy silk-blend Tang suit",
                "voice_desc": "one native mainland Chinese senior male speaker, authoritative, steady and resonant voice",
                "voice_label": "senior male"
            }
        }

    def _get_age_bracket(self, age: int) -> str:
        if age <= 28:
            return "young"
        elif age <= 55:
            return "mid"
        else:
            return "older"

    def _chinese_to_pinyin(self, text: str) -> str:
        if not text:
            return ""
        pinyin_list = pypinyin.lazy_pinyin(text, style=pypinyin.Style.TONE)
        return " ".join(pinyin_list)

    def build_prompt_for_segment(
        self,
        frame_filename: str,
        dialogue_text: str,
        meta: dict = None,
        segment_index: int = 1,
        total_segments: int = 1
    ) -> str:
        meta = meta or {"gender": "male", "age": 45}
        gender = meta.get("gender", "male")
        age = meta.get("age", 45)
        bracket = self._get_age_bracket(age)
        style = self.character_styles.get((gender, bracket), self.character_styles[("male", "mid")])

        clean_text = dialogue_text.strip() if dialogue_text else "在家里点松塔香，每天点上一根就可以了。"
        pinyin_text = self._chinese_to_pinyin(clean_text)

        # 核心改动：在 REFERENCES 和 PRODUCT LOCK 中明确指示替换全身体型与服饰，仅保留桌景与商品
        prompt = f"""Generate exactly 10 seconds of 9:16 photorealistic image-to-video.

REFERENCES & SUBJECT REPLACEMENT:
Use {frame_filename} as composition, table setup and background reference.
OVERRIDE AND REPLACE THE ACTOR: Completely replace the person with {style['appearance']}.
Replace head, hair, body physique and clothing completely according to the new persona.
Retain only the original room background, lighting mood and tabletop composition.
Remove any reference baked-in subtitles, text watermarks or overlays.

PRODUCT AND HAND LOCK:
The actor's new hands hold or gesture naturally according to the dialogue.
Both hands remain empty of foreign objects; no extra phantom fingers or morphing.
The identical incense stick and burner setup on the table remain 100% stable in geometry and position.
Preserve all tabletop decor, miniature landscape and props exactly as shown.
No disappearing objects, sudden product swaps or hand warping.

ACTION:
The new Chinese actor begins seated, leaning slightly forward, naturally presenting and explaining the product.
Natural lip sync and mouth shapes matching the Chinese speech rhythm precisely.
Natural head tilt, subtle blinks, and realistic chest rise breathing.
Camera is locked steadily on tripod; no wild shaking or abrupt zooms.

EXACT CHINESE DIALOGUE:
“{clean_text}”

FULL PINYIN — SILENT GUIDANCE:
{pinyin_text}

UNIFIED VOICE:
Use {style['voice_desc']}.
Establish a natural {style['voice_label']} voice consistent across all {total_segments} segments.
Clear Mandarin, zero foreign accent, studio clean acoustics.

DELIVERY:
Begin speaking immediately with clear, efficient delivery.
Aim to finish around 9.2-9.5 seconds within the 10-second boundary.
Speak every supplied word exactly once.
Pinyin is silent acoustic guidance only, strictly never spoken aloud or rendered on screen.

EDITING OVERLAYS:
Strictly no subtitles, headlines, animated stickers or decorative overlays.
Natural thin smoke rising steadily, locked camera and realistic ambient room lighting."""

        return prompt.strip()

    def build_full_task_prompts(self, frames_info: list, transcript_segments: list, meta: dict = None) -> list:
        total_segments = len(frames_info)
        prompts = []
        for i, frame in enumerate(frames_info):
            frame_file = f"frame_{i+1:02d}.jpg"
            text = transcript_segments[i] if i < len(transcript_segments) else ""
            seg_prompt = self.build_prompt_for_segment(
                frame_filename=frame_file,
                dialogue_text=text,
                meta=meta,
                segment_index=i + 1,
                total_segments=total_segments
            )
            prompts.append(seg_prompt)
        return prompts
