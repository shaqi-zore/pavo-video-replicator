import pypinyin

class PromptBuilder:
    def __init__(self):
        # 针对不同性别和年龄段的配音精细描述
        self.voice_profiles = {
            ("female", "young"): {
                "desc": "one native mainland Chinese young female speaker, clear and natural tone, youthful cadence",
                "label": "young female"
            },
            ("female", "mid"): {
                "desc": "one native mainland Chinese mature female speaker, warm, intellectual and confident tone",
                "label": "mid-age female"
            },
            ("female", "older"): {
                "desc": "one native mainland Chinese older female speaker, kind, gentle, maternal and steady tone",
                "label": "senior female"
            },
            ("male", "young"): {
                "desc": "one native mainland Chinese young male speaker, energetic, clear and authentic",
                "label": "young male"
            },
            ("male", "mid"): {
                "desc": "one native mainland Chinese middle-aged male speaker, calm, grounded and professional delivery",
                "label": "middle-aged male"
            },
            ("male", "older"): {
                "desc": "one native mainland Chinese senior male speaker, authoritative, steady, wise and resonant voice",
                "label": "senior male"
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
        """自动将中文字幕逐字稿转换为带调拼音，作为语音引擎的发音引导"""
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
        voice_info = self.voice_profiles.get((gender, bracket), self.voice_profiles[("male", "mid")])

        # 拼音音素引导
        clean_text = dialogue_text.strip() if dialogue_text else "自然展示产品，保持生活化解说状态。"
        pinyin_text = self._chinese_to_pinyin(clean_text)

        prompt = f"""Generate exactly 10 seconds of 9:16 photorealistic image-to-video.

REFERENCES:
Use {frame_filename} as the actual first frame and the actor, clothing, composition and background reference.
Preserve authentic East Asian facial features, natural skin texture, detailed eyes, clothing folds and original background elements.
Remove the reference image's baked-in captions, watermarks or UI icons before animation. Reconstruct the underlying elements naturally.

PRODUCT AND HAND LOCK:
Maintain consistent hand posture without unnatural warping, phantom extra fingers or disappearing items.
Both hands hold or gesture naturally according to the visual reference.
All tabletop props, product vessels, and micro-details remain geometrically stable throughout.
No morphing items, random pickup extractions, hand swaps or sudden visual anomalies.

ACTION:
Maintain natural seated posture and realistic, subtle micro-movements.
Match mouth opening and lip sync precisely to the Chinese speech delivery.
Deliver natural explanatory hand gestures that correlate rhythmically with dialogue pauses.
Realistic subtle eye blinks, chest micro-rise breathing and steady head movements.
Keep the camera locked on tripod, no erratic zoom, sudden cuts or wild camera swing.

EXACT CHINESE DIALOGUE:
“{clean_text}”

FULL PINYIN — SILENT GUIDANCE:
{pinyin_text}

UNIFIED VOICE:
Use {voice_info['desc']}.
Establish a natural {voice_info['label']} voice consistent across all {total_segments} segments.
Keep pitch, timbre, accent, volume and microphone acoustic space identical.
Clear standard Mandarin, zero foreign accent, no slurring or synthetic robotic tone.

DELIVERY:
Begin speaking immediately with clear, conversational pacing and authentic emotion.
Aim to finish speaking around 9.2-9.5 seconds within the 10-second boundary.
Speak every supplied word exactly once, no word omission, hallucinated syllables or repetition.
Pinyin is silent acoustic guidance only, strictly never spoken aloud or rendered on screen.
During the remaining split-second hold, maintain subtle breathing and a relaxed pose suitable for continuous loop.

EDITING OVERLAYS:
Strictly no subtitles, headlines, animated stickers, logos, promotional tickers or decorative overlays.
Zero watermarks or synthetic UI elements.
Preserve original color balance, realistic ambient shadows and unchanged directional lighting.
Clean room ambience with pure spoken voice, no added background music, sirens or extraneous sound effects."""

        return prompt.strip()

    def build_full_task_prompts(self, frames_info: list, transcript_segments: list, meta: dict = None) -> list:
        """为所有切分的 10 秒关键帧批量组装结构化分镜提示词"""
        total_segments = len(frames_info)
        prompts = []
        for i, frame in enumerate(frames_info):
            frame_file = f"swapped_frame_{i+1:02d}.jpg"
            # 匹配对应时间切片的字幕，如无切片则回退
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
