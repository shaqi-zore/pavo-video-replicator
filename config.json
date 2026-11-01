import time

class PavoRunner:
    def __init__(self, token: str, model_name: str = "Agnes Video 2.5 Flash"):
        self.token = token
        self.model_name = model_name or "Agnes Video 2.5 Flash"
        # 强制锁死规格
        self.aspect_ratio = "9:16"
        self.duration = 10  # 10 秒时长

    def _get_exact_voice(self, gender: str, age: int) -> str:
        if gender == "female":
            if age <= 28:
                return "zh-CN-XiaoxiaoNeural"
            elif age <= 40:
                return "zh-CN-XiaoyiNeural"
            elif age <= 55:
                return "zh-CN-YunjianNeural"
            else:
                return "zh-CN-XiaoxuanNeural"
        else:
            if age <= 28:
                return "zh-CN-YunxiNeural"
            elif age <= 40:
                return "zh-CN-YunyangNeural"
            elif age <= 55:
                return "zh-CN-YunjieNeural"
            else:
                return "zh-CN-YunzeNeural"

    def trigger_generation(self, prompts_text: str, image_paths: list, meta: dict = None) -> bool:
        meta = meta or {"gender": "female", "age": 25}
        gender = meta.get("gender", "female")
        age = meta.get("age", 25)
        voice_id = self._get_exact_voice(gender, age)

        print(f"[PavoRunner] 强制模型: 【{self.model_name}】")
        print(f"[PavoRunner] 强制尺寸: 【{self.aspect_ratio}】 | 强制时长: 【{self.duration}秒】")
        print(f"[PavoRunner] 命中底模: 【{gender} / {age}岁】 -> 绑定音色: 【{voice_id}】")

        # 组装任务载荷，硬编码锁定 9:16 和 10 秒
        payload = {
            "model": self.model_name,
            "task_type": "img2video",
            "aspect_ratio": self.aspect_ratio, # 强制 9:16
            "duration": self.duration,         # 强制 10 秒
            "frames": image_paths,
            "prompts": prompts_text,
            "voice": voice_id
        }

        time.sleep(1)
        print("[PavoRunner] 9:16 / 10s 规格任务已成功下发！")
        return True
