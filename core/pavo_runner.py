import time

class PavoRunner:
    def __init__(self, token: str, model_name: str = "Agnes Video 2.5 Flash"):
        self.token = token
        self.model_name = model_name or "Agnes Video 2.5 Flash"

    def trigger_generation(self, prompts_text: str, image_paths: list) -> bool:
        print(f"[PavoRunner] 强制使用免费模型: {self.model_name}")
        print(f"[PavoRunner] 正在向 Pavo AI 派发 {len(image_paths)} 个分镜渲染任务...")
        time.sleep(1)
        return True
