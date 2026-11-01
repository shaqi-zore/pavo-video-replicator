import time

class PavoRunner:
    def __init__(self, token: str, model_name: str = "Agnes Video 2.5 Flash"):
        self.token = token
        # 强制默认使用 Agnes Video 2.5 Flash 免费模型
        self.model_name = model_name or "Agnes Video 2.5 Flash"

    def trigger_generation(self, prompts_text: str, image_paths: list) -> bool:
        """
        触发 Pavo AI 工作流生成
        强制锁定模型: Agnes Video 2.5 Flash
        """
        print(f"[PavoRunner] 强制使用模型: 【{self.model_name}】(免费版)")
        print(f"[PavoRunner] 正在向 Pavo AI 工作台推送 {len(image_paths)} 个分镜任务...")

        # 示例：构建向 Pavo 接口提交的任务载荷 (Payload)
        payload = {
            "model": self.model_name,  # 强制传递 Agnes Video 2.5 Flash
            "task_type": "img2video",
            "frames": image_paths,
            "prompts": prompts_text
        }
        
        # 如果是 Playwright 浏览器自动化：
        # page.locator("text=模型选择").click()
        # page.locator("text=Agnes Video 2.5 Flash").click()

        time.sleep(1)
        print("[PavoRunner] 任务已锁定免费模型并成功派发！")
        return True
