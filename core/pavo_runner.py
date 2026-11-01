import time

class PavoRunner:
    def __init__(self, token: str):
        self.token = token

    def trigger_generation(self, prompts_text: str, image_paths: list) -> bool:
        """
        触发 Pavo AI 工作流生成
        通过 API 请求或自动化驱动分镜节点
        """
        print(f"收到确认指令，开始向 Pavo AI 推送 {len(image_paths)} 个分镜任务...")
        time.sleep(2)  # 模拟网络交互过程
        return True
