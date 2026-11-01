import os
import json
import gradio as gr

from core.transcriber import VideoTranscriber
from core.extractor import VideoExtractor
from core.face_swap import FaceReplacer
from core.prompt_builder import PromptBuilder
from core.pavo_runner import PavoRunner

# 加载配置
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 初始化业务组件
transcriber = VideoTranscriber(model_size="base")
extractor = VideoExtractor(interval_seconds=config["prompt_template"].get("interval_seconds", 10))
replacer = FaceReplacer(chinese_faces_dir="chinese_faces")
prompt_builder = PromptBuilder(config["prompt_template"])
runner = PavoRunner(token=config["pavo"]["auth_token"])

# 全局暂存变量
state = {
    "swapped_images": [],
    "frames_info": []
}

def analyze_video(video_path):
    if not video_path:
        return "请先上传视频", [], ""

    output_dir = "output_assets"
    raw_frames_dir = os.path.join(output_dir, "raw_frames")
    swapped_frames_dir = os.path.join(output_dir, "swapped_frames")

    # 1. 提取完整口播
    transcript = transcriber.extract_transcript(video_path, output_dir)

    # 2. 抽帧（首尾帧 + 每10秒）
    frames_info = extractor.extract_keyframes(video_path, raw_frames_dir)
    state["frames_info"] = frames_info

    # 3. 替换随机华人面孔
    swapped_paths = replacer.replace_faces_in_frames(frames_info, swapped_frames_dir)
    state["swapped_images"] = swapped_paths

    # 4. 套用模板生成分镜提示词
    prompts_text = prompt_builder.build_prompts(frames_info, transcript)

    return transcript, swapped_paths, prompts_text

def submit_to_pavo(confirmed_transcript, confirmed_prompts):
    if not state["swapped_images"]:
        return "❌ 尚未生成关键帧与提示词，请先上传并解析视频！"

    success = runner.trigger_generation(confirmed_prompts, state["swapped_images"])
    if success:
        return "✅ 已成功提交至 Pavo AI 工作流！所有分镜与华人首尾帧已开始排队渲染。"
    return "❌ 提交失败，请检查网络或配置中的 Pavo Token。"

# 构建 Gradio 交互界面
with gr.Blocks(title="Pavo AI 视频分镜复刻控制台") as demo:
    gr.Markdown("# 🎬 爆款视频分镜重构与华人模特复刻控制台")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="1. 上传爆款原视频")
            analyze_btn = gr.Button("⚡ 开始提取文案、抽帧换脸并生成提示词", variant="primary")

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(label="完整口播文案（确认/微调）", lines=5)
            prompts_box = gr.Textbox(label="每十秒视频分镜提示词（确认/微调）", lines=8)
            frames_gallery = gr.Gallery(label="替换华人模特后的抽帧（首尾帧 + 每10秒）", columns=3, height=280)

    with gr.Row():
        confirm_btn = gr.Button("🚀 确认无误，下发 Pavo AI 生成视频", variant="stop")
        status_output = gr.Label(label="执行状态")

    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[transcript_box, frames_gallery, prompts_box]
    )

    confirm_btn.click(
        fn=submit_to_pavo,
        inputs=[transcript_box, prompts_box],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
