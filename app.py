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

# 初始化组件
transcriber = VideoTranscriber(model_size="base")
extractor = VideoExtractor(interval_seconds=config.get("prompt_template", {}).get("interval_seconds", 10))
replacer = FaceReplacer(chinese_faces_dir="chinese_faces")
prompt_builder = PromptBuilder(config.get("prompt_template", {}))
runner = PavoRunner(token=config.get("pavo", {}).get("auth_token", ""))

# 全局状态缓存当前处理的任务
current_job = {
    "prompts": "",
    "frames": []
}

def auto_process_video(video_path):
    """用户上传视频后，自动触发：提取文案 -> 抽帧换脸 -> 生成提示词"""
    if not video_path:
        return "", [], "", "⚠️ 请上传视频文件！"

    output_dir = "output_assets"
    raw_frames = os.path.join(output_dir, "raw_frames")
    swapped_frames = os.path.join(output_dir, "swapped_frames")

    # 1. Whisper 提取口播
    yield "⏳ 正在提取口播逐字稿...", [], "", "处理中..."
    transcript = transcriber.extract_transcript(video_path, output_dir)

    # 2. 抽帧（首尾帧 + 每10秒）
    yield transcript, [], "⏳ 正在抽帧并替换华人模特...", "处理中..."
    frames_info = extractor.extract_keyframes(video_path, raw_frames)

    # 3. 换脸
    swapped_paths = replacer.replace_faces_in_frames(frames_info, swapped_frames)
    current_job["frames"] = swapped_paths

    # 4. 生成提示词
    prompts_text = prompt_builder.build_prompts(frames_info, transcript)
    current_job["prompts"] = prompts_text

    yield transcript, swapped_paths, prompts_text, "✅ 解析完成！请检查/微调下方提示词，确认后点击下方按钮下发生图/视频"

def trigger_pavo_generation(confirmed_prompts):
    """用户确认无误后，真正发送到 Pavo 生成视频"""
    if not current_job["frames"]:
        return "❌ 尚未解析视频，无法提交"
    
    # 提交最终确认的提示词和换好脸的关键帧
    runner.trigger_generation(confirmed_prompts, current_job["frames"])
    return "🚀 已成功下发 Pavo AI 工作流！正在云端批量渲染视频..."

# 构建界面
with gr.Blocks(title="Pavo AI 视频爆款复刻工作流") as demo:
    gr.Markdown("# 🚀 爆款视频智能分镜复刻与华人模特替换")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 设置 upload 事件：视频上传完成直接自动开始解析！
            video_input = gr.Video(label="上传爆款视频（拖入后自动触发解析）")
            status_box = gr.Label(label="当前进度状态")

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(label="1. 提取的完整口播文案（可编辑）", lines=4)
            frames_gallery = gr.Gallery(label="2. 首尾帧及每10秒关键帧（已替换随机华人）", columns=3, height=260)
            prompts_box = gr.Textbox(label="3. 生成的分镜提示词（可根据需要微调）", lines=6)

    with gr.Row():
        confirm_btn = gr.Button("🔥 确认无误，下发 Pavo AI 生成视频", variant="primary", size="lg")
        result_label = gr.Label(label="执行结果")

    # 只要视频一上传，立即自动开始流水线（无需手动按开始按钮）
    video_input.upload(
        fn=auto_process_video,
        inputs=[video_input],
        outputs=[transcript_box, frames_gallery, prompts_box, status_box]
    )

    # 确认后提交
    confirm_btn.click(
        fn=trigger_pavo_generation,
        inputs=[prompts_box],
        outputs=[result_label]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
