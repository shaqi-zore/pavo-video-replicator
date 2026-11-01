import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from core.transcriber import VideoTranscriber
from core.extractor import VideoExtractor
from core.face_swap import FaceReplacer
from core.prompt_builder import PromptBuilder
from core.pavo_runner import PavoRunner

app = FastAPI(title="Pavo AI Automation Pipeline")

# 确保必要的本地存储目录存在
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("output_assets/swapped_frames", exist_ok=True)
os.makedirs("output_assets/raw_frames", exist_ok=True)

# 挂载静态文件与切片图片预览目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output_assets", StaticFiles(directory="output_assets"), name="output_assets")

# 读取配置
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 初始化各核心算法模块
transcriber = VideoTranscriber(model_size="base")
extractor = VideoExtractor(interval_seconds=config.get("prompt_template", {}).get("interval_seconds", 10))
replacer = FaceReplacer(chinese_faces_dir="chinese_faces")
prompt_builder = PromptBuilder(config.get("prompt_template", {}))

# 强制锁定免费模型 Agnes Video 2.5 Flash
target_model = config.get("pavo", {}).get("model_name", "Agnes Video 2.5 Flash")
runner = PavoRunner(
    token=config.get("pavo", {}).get("auth_token", ""),
    model_name=target_model
)

# 内存缓存当前解析任务状态
task_cache = {
    "frames": [],
    "prompts": "",
    "transcript": ""
}

class SubmitPayload(BaseModel):
    transcript: str
    prompts: str

@app.get("/")
async def index():
    """打开根路径返回前端界面"""
    return FileResponse("static/index.html")

@app.post("/api/process")
async def process_video(file: UploadFile = File(...)):
    """上传视频后自动触发：文案转写 -> 抽首尾帧及10秒间隔 -> 换华人脸 -> 组装提示词"""
    try:
        temp_video_path = os.path.join("uploads", file.filename)
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_dir = "output_assets"
        raw_frames_dir = os.path.join(output_dir, "raw_frames")
        swapped_frames_dir = os.path.join(output_dir, "swapped_frames")

        # 1. Whisper 提取口播逐字稿
        transcript = transcriber.extract_transcript(temp_video_path, output_dir)

        # 2. 抽帧：首帧、尾帧、每10秒关键帧
        frames_info = extractor.extract_keyframes(temp_video_path, raw_frames_dir)

        # 3. 换脸：随机匹配华人模特面孔
        swapped_paths = replacer.replace_faces_in_frames(frames_info, swapped_frames_dir)

        # 4. 根据模板生成各分镜提示词
        prompts_text = prompt_builder.build_prompts(frames_info, transcript)

        # 缓存状态供确认下发时读取
        task_cache["frames"] = swapped_paths
        task_cache["prompts"] = prompts_text
        task_cache["transcript"] = transcript

        # 封装给前端展示的图片列表
        frames_response = []
        for idx, p in enumerate(swapped_paths):
            tag = frames_info[idx].get("tag", "frame") if idx < len(frames_info) else "frame"
            frames_response.append({
                "url": f"/{p.replace(os.sep, '/')}",
                "tag": tag
            })

        return {
            "success": True,
            "transcript": transcript,
            "prompts": prompts_text,
            "frames": frames_response
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/submit")
async def submit_to_pavo(payload: SubmitPayload):
    """用户二次确认后下发 Pavo AI，强制调用 Agnes Video 2.5 Flash 生成视频"""
    try:
        frames = task_cache.get("frames", [])
        if not frames:
            return {"success": False, "error": "未检测到已处理的关键帧图片"}

        success = runner.trigger_generation(payload.prompts, frames)
        if success:
            return {"success": True}
        return {"success": False, "error": "Pavo 任务调度失败"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
