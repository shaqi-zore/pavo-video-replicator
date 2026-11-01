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

# 确保文件夹存在
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("output_assets/swapped_frames", exist_ok=True)
os.makedirs("output_assets/raw_frames", exist_ok=True)

# 挂载静态路由与切片图片预览目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output_assets", StaticFiles(directory="output_assets"), name="output_assets")

# 读取配置与初始化算法类
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

transcriber = VideoTranscriber(model_size="base")
extractor = VideoExtractor(interval_seconds=config.get("prompt_template", {}).get("interval_seconds", 10))
replacer = FaceReplacer(chinese_faces_dir="chinese_faces")
prompt_builder = PromptBuilder(config.get("prompt_template", {}))

# 强制绑定 Agnes Video 2.5 Flash 免费模型
target_model = config.get("pavo", {}).get("model_name", "Agnes Video 2.5 Flash")
runner = PavoRunner(
    token=config.get("pavo", {}).get("auth_token", ""),
    model_name=target_model
)

task_cache = {"frames": [], "prompts": "", "transcript": ""}

class SubmitPayload(BaseModel):
    transcript: str
    prompts: str

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/process")
async def process_video(file: UploadFile = File(...)):
    try:
        temp_video_path = os.path.join("uploads", file.filename)
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_dir = "output_assets"
        raw_frames_dir = os.path.join(output_dir, "raw_frames")
        swapped_frames_dir = os.path.join(output_dir, "swapped_frames")

        # 1. 提取口播
        transcript = transcriber.extract_transcript(temp_video_path, output_dir)

        # 2. 抽帧
        frames_info = extractor.extract_keyframes(temp_video_path, raw_frames_dir)

        # 3. 换脸
        swapped_paths = replacer.replace_faces_in_frames(frames_info, swapped_frames_dir)

        # 4. 组装提示词
        prompts_text = prompt_builder.build_prompts(frames_info, transcript)

        task_cache["frames"] = swapped_paths
        task_cache["prompts"] = prompts_text
        task_cache["transcript"] = transcript

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
    try:
        frames = task_cache.get("frames", [])
        if not frames:
            return {"success": False, "error": "未检测到关键帧数据"}

        success = runner.trigger_generation(payload.prompts, frames)
        if success:
            return {"success": True}
        return {"success": False, "error": "Pavo 任务调度失败"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
