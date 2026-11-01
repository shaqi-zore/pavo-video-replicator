import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from typing import List, Dict

from core.extractor import VideoExtractor
from core.transcriber import AudioTranscriber
from core.face_swap import FaceReplacer
from core.prompt_builder import PromptBuilder
from core.pavo_runner import PavoRunner

app = FastAPI(title="Pavo Video Replicator")

# 挂载静态文件目录供前端预览图片
os.makedirs("workspace", exist_ok=True)
os.makedirs("workspace/swapped_frames", exist_ok=True)
app.mount("/workspace", StaticFiles(directory="workspace"), name="workspace")
templates = Jinja2Templates(directory="templates")

# 初始化核心处理器
extractor = VideoExtractor(interval_seconds=10)
transcriber = AudioTranscriber()
replacer = FaceReplacer(chinese_faces_dir="chinese_faces")
builder = PromptBuilder()

# 任务缓存（单任务演示，可按 session 扩展）
current_task = {
    "segments": [],
    "meta": {}
}

class SegmentReviewItem(BaseModel):
    segment_index: int
    image_url: str
    prompt: str

class ConfirmGenerationRequest(BaseModel):
    token: str
    segments: List[SegmentReviewItem]

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """第一阶段：上传视频 -> 抽帧 9:16 -> 换脸 -> 生成提示词 -> 返回前端审查"""
    temp_video_path = f"workspace/{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_frames_dir = "workspace/raw_frames"
    swapped_frames_dir = "workspace/swapped_frames"

    # 1. 抽取 9:16 关键帧 (每10秒)
    frames_info = extractor.extract_keyframes(temp_video_path, raw_frames_dir)

    # 2. 识别字幕并按 10 秒拆分
    transcript_segments = transcriber.transcribe_and_split(temp_video_path, interval_sec=10)

    # 3. 随机选定一个华人模特底片换脸，提取 meta (gender, age)
    swapped_paths, meta = replacer.replace_faces_in_frames(frames_info, swapped_frames_dir)

    # 4. 构建每段 10 秒的精细提示词
    prompts = builder.build_full_task_prompts(swapped_paths, transcript_segments, meta=meta)

    # 组装返回给前端审查的数据
    segments_data = []
    for i, path in enumerate(swapped_paths):
        # 转为 Web 可访问的相对路径
        web_url = f"/workspace/swapped_frames/{os.path.basename(path)}"
        segments_data.append({
            "segment_index": i + 1,
            "image_url": web_url,
            "prompt": prompts[i] if i < len(prompts) else ""
        })

    current_task["segments"] = segments_data
    current_task["meta"] = meta

    return JSONResponse({
        "status": "ready_for_review",
        "model_used": meta.get("filename", "default"),
        "gender": meta.get("gender"),
        "age": meta.get("age"),
        "segments": segments_data
    })

@app.post("/api/generate")
async def confirm_and_generate(req: ConfirmGenerationRequest):
    """第二阶段：人工审查确认后，才正式下发至 Pavo AI 生成视频"""
    if not req.segments:
        raise HTTPException(status_code=400, detail="没有可生成的切片数据")

    runner = PavoRunner(token=req.token)
    meta = current_task.get("meta", {"gender": "male", "age": 45})

    success_count = 0
    for seg in req.segments:
        local_img = seg.image_url.lstrip("/")
        # 调用 Pavo 提交任务
        ok = runner.trigger_generation(
            prompts_text=seg.prompt,
            image_paths=[local_img],
            meta=meta
        )
        if ok:
            success_count += 1

    return JSONResponse({
        "status": "success",
        "message": f"成功提交 {success_count} 个分镜生成任务到 Pavo AI！"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
