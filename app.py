import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from core.extractor import VideoExtractor
from core.transcriber import AudioTranscriber
from core.face_swap import FaceReplacer
from core.prompt_builder import PromptBuilder
from core.pavo_runner import PavoRunner

app = FastAPI(title="Pavo Video Replicator")

# 挂载目录
os.makedirs("workspace/swapped_frames", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/workspace", StaticFiles(directory="workspace"), name="workspace")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化
extractor = VideoExtractor(interval_seconds=10)
transcriber = AudioTranscriber()
replacer = FaceReplacer(chinese_faces_dir="chinese_faces")
builder = PromptBuilder()

current_task = {"segments": [], "meta": {}}

class SegmentReviewItem(BaseModel):
    segment_index: int
    image_url: str
    prompt: str

class ConfirmGenerationRequest(BaseModel):
    token: str
    segments: List[SegmentReviewItem]

# 根路径直接返回 static 里的 index.html
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_video_path = f"workspace/{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_frames_dir = "workspace/raw_frames"
    swapped_frames_dir = "workspace/swapped_frames"

    frames_info = extractor.extract_keyframes(temp_video_path, raw_frames_dir)
    transcript_segments = transcriber.transcribe_and_split(temp_video_path, interval_sec=10)
    swapped_paths, meta = replacer.replace_faces_in_frames(frames_info, swapped_frames_dir)
    prompts = builder.build_full_task_prompts(swapped_paths, transcript_segments, meta=meta)

    segments_data = []
    for i, path in enumerate(swapped_paths):
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
    if not req.segments:
        raise HTTPException(status_code=400, detail="无切片数据")

    runner = PavoRunner(token=req.token)
    meta = current_task.get("meta", {"gender": "male", "age": 45})

    success_count = 0
    for seg in req.segments:
        local_img = seg.image_url.lstrip("/")
        if runner.trigger_generation(prompts_text=seg.prompt, image_paths=[local_img], meta=meta):
            success_count += 1

    return JSONResponse({"status": "success", "message": f"成功提交 {success_count} 个分镜任务到 Pavo！"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
