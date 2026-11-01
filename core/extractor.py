import os
import cv2
from typing import List, Dict

class VideoExtractor:
    def __init__(self, interval_seconds: int = 10):
        self.interval_seconds = interval_seconds

    def extract_keyframes(self, video_path: str, raw_frames_dir: str) -> List[Dict]:
        os.makedirs(raw_frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        extracted_info = []

        def save_frame(frame_idx: int, timestamp: float, tag: str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                save_path = os.path.join(raw_frames_dir, f"frame_{tag}_{int(timestamp):03d}s.jpg")
                cv2.imwrite(save_path, frame)
                extracted_info.append({"timestamp": timestamp, "tag": tag, "file_path": save_path})

        save_frame(0, 0.0, "start")
        current_sec = self.interval_seconds
        while current_sec < duration_sec - 1:
            save_frame(int(current_sec * fps), float(current_sec), "interval")
            current_sec += self.interval_seconds

        if total_frames > 1:
            save_frame(total_frames - 1, duration_sec, "end")

        cap.release()
        return extracted_info
