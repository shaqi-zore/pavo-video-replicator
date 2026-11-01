import os
import cv2
from typing import List, Dict

class VideoExtractor:
    def __init__(self, interval_seconds: int = 10):
        # 强制写死 10 秒间隔
        self.interval_seconds = 10

    def _crop_to_9_16(self, frame):
        """将任意比例画面居中裁剪并缩放到 1080x1920 (9:16)"""
        h, w = frame.shape[:2]
        target_aspect = 9 / 16  # 0.5625
        current_aspect = w / h

        if current_aspect > target_aspect:
            # 画面过宽，裁切两侧
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            cropped = frame[:, start_x:start_x + new_w]
        else:
            # 画面过高，裁切上下
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            cropped = frame[start_y:start_y + new_h, :]

        # 统一输出高清 9:16 规格
        return cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)

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
                # 强制转为 9:16 竖屏格式
                frame_9_16 = self._crop_to_9_16(frame)
                save_path = os.path.join(raw_frames_dir, f"frame_{tag}_{int(timestamp):03d}s.jpg")
                cv2.imwrite(save_path, frame_9_16)
                extracted_info.append({"timestamp": timestamp, "tag": tag, "file_path": save_path})

        # 首帧
        save_frame(0, 0.0, "start")

        # 严格每 10 秒抽取一个镜头
        current_sec = self.interval_seconds
        while current_sec < duration_sec - 1:
            save_frame(int(current_sec * fps), float(current_sec), "interval")
            current_sec += self.interval_seconds

        # 尾帧
        if total_frames > 1:
            save_frame(total_frames - 1, duration_sec, "end")

        cap.release()
        return extracted_info
