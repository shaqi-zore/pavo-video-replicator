import os
import re
import random
import cv2
from typing import List, Tuple, Dict

class FaceReplacer:
    def __init__(self, chinese_faces_dir: str = "chinese_faces"):
        self.chinese_faces_dir = chinese_faces_dir
        self.face_banks = []
        self.app = None
        self.swapper = None
        self._init_models()

    def _init_models(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            model_path = os.path.join("models", "inswapper_128.onnx")
            if os.path.exists(model_path):
                self.swapper = insightface.model_zoo.get_model(model_path, download=False)
            self._load_chinese_faces()
        except Exception:
            pass

    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """解析类似 female_20.jpg 或 male_45.jfif"""
        name_lower = filename.lower()
        gender = "female" if name_lower.startswith("female") else "male"
        
        # 提取文件名中的数字作为年龄
        age_match = re.search(r'\d+', name_lower)
        age = int(age_match.group()) if age_match else 25
        return {"gender": gender, "age": age}

    def _load_chinese_faces(self):
        if not os.path.exists(self.chinese_faces_dir) or not self.app:
            return
        for file in os.listdir(self.chinese_faces_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif')):
                img_path = os.path.join(self.chinese_faces_dir, file)
                img = cv2.imread(img_path)
                faces = self.app.get(img)
                if faces:
                    meta = self._parse_filename(file)
                    self.face_banks.append({
                        "face": faces[0],
                        "gender": meta["gender"],
                        "age": meta["age"],
                        "filename": file
                    })

    def replace_faces_in_frames(self, frames_info: List[dict], output_dir: str) -> Tuple[List[str], Dict]:
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        
        # 默认元数据
        selected_meta = {"gender": "female", "age": 25, "filename": "default"}
        target_item = None

        if self.face_banks:
            target_item = random.choice(self.face_banks)
            selected_meta = {
                "gender": target_item["gender"],
                "age": target_item["age"],
                "filename": target_item["filename"]
            }
            print(f"[FaceReplacer] 选中底模: {selected_meta['filename']} (性别: {selected_meta['gender']}, 年龄: {selected_meta['age']})")

        for item in frames_info:
            src_path = item["file_path"]
            img = cv2.imread(src_path)
            if self.swapper and target_item and self.app:
                faces = self.app.get(img)
                if faces:
                    for f in faces:
                        img = self.swapper.get(img, f, target_item["face"], paste_back=True)

            out_path = os.path.join(output_dir, f"swapped_{os.path.basename(src_path)}")
            cv2.imwrite(out_path, img)
            processed_paths.append(out_path)
            
        return processed_paths, selected_meta
