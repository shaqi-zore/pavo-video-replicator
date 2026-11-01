import os
import random
import cv2
from typing import List

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

    def _load_chinese_faces(self):
        if not os.path.exists(self.chinese_faces_dir) or not self.app:
            return
        for file in os.listdir(self.chinese_faces_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(self.chinese_faces_dir, file))
                faces = self.app.get(img)
                if faces:
                    self.face_banks.append(faces[0])

    def replace_faces_in_frames(self, frames_info: List[dict], output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        for item in frames_info:
            src_path = item["file_path"]
            img = cv2.imread(src_path)
            if self.swapper and self.face_banks and self.app:
                faces = self.app.get(img)
                if faces:
                    target_face = random.choice(self.face_banks)
                    for f in faces:
                        img = self.swapper.get(img, f, target_face, paste_back=True)

            out_path = os.path.join(output_dir, f"swapped_{os.path.basename(src_path)}")
            cv2.imwrite(out_path, img)
            processed_paths.append(out_path)
        return processed_paths
