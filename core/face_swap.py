import os
import cv2
import glob
import insightface
from insightface.app import FaceAnalysis

class FaceReplacer:
    def __init__(self, model_path: str = "models/inswapper_128.onnx", chinese_faces_dir: str = "chinese_faces", faces_dir: str = None):
        self.model_path = model_path
        # 兼容两种参数传法，避免报错
        self.faces_dir = faces_dir if faces_dir is not None else chinese_faces_dir

        # 优先使用 CUDA 运算，若显卡耗尽或不可用则自动降级到 CPU
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        except Exception:
            providers = ['CPUExecutionProvider']

        # 初始化人脸分析检测器 (buffalo_l)
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # 加载人脸替换核心权重 inswapper_128.onnx
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到人脸模型文件: {self.model_path}，请确保已下载到 models 目录！")
            
        self.swapper = insightface.model_zoo.get_model(
            self.model_path, 
            download=False, 
            providers=providers
        )

    def _parse_meta_from_filename(self, filename: str) -> dict:
        """从底模文件名中提取性别和年龄，如: female_25.jpg -> {'gender': 'female', 'age': 25}"""
        base = os.path.splitext(os.path.basename(filename))[0]
        parts = base.split('_')
        gender = "male"
        age = 30
        if len(parts) >= 2:
            gender = parts[0].lower()
            try:
                age = int(parts[1])
            except ValueError:
                age = 30
        return {"gender": gender, "age": age, "filename": filename}

    def _find_best_match_face(self, target_gender: str = None, target_age: int = None) -> tuple:
        """从 chinese_faces 目录查找最匹配的底模图片"""
        image_files = glob.glob(os.path.join(self.faces_dir, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(self.faces_dir, "*.[pP][nN][gG]"))
        
        if not image_files:
            return None, {"gender": "male", "age": 30, "filename": ""}

        if not target_gender and not target_age:
            meta = self._parse_meta_from_filename(image_files[0])
            return image_files[0], meta

        best_file = image_files[0]
        min_diff = float("inf")
        best_meta = self._parse_meta_from_filename(best_file)

        for f in image_files:
            meta = self._parse_meta_from_filename(f)
            gender_match = (meta["gender"] == target_gender) if target_gender else True
            age_diff = abs(meta["age"] - (target_age or 30))
            score = age_diff if gender_match else age_diff + 100

            if score < min_diff:
                min_diff = score
                best_file = f
                best_meta = meta

        return best_file, best_meta

    def swap_face(self, source_img_path: str, target_gender: str = None, target_age: int = None) -> tuple:
        """
        对目标首帧图片进行华人模特换脸
        :param source_img_path: 视频抽出来的 9:16 首帧图片路径
        :return: (换脸后的 cv2 图像, 匹配到的模特元数据 dict)
        """
        target_img = cv2.imread(source_img_path)
        if target_img is None:
            raise ValueError(f"无法读取图片: {source_img_path}")

        face_path, meta = self._find_best_match_face(target_gender, target_age)
        if not face_path:
            return target_img, meta

        source_face_img = cv2.imread(face_path)
        source_faces = self.app.get(source_face_img)
        if not source_faces:
            return target_img, meta
        source_face = source_faces[0]

        target_faces = self.app.get(target_img)
        if not target_faces:
            return target_img, meta

        result_img = target_img.copy()
        for face in target_faces:
            result_img = self.swapper.get(result_img, face, source_face, paste_back=True)

        return result_img, meta
