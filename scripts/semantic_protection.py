"""Bundled, CPU-only DeepLab v3 protection; no network and no image generation."""
from pathlib import Path
import hashlib
import zipfile
import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'assets' / 'deeplab_v3.tflite'
MODEL_SHA256 = 'ff36e24d40547fe9e645e2f4e8745d1876d6e38b332d39a82f0bf0f5d1d561b3'
PROTECTED_LABELS = ('person', 'cat', 'dog')


class SemanticProtector:
    def __init__(self, model=MODEL):
        import mediapipe as mp
        self.mp = mp
        model = Path(model)
        if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError('Bundled DeepLab model SHA256 mismatch')
        with zipfile.ZipFile(model) as z:
            self.labels = z.read('labels.txt').decode().splitlines()
        if len(self.labels) != 21 or any(x not in self.labels for x in PROTECTED_LABELS):
            raise RuntimeError('Unexpected DeepLab category schema')
        self.ids = [self.labels.index(x) for x in PROTECTED_LABELS]
        opts = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True, output_confidence_masks=False)
        self.task = mp.tasks.vision.ImageSegmenter.create_from_options(opts)
        if list(self.task.labels) != self.labels:
            self.task.close()
            raise RuntimeError('MediaPipe output labels disagree with bundled metadata')
        self.samples = 0

    def categories(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb, dtype=np.uint8))
        result = self.task.segment(image)
        mask = result.category_mask.numpy_view()
        if mask.dtype != np.uint8 or mask.shape[:2] != rgb.shape[:2]:
            raise RuntimeError('Unexpected MediaPipe category output')
        self.samples += 1
        return mask.reshape(rgb.shape[:2]).copy()

    def mask(self, rgb):
        return self.mask_from_categories(self.categories(rgb))

    def mask_from_categories(self, categories):
        mask = np.isin(categories, self.ids).astype(np.uint8)
        # Small boundary guard, not a large component/garment expansion.
        radius = max(1, round(min(categories.shape[:2]) * .008))
        return cv2.dilate(mask, np.ones((2*radius+1, 2*radius+1), np.uint8))

    def close(self):
        self.task.close()


_protector = None

def protector():
    global _protector
    if _protector is None:
        _protector = SemanticProtector()
    return _protector
