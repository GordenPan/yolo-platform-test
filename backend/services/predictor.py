"""推論服務：對上傳影像執行偵測，回傳結構化結果與標註後影像（base64 JPEG）。"""
from __future__ import annotations

import base64
from functools import lru_cache

import cv2
import numpy as np

from .registry import resolve_model


@lru_cache(maxsize=4)
def _load_model(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def predict_path(model_id: str, image_path: str, conf: float = 0.25) -> dict:
    """對伺服器本機路徑的影像推論（資料夾瀏覽模式用）。"""
    from pathlib import Path

    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到影像: {image_path}")
    return predict_image(model_id, p.read_bytes(), conf=conf)


def predict_image(model_id: str, image_bytes: bytes, conf: float = 0.25) -> dict:
    model = _load_model(resolve_model(model_id))

    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("無法解析上傳的影像")

    result = model.predict(arr, conf=conf, verbose=False)[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "class": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })

    annotated = result.plot()  # BGR ndarray，已畫好框
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("影像編碼失敗")

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image_b64": base64.b64encode(buf.tobytes()).decode(),
    }
