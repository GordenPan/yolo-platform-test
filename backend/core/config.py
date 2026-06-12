"""全域路徑與設定。所有模組統一從這裡取得目錄位置。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
YOLOV26_DIR = MODELS_DIR / "YOLOV26"   # 預留：Ultralytics 未來發布的 YOLOv26 權重（.pt）或架構設定（.yaml）
RUNS_DIR = BASE_DIR / "runs"
NEED_TO_TRAIN_DIR = BASE_DIR / "need_to_train"   # 推論時標記「需要再訓練」的影像集中區

for _d in (DATASETS_DIR, PRETRAINED_DIR, YOLOV26_DIR, RUNS_DIR, NEED_TO_TRAIN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 官方 YOLOv11 預訓練模型（首次使用時 ultralytics 會自動下載到 PRETRAINED_DIR）
PRETRAINED_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
