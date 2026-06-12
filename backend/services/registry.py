"""模型註冊中心。

統一管理三類模型來源，並提供 model_id -> 實際路徑的解析：
  1. pretrained : 官方 YOLOv11 預訓練（yolo11n.pt ...），id 即檔名
  2. trained    : runs/<run>/weights/best.pt，id 為 "runs/<run>"
  3. yolov26    : models/YOLOV26/ 下的 .pt（權重）或 .yaml（架構設定），
                  id 為 "YOLOV26/<檔名>" —— 預留給 Ultralytics 官方的 YOLOv26，
                  發布後升級 ultralytics 套件、放入權重即可使用，無需改程式
"""
from __future__ import annotations

import csv

from ..core.config import PRETRAINED_DIR, PRETRAINED_MODELS, RUNS_DIR, YOLOV26_DIR


def list_models() -> list[dict]:
    models: list[dict] = []

    for name in PRETRAINED_MODELS:
        models.append({"id": name, "type": "pretrained", "label": f"官方預訓練｜{name}"})

    for f in sorted(YOLOV26_DIR.iterdir()) if YOLOV26_DIR.exists() else []:
        if f.suffix in (".yaml", ".yml", ".pt"):
            kind = "架構設定" if f.suffix in (".yaml", ".yml") else "預訓練"
            models.append({"id": f"YOLOV26/{f.name}", "type": "yolov26", "label": f"YOLOv26 {kind}｜{f.name}"})

    for run in sorted(RUNS_DIR.iterdir(), reverse=True) if RUNS_DIR.exists() else []:
        best = run / "weights" / "best.pt"
        if best.exists():
            entry = {"id": f"runs/{run.name}", "type": "trained", "label": f"已訓練｜{run.name}"}
            entry["metrics"] = _read_final_metrics(run)
            models.append(entry)

    return models


def resolve_model(model_id: str) -> str:
    """將 model_id 解析為可直接傳給 ultralytics.YOLO() 的路徑。

    YOLOv26 與 YOLOv11 同為 Ultralytics 官方模型，YOLO() 可原生載入，
    因此這裡只做路徑解析，不需要任何自訂載入邏輯。
    """
    if model_id in PRETRAINED_MODELS:
        return str(PRETRAINED_DIR / model_id)
    if model_id.startswith("YOLOV26/"):
        path = YOLOV26_DIR / model_id.split("/", 1)[1]
        if not path.exists():
            raise FileNotFoundError(f"找不到 YOLOv26 模型: {path}")
        return str(path)
    if model_id.startswith("runs/"):
        path = RUNS_DIR / model_id.split("/", 1)[1] / "weights" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"找不到訓練權重: {path}")
        return str(path)
    raise ValueError(f"無法解析的 model_id: {model_id}")


def _read_final_metrics(run_dir) -> dict:
    """從 results.csv 讀取最後一個 epoch 的主要指標（供模型庫顯示）。"""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last = {k.strip(): v.strip() for k, v in rows[-1].items() if k}
        keep = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)")
        return {k: round(float(last[k]), 4) for k in keep if k in last and last[k]}
    except Exception:
        return {}
