"""訓練任務管理：背景執行緒跑 ultralytics 訓練，透過 callback 即時回報進度。"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from ..core.config import RUNS_DIR
from .registry import resolve_model


@dataclass
class TrainTask:
    id: str
    params: dict
    status: str = "pending"          # pending | running | completed | failed
    epoch: int = 0
    total_epochs: int = 0
    metrics_history: list = field(default_factory=list)
    error: str | None = None
    run_dir: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "metrics_history": self.metrics_history,
            "error": self.error,
            "run_dir": self.run_dir,
            "params": self.params,
            "created_at": self.created_at,
        }


_tasks: dict[str, TrainTask] = {}
_lock = threading.Lock()


def auto_device() -> str:
    """自動偵測訓練裝置：有 CUDA GPU 就用 GPU，否則 CPU。"""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def start_training(params: dict) -> str:
    task_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    task = TrainTask(id=task_id, params=params, total_epochs=int(params["epochs"]))
    with _lock:
        _tasks[task_id] = task
    threading.Thread(target=_run, args=(task,), daemon=True).start()
    return task_id


def get_task(task_id: str) -> dict | None:
    task = _tasks.get(task_id)
    return task.to_dict() if task else None


def list_tasks() -> list[dict]:
    return [t.to_dict() for t in sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)]


def _run(task: TrainTask) -> None:
    task.status = "running"
    try:
        from ultralytics import YOLO  # 延遲載入，避免拖慢 API 啟動

        model = YOLO(resolve_model(task.params["model"]))

        def on_fit_epoch_end(trainer):
            task.epoch = trainer.epoch + 1
            row = {"epoch": task.epoch}
            row.update({k: round(float(v), 5) for k, v in trainer.metrics.items()})
            task.metrics_history.append(row)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        kwargs = dict(
            data=task.params["data"],
            epochs=int(task.params["epochs"]),
            imgsz=int(task.params.get("imgsz", 640)),
            batch=int(task.params.get("batch", 16)),
            device=task.params.get("device") or auto_device(),
            pretrained=bool(task.params.get("pretrained", True)),
            patience=int(task.params.get("patience", 100)),
            save_period=int(task.params.get("save_period", -1)),
            close_mosaic=int(task.params.get("close_mosaic", 10)),
            degrees=float(task.params.get("degrees", 0.0)),
            flipud=float(task.params.get("flipud", 0.0)),
            fliplr=float(task.params.get("fliplr", 0.5)),
            project=str(RUNS_DIR),
            name=task.params.get("name") or task.id,
            exist_ok=False,  # 同名 run 由 ultralytics 自動編號（name2, name3...）
        )
        # 透傳進階超參數（mosaic/degrees/patience/cache...），但不允許覆蓋核心欄位
        extra = {k: v for k, v in (task.params.get("extra") or {}).items() if k not in kwargs}
        results = model.train(**kwargs, **extra)
        task.run_dir = str(results.save_dir) if results is not None else str(RUNS_DIR / task.id)
        task.status = "completed"
    except Exception:
        task.status = "failed"
        task.error = traceback.format_exc()
