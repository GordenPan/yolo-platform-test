"""訓練任務管理：背景執行緒跑 ultralytics 訓練，透過 callback 即時回報進度。

任務狀態會落地到 RUNS_DIR/.tasks/<id>.json，後端重啟後自動重建；
使用者手動拷入 runs/ 的舊訓練成果（只有 args.yaml/results.csv）也會被掃描成
completed 任務，以便在「訓練監控」與「模型庫」看到。
"""
from __future__ import annotations

import csv
import json
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import RUNS_DIR
from .registry import resolve_model

TASKS_DIR = RUNS_DIR / ".tasks"   # 任務狀態索引（run 資料夾建立前就需要落地）
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# pending | running | completed | failed | cancelled | interrupted
_ACTIVE = ("pending", "running")


@dataclass
class TrainTask:
    id: str
    params: dict
    status: str = "pending"
    epoch: int = 0
    total_epochs: int = 0
    metrics_history: list = field(default_factory=list)
    error: str | None = None
    hint: str | None = None          # 失敗時的白話原因提示
    run_dir: str | None = None
    created_at: float = field(default_factory=time.time)
    cancel_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "metrics_history": self.metrics_history,
            "error": self.error,
            "hint": self.hint,
            "run_dir": self.run_dir,
            "params": self.params,
            "created_at": self.created_at,
            "cancel_requested": self.cancel_requested,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainTask":
        task = cls(id=d["id"], params=d.get("params", {}))
        task.status = d.get("status", "completed")
        task.epoch = d.get("epoch", 0)
        task.total_epochs = d.get("total_epochs", 0)
        task.metrics_history = d.get("metrics_history", [])
        task.error = d.get("error")
        task.hint = d.get("hint")
        task.run_dir = d.get("run_dir")
        task.created_at = d.get("created_at", time.time())
        task.cancel_requested = d.get("cancel_requested", False)
        return task


_tasks: dict[str, TrainTask] = {}
_lock = threading.Lock()


def auto_device() -> str:
    """自動偵測訓練裝置：有 CUDA GPU 就用 GPU，否則 CPU。"""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _save_task(task: TrainTask) -> None:
    """把任務狀態寫到 .tasks/<id>.json（容錯：寫檔失敗不影響訓練）。"""
    try:
        with open(TASKS_DIR / f"{task.id}.json", "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, ensure_ascii=False)
    except Exception:
        pass


def active_task_id() -> str | None:
    """回傳目前進行中（pending/running）的任務 id，沒有則 None。"""
    return next((tid for tid, t in _tasks.items() if t.status in _ACTIVE), None)


def start_training(params: dict) -> str:
    task = TrainTask(id="", params=params, total_epochs=int(params["epochs"]))
    with _lock:
        aid = active_task_id()
        if aid:
            raise RuntimeError(f"已有訓練進行中（{aid}）。GPU 一次只跑一個訓練，"
                               "請等它完成或先到「訓練監控」取消。")
        task.id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        _tasks[task.id] = task
    _save_task(task)
    threading.Thread(target=_run, args=(task,), daemon=True).start()
    return task.id


def cancel_task(task_id: str) -> bool:
    """要求取消訓練。只在進行中的任務有效；實際停止會在當前 epoch 結束時生效。"""
    task = _tasks.get(task_id)
    if task is None or task.status not in _ACTIVE:
        return False
    task.cancel_requested = True
    _save_task(task)
    return True


def resume_training(task_id: str) -> str:
    """續訓：載入上次的 last.pt 當起始權重，再訓練剩餘的 epochs（開新 run）。

    不用 ultralytics 的 resume=True——取消時 last.pt 會被標記為「已完成」而拒絕 resume；
    改以續訓權重重新開訓，對使用者「繼續訓練」的意圖更穩健（OOM 失敗也適用）。
    """
    old = _tasks.get(task_id)
    if old is None:
        raise FileNotFoundError("找不到任務")
    aid = active_task_id()
    if aid:
        raise RuntimeError(f"已有訓練進行中（{aid}）。GPU 一次只跑一個訓練，"
                           "請等它完成或先取消。")
    if old.status not in ("cancelled", "interrupted", "failed"):
        raise ValueError("只有已中斷／取消／失敗的任務可以續訓")
    if not old.run_dir:
        raise ValueError("此任務沒有對應的 run 資料夾，無法續訓")
    last_pt = Path(old.run_dir) / "weights" / "last.pt"
    if not last_pt.is_file():
        raise ValueError(f"找不到續訓檢查點: {last_pt}")
    if not old.params.get("data"):
        raise ValueError("此任務缺少資料集資訊，無法續訓")

    remaining = max(1, int(old.total_epochs) - int(old.epoch))
    new_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    new_params = {**old.params, "epochs": remaining, "resumed_from": task_id,
                  "name": (old.params.get("name") or old.id) + "_resume"}
    new = TrainTask(id=new_id, params=new_params, total_epochs=remaining)
    with _lock:
        _tasks[new_id] = new
    _save_task(new)
    threading.Thread(target=_run, args=(new,), kwargs={"resume_from": str(last_pt)},
                     daemon=True).start()
    return new_id


def get_task(task_id: str) -> dict | None:
    task = _tasks.get(task_id)
    return task.to_dict() if task else None


def delete_task(task_id: str) -> bool:
    """刪除一筆訓練任務紀錄：移除記憶體任務 + .tasks json + 其 run 資料夾
    （若在 RUNS_DIR 內）。連 run 一起刪是為了避免重啟後又被掃描合成回來。
    進行中的任務不可刪——請先取消。"""
    task = _tasks.get(task_id)
    if task and task.status in _ACTIVE:
        raise ValueError("任務進行中，請先取消再刪除")
    _tasks.pop(task_id, None)
    try:
        (TASKS_DIR / f"{task_id}.json").unlink(missing_ok=True)
    except OSError:
        pass
    if task and task.run_dir:
        try:
            rd = Path(task.run_dir).resolve()
            if RUNS_DIR.resolve() in rd.parents:
                shutil.rmtree(rd, ignore_errors=True)
        except OSError:
            pass
    return task is not None


def purge_run(run_name: str) -> None:
    """刪除某個 run 資料夾後，清掉對應的記憶體任務與 .tasks 索引 json。"""
    to_del = [tid for tid, t in _tasks.items()
              if tid == run_name or (t.run_dir and Path(t.run_dir).name == run_name)]
    for tid in to_del:
        _tasks.pop(tid, None)
        try:
            (TASKS_DIR / f"{tid}.json").unlink(missing_ok=True)
        except OSError:
            pass


def list_tasks() -> list[dict]:
    return [t.to_dict() for t in sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)]


def _run(task: TrainTask, resume_from: str | None = None) -> None:
    task.status = "running"
    _save_task(task)
    try:
        from ultralytics import YOLO  # 延遲載入，避免拖慢 API 啟動

        def on_fit_epoch_end(trainer):
            # 取消：在 epoch 邊界設 trainer.stop，ultralytics 會優雅中止並保留權重
            if task.cancel_requested:
                trainer.stop = True
            task.epoch = trainer.epoch + 1
            row = {"epoch": task.epoch}
            row.update({k: round(float(v), 5) for k, v in trainer.metrics.items()})
            task.metrics_history.append(row)
            _save_task(task)

        # 續訓時用 last.pt 當起始權重；一般訓練用選定的模型
        model = YOLO(resume_from) if resume_from else YOLO(resolve_model(task.params["model"]))
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

        task.run_dir = str(results.save_dir) if results is not None else task.run_dir
        task.status = "cancelled" if task.cancel_requested else "completed"
    except Exception:
        task.status = "failed"
        task.error = traceback.format_exc()
        task.hint = _failure_hint(task.params.get("model", ""), task.error)
    _save_task(task)


def _failure_hint(model_id: str, tb: str) -> str | None:
    """把常見的失敗原因翻成白話提示。"""
    low = tb.lower()
    if "out of memory" in low or "cuda out of memory" in low:
        return ("GPU 記憶體不足（batch 或影像尺寸過大）。請調小 Batch size／影像尺寸，"
                "或勾選「自動 batch」後重新訓練。系統本身不受影響。")
    # YOLOv26 等新架構在舊版 ultralytics 無法解析（模組簽章不符）
    arch_sig = ("positional argument" in low and "parse_model" in low) or \
               ("yolo26" in model_id.lower() or model_id.startswith("YOLOV26/"))
    if arch_sig and ("parse_model" in low or "positional argument" in low):
        import ultralytics
        return (f"此模型的架構無法被目前的 ultralytics {ultralytics.__version__} 解析"
                "（常見於 YOLOv26 等較新架構）。請先升級套件：`pip install -U ultralytics`，"
                "再重新訓練；升級前請留意是否影響既有環境。")
    return None


# ---------- 啟動時重建任務（後端重啟不消失） ----------

def _restore_tasks() -> None:
    """從 .tasks/*.json 重建任務；上次中斷的標 interrupted；
    再掃 runs/ 補上只有 args.yaml/results.csv 的舊 run（使用者手動拷入的）。"""
    seen_run_dirs: set[str] = set()

    for jf in sorted(TASKS_DIR.glob("*.json")):
        try:
            with open(jf, encoding="utf-8") as f:
                task = TrainTask.from_dict(json.load(f))
        except Exception:
            continue
        # 後端重啟前還在跑的任務不可能還活著 → 標為中斷
        if task.status in _ACTIVE:
            task.status = "interrupted"
            _save_task(task)
        _tasks[task.id] = task
        if task.run_dir:
            seen_run_dirs.add(str(Path(task.run_dir).resolve()))

    if not RUNS_DIR.exists():
        return
    for run in sorted(RUNS_DIR.iterdir()):
        if not run.is_dir() or run.name == ".tasks":
            continue
        if str(run.resolve()) in seen_run_dirs:
            continue
        if not (run / "results.csv").exists() and not (run / "args.yaml").exists():
            continue
        # 合成一筆 completed 任務（使用者手動拷入或舊版未落地的 run）
        task = TrainTask(id=run.name, params=_args_summary(run))
        task.status = "completed"
        task.run_dir = str(run)
        task.created_at = run.stat().st_mtime
        task.metrics_history = _metrics_from_csv(run / "results.csv")
        task.epoch = task.metrics_history[-1]["epoch"] if task.metrics_history else 0
        task.total_epochs = task.epoch
        _tasks[task.id] = task


def _args_summary(run: Path) -> dict:
    """從 args.yaml 取出顯示用的 model/data（失敗則留空）。"""
    args_file = run / "args.yaml"
    if not args_file.exists():
        return {"model": "?", "data": "?"}
    try:
        import yaml

        with open(args_file, encoding="utf-8") as f:
            args = yaml.safe_load(f) or {}
        return {"model": Path(str(args.get("model", "?"))).name, "data": args.get("data", "?")}
    except Exception:
        return {"model": "?", "data": "?"}


def _metrics_from_csv(csv_path: Path) -> list[dict]:
    """從 results.csv 重建 metrics_history（供圖表顯示）。"""
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    history = []
    for i, row in enumerate(rows, 1):
        clean = {k.strip(): v.strip() for k, v in row.items() if k}
        rec = {"epoch": int(float(clean.get("epoch", i)))}
        for k, v in clean.items():
            if k.startswith("metrics/") or "loss" in k:
                try:
                    rec[k] = round(float(v), 5)
                except (ValueError, TypeError):
                    pass
        history.append(rec)
    return history


_restore_tasks()
