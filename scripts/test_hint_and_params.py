"""驗證：1 監控參數齊全、2 YOLOv26 在舊版 ultralytics 失敗時給友善 hint。"""
import os
import sys
import time
from pathlib import Path

import requests

API = os.environ.get("YOLO_API_URL", "http://127.0.0.1:8000")
SRC = r"D:\Project\a027\Train_pic1"
FAILS = []


def step(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name} {detail}")
    if not ok:
        FAILS.append(name)


for _ in range(30):
    try:
        if requests.get(f"{API}/api/health", timeout=2).ok:
            break
    except requests.ConnectionError:
        time.sleep(1)

# 單元測試 _failure_hint（OOM）
sys.path.insert(0, str(Path(__file__).parents[1]))
from backend.services.trainer import _failure_hint  # noqa: E402

step("hint: OOM", "記憶體不足" in (_failure_hint("yolo11n.pt",
     "RuntimeError: CUDA out of memory. Tried to allocate...") or ""))
step("hint: yolo26 parse",
     "ultralytics" in (_failure_hint("YOLOV26/yolo26n.pt",
     "TypeError: SPPF.__init__() ... positional argument ... in parse_model") or ""))
step("hint: normal -> None", _failure_hint("yolo11n.pt", "some unrelated error") is None)

# 準備資料集
requests.post(f"{API}/api/datasets/register",
              data={"folder": SRC, "name": "hint_test", "split": "same",
                    "classes": "crack_label, label_2"})

# 1. 一般訓練 1 epoch，檢查 params 是否完整回傳（監控顯示用）
r = requests.post(f"{API}/api/train", json={
    "model": "yolo11n.pt", "dataset": "hint_test", "name": "param_show_test",
    "epochs": 1, "imgsz": 160, "batch": 4, "patience": 77, "degrees": 12})
tid = r.json()["task_id"]
t = requests.get(f"{API}/api/train/{tid}").json()
p = t["params"]
step("params 含 dataset/strategy 欄位",
     all(k in p for k in ("data", "model", "epochs", "imgsz", "batch",
                          "pretrained", "patience", "close_mosaic", "degrees")),
     f"(patience={p.get('patience')}, degrees={p.get('degrees')})")
step("params 值正確帶入", p["patience"] == 77 and p["degrees"] == 12)

# 2. 用 YOLOV26/yolo26n.pt 訓練 → 應失敗並帶友善 hint
r = requests.post(f"{API}/api/train", json={
    "model": "YOLOV26/yolo26n.pt", "dataset": "hint_test", "name": "yolo26_fail_test",
    "epochs": 1, "imgsz": 160, "batch": 4})
step("yolo26 train accepted (will fail in training)", r.ok, r.text if not r.ok else "")
tid26 = r.json()["task_id"]
deadline = time.time() + 180
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{tid26}").json()
    if t["status"] in ("failed", "completed"):
        break
    time.sleep(3)
step("yolo26 -> failed", t["status"] == "failed", f"(status={t['status']})")
step("yolo26 has friendly hint (建議升級 ultralytics)",
     t.get("hint") and "ultralytics" in t["hint"], f"-> hint={t.get('hint')}")

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
