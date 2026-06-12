"""驗證進階參數透傳：用使用者 train.py 的設定跑 1 epoch，檢查 args.yaml。"""
import sys
import time
from pathlib import Path

import requests
import yaml

API = "http://127.0.0.1:8000"

for _ in range(30):
    try:
        if requests.get(f"{API}/api/health", timeout=2).ok:
            break
    except requests.ConnectionError:
        time.sleep(1)

r = requests.post(f"{API}/api/train", json={
    "model": "yolo11s.pt", "dataset": "train_pic1",
    "epochs": 1, "imgsz": 960, "batch": 2, "device": "0",
    "extra": {"mosaic": 0, "degrees": 15, "scale": 0.1, "flipud": 0, "fliplr": 0,
              "erasing": 0, "patience": 10000, "pretrained": False},
})
r.raise_for_status()
task_id = r.json()["task_id"]
print("task:", task_id)

deadline = time.time() + 900
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{task_id}").json()
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(8)

if t["status"] != "completed":
    print("FAIL:", t["status"], t.get("error"))
    sys.exit(1)

args = yaml.safe_load(open(Path(t["run_dir"]) / "args.yaml", encoding="utf-8"))
checks = {"mosaic": 0, "degrees": 15, "scale": 0.1, "fliplr": 0, "patience": 10000,
          "pretrained": False, "imgsz": 960, "batch": 2}
bad = {k: (args.get(k), v) for k, v in checks.items() if args.get(k) != v}
if bad:
    print("FAIL: args.yaml 不符 ->", bad)
    sys.exit(1)
print("PASS: 進階參數全部正確寫入 args.yaml")
