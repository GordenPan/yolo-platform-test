"""驗證：item 1 訓練圖表端點、item 2 同時只允許一個訓練（防 OOM）。
item 3（標註圖 ZIP）是純前端 download_button，另由 AppTest 渲染驗證不拋例外。"""
import sys
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
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

requests.post(f"{API}/api/datasets/register",
              data={"folder": SRC, "name": "pg_test", "split": "same",
                    "classes": "crack_label, label_2"})

# ---------- item 2: 同時只允許一個訓練 ----------
r1 = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "pg_test",
                                             "name": "pg_train1", "epochs": 30,
                                             "imgsz": 160, "batch": 4})
step("第一個訓練啟動", r1.ok, r1.text if not r1.ok else "")
tid1 = r1.json()["task_id"]
for _ in range(30):
    if requests.get(f"{API}/api/train/{tid1}").json()["status"] == "running":
        break
    time.sleep(1)

# 第二個應被擋（409）
r2 = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "pg_test",
                                             "name": "pg_train2", "epochs": 1,
                                             "imgsz": 160, "batch": 4})
step("第二個訓練被擋 -> 409", r2.status_code == 409, f"(status={r2.status_code}, msg={r2.json().get('detail','')[:40]})")

# 取消第一個，等結束（保留一個 completed/cancelled 的 run 供圖表測試）
requests.post(f"{API}/api/train/{tid1}/cancel")
for _ in range(60):
    s = requests.get(f"{API}/api/train/{tid1}").json()["status"]
    if s not in ("pending", "running"):
        break
    time.sleep(3)

# 取消後應能再啟動（守衛解除）
r3 = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "pg_test",
                                             "name": "pg_done", "epochs": 1,
                                             "imgsz": 160, "batch": 4})
step("取消後可再啟動", r3.ok, r3.text if not r3.ok else "")
tid3 = r3.json()["task_id"]
for _ in range(80):
    t = requests.get(f"{API}/api/train/{tid3}").json()
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(3)
step("第二輪訓練完成", t["status"] == "completed", f"(status={t['status']})")
run_name = Path(t["run_dir"]).name

# ---------- item 1: 訓練圖表端點 ----------
r = requests.get(f"{API}/api/runs/{run_name}/plots")
plots = r.json().get("plots", [])
step("plots 端點列出圖檔", r.ok and any(p.endswith(".png") for p in plots),
     f"(n={len(plots)})")
step("含混淆矩陣/PR/results", any("confusion" in p for p in plots)
     and any("PR_curve" in p for p in plots) and "results.png" in plots,
     f"-> {[p for p in plots if 'confusion' in p or 'PR' in p or p=='results.png']}")

# 取一張圖檔（應回 image bytes）
if plots:
    png = next(p for p in plots if p.endswith(".png"))
    r = requests.get(f"{API}/api/runs/{run_name}/file", params={"file": png})
    step("file 端點回傳圖片", r.ok and r.headers.get("content-type", "").startswith("image"),
         f"(type={r.headers.get('content-type')}, bytes={len(r.content)})")

# 防穿越
r = requests.get(f"{API}/api/runs/{run_name}/file", params={"file": "../../config.py"})
step("file 防穿越 -> 400/404", r.status_code in (400, 404), f"(status={r.status_code})")

# 收尾
requests.delete(f"{API}/api/train/{tid1}")
requests.delete(f"{API}/api/train/{tid3}")
requests.delete(f"{API}/api/datasets/pg_test")

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
