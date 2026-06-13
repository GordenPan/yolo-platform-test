"""驗證高優先三項：資料夾瀏覽器、取消訓練、訓練紀錄落地（重啟前半段）。

重啟後的還原驗證由 test_highprio_after_restart.py 接續（需先重啟後端）。
"""
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

# ---------- item 3: 資料夾瀏覽器 ----------
r = requests.get(f"{API}/api/fs/browse", params={"path": ""})
drives = r.json()
step("browse drives", r.ok and any(d.startswith("D:") for d in drives["dirs"]),
     f"-> {drives['dirs']}")

r = requests.get(f"{API}/api/fs/browse", params={"path": SRC})
info = r.json()
step("browse folder (subdirs + image_count)",
     r.ok and any(d.endswith("images") for d in info["dirs"]) and info["parent"],
     f"(dirs={len(info['dirs'])}, imgs={info['image_count']})")

r = requests.get(f"{API}/api/fs/browse", params={"path": SRC + r"\images"})
info2 = r.json()
step("browse images subdir (count=93)", r.ok and info2["image_count"] == 93,
     f"(imgs={info2['image_count']})")

r = requests.get(f"{API}/api/fs/browse", params={"path": r"D:\nonexistent_xyz"})
step("browse bad path 400", r.status_code == 400)

# ---------- item 1: 取消訓練 ----------
# 先確保有資料集
requests.post(f"{API}/api/datasets/register",
              data={"folder": SRC, "name": "hp_test", "split": "same",
                    "classes": "crack_label, label_2"})

r = requests.post(f"{API}/api/train", json={
    "model": "yolo11n.pt", "dataset": "hp_test", "name": "hp_cancel_test",
    "epochs": 50, "imgsz": 320, "batch": 8})
step("train start (50 epochs)", r.ok, r.text if not r.ok else "")
task_id = r.json()["task_id"]

# 等到至少跑完 1 個 epoch（有 metrics）後送取消
got_epoch = False
for _ in range(60):
    t = requests.get(f"{API}/api/train/{task_id}").json()
    if t["epoch"] >= 1:
        got_epoch = True
        break
    if t["status"] in ("failed", "completed"):
        break
    time.sleep(2)
step("reached epoch 1 before cancel", got_epoch, f"(epoch={t['epoch']}, status={t['status']})")

r = requests.post(f"{API}/api/train/{task_id}/cancel")
step("cancel accepted", r.ok, r.json() if r.ok else r.text)

# 等待狀態變 cancelled（應遠早於 50 epochs）
deadline = time.time() + 180
final = None
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{task_id}").json()
    if t["status"] not in ("pending", "running"):
        final = t
        break
    time.sleep(3)
step("status -> cancelled", final and final["status"] == "cancelled",
     f"(status={final['status'] if final else 'timeout'}, stopped@epoch={final['epoch'] if final else '?'}/50)")
step("cancelled early (< 50 epochs)", final and final["epoch"] < 50)

# 取消後權重應保留
best = Path(final["run_dir"]) / "weights" / "best.pt" if final and final["run_dir"] else None
step("weights preserved after cancel", best and best.is_file(), f"-> {best}")

# 不能取消已結束的任務
r = requests.post(f"{API}/api/train/{task_id}/cancel")
step("cancel finished task -> 400", r.status_code == 400)

# ---------- item 2: 落地保存（重啟前半段）----------
jf = Path(SRC).anchor  # dummy
tasks_json = Path(__file__).parents[1] / "runs" / ".tasks" / f"{task_id}.json"
step("task.json written", tasks_json.is_file(), f"-> {tasks_json}")

print("\n[重啟前] " + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
print(f"記住這個 task_id 供重啟後驗證: {task_id}")
(Path(__file__).parent / ".last_task_id").write_text(task_id, encoding="utf-8")
sys.exit(1 if FAILS else 0)
