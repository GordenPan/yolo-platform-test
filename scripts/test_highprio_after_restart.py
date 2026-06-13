"""重啟後驗證：訓練紀錄從 .tasks json 還原、手動拷入的舊 run 被合成 completed。"""
import sys
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
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

last_id = (Path(__file__).parent / ".last_task_id").read_text(encoding="utf-8").strip()
tasks = requests.get(f"{API}/api/train").json()
by_id = {t["id"]: t for t in tasks}

# 1. 重啟前取消的任務仍在，且狀態保持 cancelled
step("cancelled task restored", last_id in by_id
     and by_id[last_id]["status"] == "cancelled",
     f"(status={by_id.get(last_id, {}).get('status')})")

# 2. metrics_history 也還原了（可畫圖）
step("metrics_history restored", last_id in by_id
     and len(by_id[last_id]["metrics_history"]) >= 1)

# 3. 使用者手動拷入的舊 run（無 .tasks json）被合成為 completed
syn = by_id.get("train_cd43_cam2_0430")
step("manual run synthesized as completed",
     syn is not None and syn["status"] == "completed",
     f"(found={syn is not None}, status={syn['status'] if syn else 'n/a'}, "
     f"epochs={syn['total_epochs'] if syn else '?'})")

print("\n[重啟後] " + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
