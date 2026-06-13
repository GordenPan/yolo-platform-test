"""驗證：1 磁碟剩餘空間欄位、2 訓練任務可刪除（含 active 防呆與 run 目錄清除、不復活）。"""
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

# ---------- item 1: 磁碟剩餘空間 ----------
su = requests.get(f"{API}/api/storage").json()
need = ("disk_total_gb", "disk_free_gb", "disk_used_gb", "disk_used_pct", "disk_free_pct")
step("storage 有磁碟總量/剩餘欄位", all(k in su for k in need), f"-> total={su.get('disk_total_gb')}GB "
     f"free={su.get('disk_free_gb')}GB used={su.get('disk_used_pct')}%")
step("剩餘 + 使用 約= 100%", abs(su["disk_used_pct"] + su["disk_free_pct"] - 100) < 1.0)

# ---------- item 2: 任務刪除 ----------
requests.post(f"{API}/api/datasets/register",
              data={"folder": SRC, "name": "del_test", "split": "same",
                    "classes": "crack_label, label_2"})
# 跑一個完成的任務
r = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "del_test",
                                            "name": "task_del_done", "epochs": 1,
                                            "imgsz": 160, "batch": 4})
tid = r.json()["task_id"]
for _ in range(60):
    t = requests.get(f"{API}/api/train/{tid}").json()
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(3)
run_dir = t["run_dir"]
step("任務完成且有 run_dir", t["status"] == "completed" and run_dir and Path(run_dir).exists())

# active 任務不可刪：開一個長訓練，立刻嘗試刪 -> 400
r = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "del_test",
                                            "name": "task_del_active", "epochs": 50,
                                            "imgsz": 160, "batch": 4})
tid_active = r.json()["task_id"]
for _ in range(30):
    if requests.get(f"{API}/api/train/{tid_active}").json()["status"] == "running":
        break
    time.sleep(1)
r = requests.delete(f"{API}/api/train/{tid_active}")
step("刪除進行中任務 -> 400", r.status_code == 400, f"(status={r.status_code})")
requests.post(f"{API}/api/train/{tid_active}/cancel")  # 收尾

# 刪除已完成任務 -> 紀錄消失 + run 目錄一併刪除
r = requests.delete(f"{API}/api/train/{tid}")
step("刪除完成任務 ok", r.ok and r.json().get("deleted") == tid, r.json() if r.ok else r.text)
ids_after = [t["id"] for t in requests.get(f"{API}/api/train").json()]
step("任務從清單消失", tid not in ids_after)
step("run 目錄一併刪除（不會重啟復活）", not Path(run_dir).exists(), f"-> {run_dir}")

# 刪不存在 -> 404
r = requests.delete(f"{API}/api/train/nope_xyz")
step("刪不存在 -> 404", r.status_code == 404)

# 收尾：等 active 取消完，清掉它
for _ in range(40):
    s = requests.get(f"{API}/api/train/{tid_active}").json()["status"]
    if s not in ("pending", "running"):
        break
    time.sleep(3)
requests.delete(f"{API}/api/train/{tid_active}")
requests.delete(f"{API}/api/datasets/del_test")

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
