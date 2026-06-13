"""驗證五點修正：
  1 自選刪除 need_to_train、3 續訓、4 imgsz 邊界、5 batch=-1(AutoBatch) 接受、
  2 資料夾可用性判斷(_folder_usability，無影像提醒的核心邏輯)。
原生選擇視窗本身會開 GUI，無法無頭測，故只測其判斷邏輯。
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

# ---------- point 2: 資料夾可用性判斷（無影像提醒）----------
sys.path.insert(0, str(Path(__file__).parents[1]))
from backend.main import _folder_usability  # noqa: E402

u_imgs = _folder_usability(Path(SRC, "images"))   # 直接放影像
u_root = _folder_usability(Path(SRC))             # 有 images/ 子資料夾
u_empty = _folder_usability(Path(SRC, "labels"))  # 只有 txt，無影像
step("usable: images dir", u_imgs["usable"] and u_imgs["image_count"] == 93, f"-> {u_imgs}")
step("usable: root with images/", u_root["usable"], f"-> {u_root['reason']}")
step("not usable: no images -> warns", not u_empty["usable"], f"-> {u_empty['reason']}")

# ---------- point 4: imgsz 邊界 / point 5: batch=-1 ----------
requests.post(f"{API}/api/datasets/register",
              data={"folder": SRC, "name": "fix_test", "split": "same",
                    "classes": "crack_label, label_2"})

# imgsz < 100 應被 schema 擋（422）
r = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "fix_test",
                                            "epochs": 1, "imgsz": 50})
step("imgsz<100 rejected (422)", r.status_code == 422, f"(status={r.status_code})")

# imgsz=200（非 32 倍數但在範圍內）應被接受
r = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "fix_test",
                                            "name": "fix_imgsz200", "epochs": 1,
                                            "imgsz": 200, "batch": 4})
step("imgsz=200 accepted", r.ok, r.text if not r.ok else "")
if r.ok:
    tid = r.json()["task_id"]
    for _ in range(60):
        t = requests.get(f"{API}/api/train/{tid}").json()
        if t["status"] in ("completed", "failed"):
            break
        time.sleep(3)
    step("imgsz=200 trained ok", t["status"] == "completed", f"(status={t['status']})")

# batch=-1（AutoBatch）schema 接受
r = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "fix_test",
                                            "name": "fix_autobatch", "epochs": 1,
                                            "imgsz": 160, "batch": -1})
step("batch=-1 (AutoBatch) accepted by schema", r.ok, r.text if not r.ok else "")
if r.ok:
    tid_ab = r.json()["task_id"]
    for _ in range(80):
        t = requests.get(f"{API}/api/train/{tid_ab}").json()
        if t["status"] in ("completed", "failed"):
            break
        time.sleep(3)
    step("AutoBatch run finished (completed)", t["status"] == "completed",
         f"(status={t['status']})")

# ---------- point 3: 續訓 ----------
r = requests.post(f"{API}/api/train", json={"model": "yolo11n.pt", "dataset": "fix_test",
                                            "name": "fix_resume", "epochs": 8,
                                            "imgsz": 160, "batch": 4})
tid_r = r.json()["task_id"]
for _ in range(60):
    t = requests.get(f"{API}/api/train/{tid_r}").json()
    if t["epoch"] >= 2 or t["status"] in ("completed", "failed"):
        break
    time.sleep(2)
cancel_epoch = t["epoch"]
requests.post(f"{API}/api/train/{tid_r}/cancel")
for _ in range(40):
    t = requests.get(f"{API}/api/train/{tid_r}").json()
    if t["status"] == "cancelled":
        break
    time.sleep(2)
step("cancelled for resume test", t["status"] == "cancelled", f"(@epoch {cancel_epoch})")

# 續訓
r = requests.post(f"{API}/api/train/{tid_r}/resume")
step("resume accepted", r.ok, r.json() if r.ok else r.text)
if r.ok:
    new_id = r.json()["task_id"]
    for _ in range(80):
        t = requests.get(f"{API}/api/train/{new_id}").json()
        if t["status"] in ("completed", "failed"):
            break
        time.sleep(3)
    if t["status"] == "failed":
        print("   resume error:", t["error"][-300:] if t["error"] else "")
    step("resumed run completed (剩餘 epochs)",
         t["status"] == "completed" and t["epoch"] == t["total_epochs"] >= 1,
         f"(status={t['status']}, epoch={t['epoch']}/{t['total_epochs']})")

# 已完成的任務不能續訓（400）
r = requests.post(f"{API}/api/train/{tid_r}/resume")
# tid_r 仍是 cancelled，可續訓；改測一個不存在的
r2 = requests.post(f"{API}/api/train/nonexistent_xyz/resume")
step("resume nonexistent -> 404", r2.status_code == 404, f"(status={r2.status_code})")

# ---------- point 1: 自選刪除 need_to_train ----------
imgs = sorted(Path(SRC, "images").glob("*.png"))
for im in imgs[:3]:
    requests.post(f"{API}/api/mark_for_training", data={"image_path": str(im)})
listing = requests.get(f"{API}/api/need_to_train").json()
step("marked 3 files", listing["count"] >= 3, f"(count={listing['count']})")
one = listing["files"][0]["name"]
r = requests.delete(f"{API}/api/need_to_train", params={"name": one})
step("delete single file", r.ok and r.json().get("deleted") == one, r.json() if r.ok else r.text)
after = requests.get(f"{API}/api/need_to_train").json()
step("file count decreased by 1", after["count"] == listing["count"] - 1,
     f"({listing['count']} -> {after['count']})")

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
