"""驗證資料夾註冊：兩種切分模式、指向 images 子資料夾、錯誤路徑、實際訓練。"""
import sys
import time
from pathlib import Path

import requests
import yaml

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

# 1. 註冊（auto 切分、留空名稱 -> 用資料夾名）
r = requests.post(f"{API}/api/datasets/register",
                  data={"folder": SRC, "split": "auto", "val_ratio": 0.2,
                        "classes": "crack_label, label_2"})
info = r.json() if r.ok else {}
step("register auto", r.ok and info.get("name") == "Train_pic1"
     and info.get("classes") == ["crack_label", "label_2"], f"-> {info}")
d = yaml.safe_load(open(info["yaml"], encoding="utf-8"))
n_train = len(Path(d["train"]).read_text(encoding="utf-8").splitlines())
n_val = len(Path(d["val"]).read_text(encoding="utf-8").splitlines())
step("split lists (75/18)", n_train == 75 and n_val == 18, f"(train={n_train}, val={n_val})")
step("no files written to source", not (Path(SRC) / "data.yaml").exists()
     and not (Path(SRC) / "train.txt").exists())

# 2. 註冊（same 模式 + 自訂名稱）
r = requests.post(f"{API}/api/datasets/register",
                  data={"folder": SRC, "name": "pic1_full", "split": "same",
                        "classes": "crack_label, label_2"})
info2 = r.json() if r.ok else {}
step("register same", r.ok and info2.get("split") == "same_folder", f"-> {info2}")
d2 = yaml.safe_load(open(info2["yaml"], encoding="utf-8"))
step("same: path -> source folder", d2["path"] == SRC and d2["train"] == "images")

# 3. 直接指向 images 子資料夾（應自動取上層為根目錄）
r = requests.post(f"{API}/api/datasets/register",
                  data={"folder": SRC + r"\images", "name": "pic1_imgdir", "split": "same"})
info3 = r.json() if r.ok else {}
d3 = yaml.safe_load(open(info3["yaml"], encoding="utf-8")) if r.ok else {}
step("register images subdir", r.ok and d3.get("path") == SRC, f"(path={d3.get('path')})")

# 4. 錯誤路徑回 400
r = requests.post(f"{API}/api/datasets/register", data={"folder": r"D:\not_exist_xyz"})
step("400 on bad folder", r.status_code == 400)

# 5. 用 auto 切分的註冊資料集實際訓練 1 epoch（驗證絕對路徑 txt 清單可用）
r = requests.post(f"{API}/api/train", json={
    "model": "yolo11n.pt", "dataset": "Train_pic1", "name": "reg_test",
    "epochs": 1, "imgsz": 320, "batch": 4})
step("train start", r.ok, r.text if not r.ok else "")
task_id = r.json()["task_id"]
deadline = time.time() + 600
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{task_id}").json()
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(8)
if t["status"] == "failed":
    print(t["error"])
step("train on registered folder", t["status"] == "completed")

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
