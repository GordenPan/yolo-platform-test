"""驗證 v2 功能：GPU 自動偵測、平面 zip 上傳（兩種切分模式）、常用參數一級支援。"""
import sys
import time
from pathlib import Path

import requests
import yaml

API = "http://127.0.0.1:8000"
HERE = Path(__file__).parent
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

# 1. GPU 自動偵測
sysinfo = requests.get(f"{API}/api/system").json()
step("system info", "cuda" in sysinfo, f"-> {sysinfo}")

# 2. 平面 zip + train/val 同一資料夾 + 自訂類別名稱
with open(HERE / "train_pic1_flat.zip", "rb") as f:
    r = requests.post(f"{API}/api/datasets/upload",
                      data={"name": "pic1_same", "split": "same",
                            "classes": "crack_label, label_2"},
                      files={"file": ("flat.zip", f, "application/zip")}, timeout=300)
info = r.json() if r.ok else {}
step("upload flat zip (same)", r.ok and info.get("split") == "same_folder"
     and info.get("classes") == ["crack_label", "label_2"], f"-> {info}")
d = yaml.safe_load(open(info["yaml"], encoding="utf-8"))
step("data.yaml train==val", d["train"] == d["val"] == "images")

# 3. 平面 zip + 自動切分
with open(HERE / "train_pic1_flat.zip", "rb") as f:
    r = requests.post(f"{API}/api/datasets/upload",
                      data={"name": "pic1_split", "split": "auto", "val_ratio": 0.2,
                            "classes": "crack_label, label_2"},
                      files={"file": ("flat.zip", f, "application/zip")}, timeout=300)
info = r.json() if r.ok else {}
step("upload flat zip (auto split)", r.ok and info.get("split") == "separated", f"-> {info}")
root = Path(info["yaml"]).parent
n_train = len((root / "train.txt").read_text(encoding="utf-8").splitlines())
n_val = len((root / "val.txt").read_text(encoding="utf-8").splitlines())
step("split files", n_train == 75 and n_val == 18, f"(train={n_train}, val={n_val})")

# 4. 用一級參數啟動訓練（模擬使用者 train.py 的設定，1 epoch 快速驗證）
r = requests.post(f"{API}/api/train", json={
    "model": "yolo11n.pt", "dataset": "pic1_same", "name": "train_a027_test",
    "epochs": 1, "imgsz": 320, "batch": 2,
    "pretrained": False, "patience": 10000, "save_period": 100, "close_mosaic": 3000,
    "degrees": 15, "flipud": 0, "fliplr": 0,
    "extra": {"mosaic": 0, "scale": 0.1, "erasing": 0},
})
step("train start (named run)", r.ok, r.text if not r.ok else "")
task_id = r.json()["task_id"]

deadline = time.time() + 900
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{task_id}").json()
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(8)
if t["status"] == "failed":
    print(t["error"])
step("train completed", t["status"] == "completed")

run_dir = Path(t["run_dir"])
step("run dir named", run_dir.name.startswith("train_a027_test"), f"-> {run_dir.name}")

args = yaml.safe_load(open(run_dir / "args.yaml", encoding="utf-8"))
checks = {"pretrained": False, "patience": 10000, "save_period": 100, "close_mosaic": 3000,
          "degrees": 15, "flipud": 0, "fliplr": 0, "mosaic": 0, "scale": 0.1, "imgsz": 320}
bad = {k: (args.get(k), v) for k, v in checks.items() if args.get(k) != v}
step("args.yaml params", not bad, f"-> 不符: {bad}" if bad else "")
gpu_used = str(args.get("device")) in ("0", "cuda:0") or sysinfo["device"] == "cpu"
step("auto GPU device", gpu_used, f"(args device = {args.get('device')})")

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
