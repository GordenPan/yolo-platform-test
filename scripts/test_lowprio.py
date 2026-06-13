"""驗證低優先：item 11 yolo26 列入預訓練清單、可解析、可實際訓練（n 版）。

install.bat / docs / 完成通知為非 API 項目，另以檔案存在與 AppTest 渲染驗證。
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

# item 11: yolo26 出現在模型清單（pretrained 類型）
models = requests.get(f"{API}/api/models").json()
pre_ids = [m["id"] for m in models if m["type"] == "pretrained"]
step("yolo26 n/s/m/l/x 列入官方預訓練",
     all(f"yolo26{s}.pt" in pre_ids for s in "nsmlx"), f"-> {pre_ids}")

# 實際用 yolo26n 訓練 1 epoch（環境已是 8.4.66，應成功；首次會自動下載到 pretrained/）
requests.post(f"{API}/api/datasets/register",
              data={"folder": SRC, "name": "low_test", "split": "same",
                    "classes": "crack_label, label_2"})
r = requests.post(f"{API}/api/train", json={
    "model": "yolo26n.pt", "dataset": "low_test", "name": "low_yolo26_test",
    "epochs": 1, "imgsz": 160, "batch": 4})
step("yolo26n train accepted", r.ok, r.text if not r.ok else "")
tid = r.json()["task_id"]
deadline = time.time() + 300
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{tid}").json()
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(4)
if t["status"] == "failed":
    print("   err:", (t.get("hint") or t.get("error", ""))[:200])
step("yolo26n trained ok (8.4.66)", t["status"] == "completed", f"(status={t['status']})")

# 自動下載到 pretrained/
step("yolo26n.pt downloaded to pretrained/",
     (Path(__file__).parents[1] / "models" / "pretrained" / "yolo26n.pt").exists())

# 非 API 檔案存在性
root = Path(__file__).parents[1]
step("install.bat exists", (root / "install.bat").is_file())
step("docs/manual.md exists", (root / "docs" / "manual.md").is_file())

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
