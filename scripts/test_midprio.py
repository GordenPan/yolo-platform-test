"""驗證中優先四項：覆蓋確認(409)、刪除/磁碟用量、（套餐與紅綠燈為純前端，另由 AppTest 驗）。"""
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

# ---------- item 4: 覆蓋確認 ----------
# 先建一個切分式註冊（會在 datasets/ 下建立可被覆蓋的目錄）
r = requests.post(f"{API}/api/datasets/register",
                  data={"folder": SRC, "name": "mid_ovr", "split": "auto",
                        "classes": "crack_label, label_2"})
step("register first time", r.ok, r.text if not r.ok else "")

# 同名再註冊、未確認 → 409
r = requests.post(f"{API}/api/datasets/register",
                  data={"folder": SRC, "name": "mid_ovr", "split": "same"})
step("re-register without overwrite -> 409", r.status_code == 409, f"(status={r.status_code})")

# 帶 overwrite=true → 成功且切分模式變 same
r = requests.post(f"{API}/api/datasets/register",
                  data={"folder": SRC, "name": "mid_ovr", "split": "same", "overwrite": "true"})
step("re-register with overwrite -> ok", r.ok and r.json().get("split") == "same_folder",
     r.json() if r.ok else r.text)

# ---------- item 5: 磁碟用量 ----------
r = requests.get(f"{API}/api/storage")
su = r.json()
step("storage usage", r.ok and all(k in su for k in
     ("datasets_mb", "runs_mb", "need_to_train_mb")), f"-> {su}")

# runs 列表
r = requests.get(f"{API}/api/runs")
runs = r.json()
step("runs list", r.ok and isinstance(runs, list), f"(count={len(runs)})")

# ---------- item 5: 刪除資料集（只刪註冊，不碰來源）----------
src_before = len(list(Path(SRC, "images").glob("*.png")))
r = requests.delete(f"{API}/api/datasets/mid_ovr")
step("delete dataset", r.ok, r.json() if r.ok else r.text)
step("dataset gone from list",
     "mid_ovr" not in [d["name"] for d in requests.get(f"{API}/api/datasets").json()])
src_after = len(list(Path(SRC, "images").glob("*.png")))
step("source images untouched (93->93)", src_before == src_after == 93,
     f"({src_before}->{src_after})")

# 防穿越：帶 .. 的名稱應被擋（FastAPI 路由通常會正規化，這裡測 API 內保護）
r = requests.delete(f"{API}/api/datasets/..%2F..%2Fwindows")
step("path traversal blocked", r.status_code in (400, 404), f"(status={r.status_code})")

# ---------- item 5: need_to_train 管理 ----------
# 先放一張進去
imgs = sorted(Path(SRC, "images").glob("*.png"))
requests.post(f"{API}/api/mark_for_training", data={"image_path": str(imgs[0])})
r = requests.get(f"{API}/api/need_to_train")
ntt = r.json()
step("need_to_train list", r.ok and ntt["count"] >= 1, f"(count={ntt['count']})")
# 清空
r = requests.delete(f"{API}/api/need_to_train")
step("clear need_to_train", r.ok, r.json() if r.ok else r.text)
step("need_to_train empty after clear",
     requests.get(f"{API}/api/need_to_train").json()["count"] == 0)

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
