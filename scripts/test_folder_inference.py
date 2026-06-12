"""驗證資料夾瀏覽推論：列資料夾、按路徑推論、複製到 need_to_train。"""
import sys
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
FOLDER = r"D:\Project\a027\Train_pic1\images"
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

# 1. 列出資料夾影像
r = requests.get(f"{API}/api/folder/images", params={"path": FOLDER})
data = r.json()
step("folder list", r.ok and data["count"] == 93, f"(count={data.get('count')})")
first = data["images"][0]

# 2. 不存在的資料夾要回 400
r = requests.get(f"{API}/api/folder/images", params={"path": r"D:\not_exist_xyz"})
step("folder 400 on bad path", r.status_code == 400)

# 3. 按路徑推論
r = requests.post(f"{API}/api/predict_path",
                  data={"model": "yolo11n.pt", "image_path": first, "conf": 0.1}, timeout=300)
res = r.json() if r.ok else {}
step("predict by path", r.ok and "annotated_image_b64" in res and "count" in res,
     f"(偵測 {res.get('count')} 個)")

# 4. 複製到 need_to_train（應連同標註檔）
r = requests.post(f"{API}/api/mark_for_training", data={"image_path": first})
res = r.json() if r.ok else {}
copied = Path(res.get("copied_to", ""))
label = copied.parent / "labels" / f"{copied.stem}.txt"
step("mark_for_training", r.ok and copied.is_file(), f"-> {res}")
step("label copied too", res.get("label_copied") is True and label.is_file())

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "全部通過"))
sys.exit(1 if FAILS else 0)
