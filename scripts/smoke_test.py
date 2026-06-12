"""端到端冒煙測試：health -> models -> 上傳資料集 -> 訓練 -> 監控 -> 推論。"""
import sys
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
HERE = Path(__file__).parent


def step(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name} {detail}")
    if not ok:
        sys.exit(1)


# 1. health（等待後端就緒）
for _ in range(30):
    try:
        r = requests.get(f"{API}/api/health", timeout=2)
        if r.ok:
            break
    except requests.ConnectionError:
        time.sleep(1)
else:
    step("health", False, "後端 30 秒內未就緒")
step("health", True)

# 2. models
models = requests.get(f"{API}/api/models").json()
ids = [m["id"] for m in models]
step("models", "yolo11n.pt" in ids, f"({len(models)} 個模型)")

# 3. 上傳資料集
with open(HERE / "train_pic1.zip", "rb") as f:
    r = requests.post(f"{API}/api/datasets/upload",
                      data={"name": "train_pic1"},
                      files={"file": ("train_pic1.zip", f, "application/zip")},
                      timeout=120)
step("dataset upload", r.ok, r.json() if r.ok else r.text)

ds = requests.get(f"{API}/api/datasets").json()
step("dataset list", any(d["name"] == "train_pic1" and d["nc"] == 2 for d in ds))

# 4. 啟動訓練（3 epochs 快速驗證）
r = requests.post(f"{API}/api/train", json={
    "model": "yolo11n.pt", "dataset": "train_pic1",
    "epochs": 3, "imgsz": 640, "batch": 8, "device": "0",
})
step("train start", r.ok, r.text if not r.ok else "")
task_id = r.json()["task_id"]
print(f"     task_id = {task_id}")

# 5. 輪詢直到結束（首次會下載 yolo11n.pt，多留一點時間）
deadline = time.time() + 1200
while time.time() < deadline:
    t = requests.get(f"{API}/api/train/{task_id}").json()
    print(f"     [{t['status']}] epoch {t['epoch']}/{t['total_epochs']}")
    if t["status"] in ("completed", "failed"):
        break
    time.sleep(10)

if t["status"] == "failed":
    print(t["error"])
step("train completed", t["status"] == "completed",
     f"(最後 metrics: {t['metrics_history'][-1] if t['metrics_history'] else 'n/a'})")

# 6. 已訓練模型應出現在註冊中心
models = requests.get(f"{API}/api/models").json()
trained_id = f"runs/{task_id}"
step("trained model registered", trained_id in [m["id"] for m in models])

# 7. 用剛訓練的權重推論一張訓練圖
img = sorted(Path(r"D:\Project\a027\Train_pic1\images").glob("*.png"))[0]
with open(img, "rb") as f:
    r = requests.post(f"{API}/api/predict",
                      data={"model": trained_id, "conf": 0.1},
                      files={"file": (img.name, f, "image/png")}, timeout=300)
result = r.json() if r.ok else {}
step("predict (trained)", r.ok and "annotated_image_b64" in result,
     f"(偵測到 {result.get('count')} 個物件: {result.get('detections')})")

# 8. 用官方預訓練推論（驗證 pretrained 路徑與自動下載）
with open(img, "rb") as f:
    r = requests.post(f"{API}/api/predict",
                      data={"model": "yolo11n.pt", "conf": 0.25},
                      files={"file": (img.name, f, "image/png")}, timeout=300)
step("predict (pretrained)", r.ok, f"(偵測到 {r.json().get('count')} 個物件)" if r.ok else r.text)

print("\n全部通過")
