"""把 D:/Project/a027/Train_pic1（平面 images/labels）打包成平台需要的資料集 zip。

做 80/20 train/val 切分並產生 data.yaml，輸出到 scripts/train_pic1.zip。
"""
import random
import zipfile
from pathlib import Path

SRC = Path(r"D:\Project\a027\Train_pic1")
OUT = Path(__file__).parent / "train_pic1.zip"

images = sorted((SRC / "images").glob("*.png"))
random.seed(42)
random.shuffle(images)
n_val = max(1, int(len(images) * 0.2))
val_set = set(img.stem for img in images[:n_val])

data_yaml = """train: images/train
val: images/val
nc: 2
names: [crack_label, label_2]
"""

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("data.yaml", data_yaml)
    for img in images:
        split = "val" if img.stem in val_set else "train"
        zf.write(img, f"images/{split}/{img.name}")
        label = SRC / "labels" / f"{img.stem}.txt"
        if label.exists():
            zf.write(label, f"labels/{split}/{label.name}")

print(f"OK: {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB, train={len(images)-n_val}, val={n_val})")
