"""資料集管理：上傳 zip、解壓、驗證/產生 data.yaml、列表。

支援兩種 zip：
  1. 已含 data.yaml 的標準 ultralytics 格式 -> 直接使用（僅改寫 path 為絕對路徑）
  2. 平面結構（images/ + labels/，無 data.yaml）-> 依使用者選擇產生 data.yaml：
       split="auto" : 隨機切分 train/val（產生 train.txt / val.txt 檔案清單）
       split="same" : train 與 val 指向同一個 images 資料夾
"""
from __future__ import annotations

import io
import random
import re
import shutil
import zipfile
from pathlib import Path

import yaml

from ..core.config import DATASETS_DIR

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def save_dataset(
    name: str,
    zip_bytes: bytes,
    split: str = "auto",
    val_ratio: float = 0.2,
    class_names: list[str] | None = None,
) -> dict:
    safe = re.sub(r"[^\w\-]", "_", name).strip("_")
    if not safe:
        raise ValueError("資料集名稱無效")
    target = DATASETS_DIR / safe
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(target)

    yaml_path = _find_data_yaml(target)
    if yaml_path is not None:
        # 標準格式：沿用原本的 data.yaml，僅改寫 path
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["path"] = str(yaml_path.parent)
    else:
        # 平面結構：自動產生 data.yaml
        yaml_path, data = _generate_data_yaml(target, split, val_ratio, class_names)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)

    return _dataset_info(safe, yaml_path, data)


def register_folder(
    name: str | None,
    folder: str,
    split: str = "auto",
    val_ratio: float = 0.2,
    class_names: list[str] | None = None,
) -> dict:
    """註冊本機資料夾為資料集（不複製影像，data.yaml 直接指向原始資料夾）。

    支援三種輸入：
      - 含 data.yaml 的標準格式資料夾 -> 直接沿用其設定
      - 含 images/ + labels/ 的資料夾 -> 依切分選項產生 data.yaml
      - 直接指向 images 資料夾（旁邊有 labels/）-> 自動以上層為根目錄
    """
    src = Path(folder).resolve()
    if not src.is_dir():
        raise ValueError(f"找不到資料夾: {folder}")

    safe = re.sub(r"[^\w\-]", "_", name or src.name).strip("_")
    if not safe:
        raise ValueError("資料集名稱無效")
    target = DATASETS_DIR / safe
    # 來源就在 datasets/<name> 本身（使用者手動放入的資料夾）：就地註冊，絕不能刪除
    in_place = target.exists() and target.resolve() == src
    if target.exists() and not in_place:
        if src == target.resolve() or src in target.resolve().parents or target.resolve() in src.parents:
            raise ValueError(f"來源資料夾與資料集目錄重疊，無法註冊: {src}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    existing = _find_data_yaml(src)
    if existing is not None:
        # 已是標準格式：沿用原 data.yaml，僅確保 path 為絕對路徑
        with open(existing, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["path"] = str(existing.parent)
    else:
        root = _locate_root(src)
        img_dir = root / "images"
        images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        if not images:
            shutil.rmtree(target)
            raise ValueError(f"{img_dir} 中沒有影像檔")

        nc = _infer_num_classes(root / "labels")
        if class_names and len(class_names) >= nc:
            names = class_names
            nc = len(names)
        else:
            names = [f"class{i}" for i in range(nc)]

        if split == "same":
            train_field, val_field = "images", "images"
        else:
            # 切分清單放在平台的 datasets/<name>/ 下（用絕對路徑），不動使用者的資料夾
            random.seed(42)
            shuffled = images[:]
            random.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            val_imgs, train_imgs = shuffled[:n_val], shuffled[n_val:]
            train_txt, val_txt = target / "train.txt", target / "val.txt"
            train_txt.write_text("\n".join(str(f) for f in train_imgs), encoding="utf-8")
            val_txt.write_text("\n".join(str(f) for f in val_imgs), encoding="utf-8")
            train_field, val_field = str(train_txt), str(val_txt)

        data = {"path": str(root), "train": train_field, "val": val_field,
                "nc": nc, "names": names}

    yaml_path = target / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return _dataset_info(safe, yaml_path, data)


def _locate_root(src: Path) -> Path:
    """找出含 images/（與 labels/）的資料集根目錄。"""
    if (src / "images").is_dir():
        return src
    has_images = any(f.suffix.lower() in IMG_EXTS for f in src.iterdir() if f.is_file())
    if has_images and src.name.lower() == "images":
        return src.parent
    hits = [d for d in src.rglob("images") if d.is_dir()]
    if hits:
        return hits[0].parent
    raise ValueError(f"{src} 下找不到 images/ 資料夾（也沒有 data.yaml）")


def list_datasets() -> list[dict]:
    out = []
    for d in sorted(DATASETS_DIR.iterdir()) if DATASETS_DIR.exists() else []:
        if not d.is_dir():
            continue
        yaml_path = _find_data_yaml(d)
        if yaml_path is None:
            continue
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            out.append(_dataset_info(d.name, yaml_path, data))
        except Exception:
            continue
    return out


def resolve_dataset(name: str) -> str:
    """回傳資料集 data.yaml 的絕對路徑（傳給 model.train(data=...)）。"""
    target = DATASETS_DIR / name
    yaml_path = _find_data_yaml(target) if target.exists() else None
    if yaml_path is None:
        raise FileNotFoundError(f"找不到資料集: {name}")
    return str(yaml_path)


def _generate_data_yaml(
    target: Path, split: str, val_ratio: float, class_names: list[str] | None
) -> tuple[Path, dict]:
    img_dirs = [d for d in target.rglob("images") if d.is_dir()]
    if not img_dirs:
        shutil.rmtree(target)
        raise ValueError("zip 中找不到 data.yaml，也找不到 images/ 資料夾，無法辨識格式")
    img_dir = img_dirs[0]
    root = img_dir.parent

    images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not images:
        shutil.rmtree(target)
        raise ValueError("images/ 資料夾中沒有影像檔")

    nc = _infer_num_classes(root / "labels")
    if class_names and len(class_names) >= nc:
        names = class_names[:max(nc, len(class_names))]
        nc = len(names)
    else:
        names = [f"class{i}" for i in range(nc)]

    if split == "same":
        train_field, val_field = "images", "images"
    else:
        random.seed(42)
        shuffled = images[:]
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_imgs, train_imgs = shuffled[:n_val], shuffled[n_val:]
        (root / "train.txt").write_text(
            "\n".join(f"./images/{f.name}" for f in train_imgs), encoding="utf-8")
        (root / "val.txt").write_text(
            "\n".join(f"./images/{f.name}" for f in val_imgs), encoding="utf-8")
        train_field, val_field = "train.txt", "val.txt"

    data = {"path": str(root), "train": train_field, "val": val_field, "nc": nc, "names": names}
    return root / "data.yaml", data


def _infer_num_classes(labels_dir: Path) -> int:
    """掃描 YOLO 標註檔的 class id，推斷類別數。"""
    max_id = -1
    if labels_dir.is_dir():
        for txt in labels_dir.glob("*.txt"):
            try:
                for line in txt.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts:
                        max_id = max(max_id, int(float(parts[0])))
            except (ValueError, UnicodeDecodeError):
                continue
    if max_id < 0:
        raise ValueError("labels/ 中沒有可解析的 YOLO 標註，無法推斷類別數")
    return max_id + 1


def _find_data_yaml(root: Path) -> Path | None:
    direct = root / "data.yaml"
    if direct.exists():
        return direct
    hits = sorted(root.rglob("data.yaml"))
    return hits[0] if hits else None


def _dataset_info(name: str, yaml_path: Path, data: dict) -> dict:
    names = data.get("names")
    if isinstance(names, dict):
        names = list(names.values())
    return {
        "name": name,
        "yaml": str(yaml_path),
        "nc": data.get("nc", len(names) if names else None),
        "classes": names or [],
        "split": "same_folder" if data.get("train") == data.get("val") else "separated",
    }
