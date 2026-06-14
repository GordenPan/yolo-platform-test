"""FastAPI 入口：聚合模型、資料集、訓練、推論四組 API。

啟動：uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
import shutil
import string
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .core.config import DATASETS_DIR, NEED_TO_TRAIN_DIR, RUNS_DIR
from .schemas import TrainRequest, TrainResponse
from .services import datasets, predictor, registry, trainer
from .services.datasets import IMG_EXTS
from .services.fsutil import dir_size, mb, safe_child

app = FastAPI(title="YOLO Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/system")
def system_info():
    """回報硬體環境：訓練/推論會自動使用偵測到的裝置。"""
    info = {"cuda": False, "device": "cpu", "gpu_name": None}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info.update(cuda=True, device="0", gpu_name=torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return info


# ---------- 模型 ----------

@app.get("/api/models")
def get_models():
    return registry.list_models()


# ---------- 資料集 ----------

@app.get("/api/datasets")
def get_datasets():
    return datasets.list_datasets()


@app.post("/api/datasets/register")
def register_dataset(
    folder: str = Form(..., description="本機資料集資料夾路徑"),
    name: str = Form("", description="資料集名稱；留空用資料夾名"),
    split: str = Form("auto", description="auto=隨機切分 train/val；same=train 與 val 同一資料夾"),
    val_ratio: float = Form(0.2, ge=0.05, le=0.5),
    classes: str = Form("", description="類別名稱（逗號分隔，選填）"),
    overwrite: bool = Form(False, description="同名資料集已存在時是否覆蓋"),
):
    class_names = [s.strip() for s in classes.split(",") if s.strip()] or None
    try:
        return datasets.register_folder(name or None, folder, split=split, val_ratio=val_ratio,
                                        class_names=class_names, overwrite=overwrite)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=f"資料集「{e}」已存在，確認要覆蓋嗎？")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    """刪除資料集註冊（只刪平台的 datasets/<name>，不碰使用者來源影像）。"""
    try:
        datasets.delete_dataset(name)
        return {"deleted": name}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    split: str = Form("auto", description="auto=隨機切分 train/val；same=train 與 val 同一資料夾"),
    val_ratio: float = Form(0.2, ge=0.05, le=0.5),
    classes: str = Form("", description="類別名稱（逗號分隔，選填；僅用於 zip 無 data.yaml 時）"),
):
    class_names = [s.strip() for s in classes.split(",") if s.strip()] or None
    try:
        return datasets.save_dataset(name, await file.read(),
                                     split=split, val_ratio=val_ratio, class_names=class_names)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- 訓練 ----------

@app.post("/api/train", response_model=TrainResponse)
def start_train(req: TrainRequest):
    try:
        data_yaml = datasets.resolve_dataset(req.dataset)
        registry.resolve_model(req.model)  # 先驗證 model_id 可解析
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        task_id = trainer.start_training({
            "model": req.model,
            "data": data_yaml,
            "name": req.name,
            "epochs": req.epochs,
            "imgsz": req.imgsz,
            "batch": req.batch,
            "device": req.device,
            "pretrained": req.pretrained,
            "patience": req.patience,
            "save_period": req.save_period,
            "close_mosaic": req.close_mosaic,
            "degrees": req.degrees,
            "flipud": req.flipud,
            "fliplr": req.fliplr,
            "extra": req.extra,
        })
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return TrainResponse(task_id=task_id)


@app.get("/api/train")
def get_tasks():
    return trainer.list_tasks()


@app.get("/api/train/{task_id}")
def get_task(task_id: str):
    task = trainer.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="找不到任務")
    return task


@app.post("/api/train/{task_id}/cancel")
def cancel_train(task_id: str):
    """要求取消訓練（在當前 epoch 結束後生效，已訓練的權重會保留）。"""
    if trainer.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="找不到任務")
    if not trainer.cancel_task(task_id):
        raise HTTPException(status_code=400, detail="此任務非進行中，無法取消")
    return {"status": "cancelling"}


@app.delete("/api/train/{task_id}")
def delete_train_task(task_id: str):
    """刪除一筆訓練任務紀錄（含其 run 資料夾與權重）。進行中的需先取消。"""
    if trainer.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="找不到任務")
    try:
        trainer.delete_task(task_id)
        return {"deleted": task_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train/{task_id}/resume", response_model=TrainResponse)
def resume_train(task_id: str):
    """從中斷／取消／失敗的任務的 last.pt 續訓，繼續到原定 epochs。"""
    try:
        return TrainResponse(task_id=trainer.resume_training(task_id))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- 推論 ----------

@app.post("/api/predict")
async def predict(
    model: str = Form(...),
    conf: float = Form(0.25),
    file: UploadFile = File(...),
):
    try:
        return predictor.predict_image(model, await file.read(), conf=conf)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/folder/images")
def list_folder_images(path: str):
    """列出本機資料夾中的影像檔（資料夾瀏覽推論模式）。"""
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"找不到資料夾: {path}")
    images = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
    return {"images": images, "count": len(images)}


@app.get("/api/fs/browse")
def browse_filesystem(path: str = ""):
    """瀏覽本機檔案系統：path 空回磁碟機清單，否則回子資料夾與該層影像數。

    供前端資料夾選擇器使用，取代手打路徑。
    """
    if not path:
        drives = [f"{c}:\\" for c in string.ascii_uppercase if Path(f"{c}:\\").exists()]
        return {"path": "", "parent": None, "dirs": drives, "image_count": 0}

    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"找不到資料夾: {path}")

    subdirs, image_count = [], 0
    try:
        for entry in p.iterdir():
            try:
                if entry.is_dir():
                    subdirs.append(str(entry))
                elif entry.suffix.lower() in IMG_EXTS:
                    image_count += 1
            except OSError:
                continue  # 略過無權限/損壞的項目
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"無權限存取此資料夾: {path}")

    parent = str(p.parent) if str(p.parent) != str(p) else ""
    return {"path": str(p), "parent": parent, "dirs": sorted(subdirs), "image_count": image_count}


def _folder_usability(p: Path) -> dict:
    """判斷資料夾是否可當資料集／推論來源，並回報影像數（-1=未知，例如已有 data.yaml）。"""
    if not p.is_dir():
        return {"usable": False, "image_count": 0, "reason": "資料夾不存在"}
    if (p / "data.yaml").exists():
        return {"usable": True, "image_count": -1, "reason": "含 data.yaml"}
    direct = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if direct:
        return {"usable": True, "image_count": direct, "reason": "資料夾內含影像"}
    img_dir = p / "images"
    if img_dir.is_dir():
        c = sum(1 for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
        if c:
            return {"usable": True, "image_count": c, "reason": "含 images/ 子資料夾"}
    return {"usable": False, "image_count": 0,
            "reason": "此資料夾沒有影像（也沒有 images/ 子資料夾或 data.yaml）"}


@app.post("/api/fs/pick_folder")
def pick_folder():
    """開啟作業系統原生資料夾選擇視窗（僅本機可用），回傳選到的路徑與可用性。"""
    script = Path(__file__).parent / "services" / "_pick_folder.py"
    try:
        out = subprocess.run([sys.executable, str(script)],
                             capture_output=True, text=True, timeout=180)
        path = (out.stdout or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"無法開啟資料夾選擇視窗: {e}")

    if not path:
        return {"path": "", "cancelled": True, "usable": False, "image_count": 0, "reason": "已取消"}
    return {"path": path, "cancelled": False, **_folder_usability(Path(path))}


@app.post("/api/predict_path")
def predict_by_path(
    model: str = Form(...),
    image_path: str = Form(...),
    conf: float = Form(0.25),
):
    try:
        return predictor.predict_path(model, image_path, conf=conf)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mark_for_training")
def mark_for_training(image_path: str = Form(...)):
    """把影像複製到 need_to_train/，作為之後補標註/再訓練的集中區。

    若來源旁有對應的 YOLO 標註（../labels/<同名>.txt），一併複製到
    need_to_train/labels/，方便直接拿去組新資料集。
    """
    src = Path(image_path)
    if not src.is_file():
        raise HTTPException(status_code=400, detail=f"找不到影像: {image_path}")

    dst = NEED_TO_TRAIN_DIR / src.name
    shutil.copy2(src, dst)

    label_copied = False
    label = src.parent.parent / "labels" / f"{src.stem}.txt"
    if label.is_file():
        label_dst_dir = NEED_TO_TRAIN_DIR / "labels"
        label_dst_dir.mkdir(exist_ok=True)
        shutil.copy2(label, label_dst_dir / label.name)
        label_copied = True

    return {"copied_to": str(dst), "label_copied": label_copied}


# ---------- 管理：訓練成果、need_to_train、磁碟用量 ----------

@app.get("/api/runs")
def list_runs():
    """列出 runs/ 下的訓練成果（名稱、大小、是否有權重、修改時間）。"""
    out = []
    for run in sorted(RUNS_DIR.iterdir(), reverse=True) if RUNS_DIR.exists() else []:
        if not run.is_dir() or run.name == ".tasks":
            continue
        out.append({
            "name": run.name,
            "size_mb": mb(dir_size(run)),
            "has_weights": (run / "weights" / "best.pt").exists(),
            "mtime": run.stat().st_mtime,
        })
    return out


@app.get("/api/runs/{name}/plots")
def list_run_plots(name: str):
    """列出某個 run 資料夾內 ultralytics 產生的圖表/樣本影像檔名。"""
    try:
        run = safe_child(RUNS_DIR, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not run.is_dir():
        raise HTTPException(status_code=404, detail="找不到訓練成果")
    plots = [f.name for f in sorted(run.iterdir())
             if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return {"plots": plots}


@app.get("/api/runs/{name}/file")
def get_run_file(name: str, file: str):
    """提供 run 資料夾內的單一圖檔（給前端顯示訓練圖表）。"""
    try:
        run = safe_child(RUNS_DIR, name)
        target = safe_child(run, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not target.is_file():
        raise HTTPException(status_code=404, detail="找不到檔案")
    return FileResponse(str(target))


@app.delete("/api/runs/{name}")
def delete_run(name: str):
    """刪除一個訓練成果資料夾，並清掉對應的任務紀錄。"""
    try:
        target = safe_child(RUNS_DIR, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"找不到訓練成果: {name}")
    shutil.rmtree(target)
    trainer.purge_run(name)
    return {"deleted": name}


@app.get("/api/need_to_train")
def list_need_to_train():
    """列出 need_to_train/ 的影像（不含 labels 子資料夾）。"""
    files = []
    for f in sorted(NEED_TO_TRAIN_DIR.iterdir()) if NEED_TO_TRAIN_DIR.exists() else []:
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            files.append({"name": f.name, "size_mb": mb(f.stat().st_size)})
    return {"files": files, "count": len(files), "total_mb": mb(dir_size(NEED_TO_TRAIN_DIR))}


@app.delete("/api/need_to_train")
def clear_need_to_train(name: str = ""):
    """刪除 need_to_train 內的影像：指定 name 刪單張（含同名標註），否則清空全部。"""
    if name:
        try:
            img = safe_child(NEED_TO_TRAIN_DIR, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not img.is_file():
            raise HTTPException(status_code=404, detail=f"找不到檔案: {name}")
        img.unlink()
        label = NEED_TO_TRAIN_DIR / "labels" / f"{Path(name).stem}.txt"
        if label.is_file():
            label.unlink()
        return {"deleted": name}

    removed = 0
    for f in list(NEED_TO_TRAIN_DIR.iterdir()):
        try:
            if f.is_file():
                if f.name == ".gitkeep":   # 保留版控佔位檔
                    continue
                f.unlink(); removed += 1
            elif f.is_dir() and f.name == "labels":
                shutil.rmtree(f)
        except OSError:
            continue
    return {"cleared": True, "removed_files": removed}


@app.get("/api/storage")
def storage_usage():
    """各資料區的磁碟用量（MB）+ 所在磁碟的總量/剩餘/使用率，供側邊欄顯示。"""
    usage = shutil.disk_usage(str(RUNS_DIR))
    return {
        "datasets_mb": mb(dir_size(DATASETS_DIR)),
        "runs_mb": mb(dir_size(RUNS_DIR)),
        "need_to_train_mb": mb(dir_size(NEED_TO_TRAIN_DIR)),
        "disk_total_gb": round(usage.total / 1024**3, 1),
        "disk_free_gb": round(usage.free / 1024**3, 1),
        "disk_used_gb": round(usage.used / 1024**3, 1),
        "disk_used_pct": round(usage.used / usage.total * 100, 1),
        "disk_free_pct": round(usage.free / usage.total * 100, 1),
    }
