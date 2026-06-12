"""FastAPI 入口：聚合模型、資料集、訓練、推論四組 API。

啟動：uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .core.config import NEED_TO_TRAIN_DIR
from .schemas import TrainRequest, TrainResponse
from .services import datasets, predictor, registry, trainer
from .services.datasets import IMG_EXTS

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
):
    class_names = [s.strip() for s in classes.split(",") if s.strip()] or None
    try:
        return datasets.register_folder(name or None, folder,
                                        split=split, val_ratio=val_ratio, class_names=class_names)
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
