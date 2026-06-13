# YOLO Platform — All-in-One YOLO Training & Testing Platform

*[繁體中文](README.md) | English*

A decoupled architecture: **FastAPI** (core backend, calls `ultralytics` to run YOLOv11/YOLOv26) + **Streamlit** (UI, talks to the backend purely over REST).

> ⚠️ **Security warning: local single-user use only**
>
> This platform is designed as a local tool on a personal workstation. It has **no authentication**,
> and by design the API can read arbitrary file paths on the server (folder-browsing inference,
> dataset registration), write files (`need_to_train`), and launch training jobs. Therefore:
>
> - Bind to `127.0.0.1` only (the default, and what `start.bat` does). **Never** use
>   `--host 0.0.0.0` or otherwise expose ports 8000/8501 to a LAN or the internet.
> - Do not run it on a shared host.
> - For remote use, put it behind a reverse proxy with authentication and restrict the
>   path-related APIs yourself.

## Architecture

```
┌─────────────────────┐         REST API          ┌──────────────────────────────┐
│  Streamlit frontend │  ◄──────────────────────► │  FastAPI backend             │
│  frontend/app.py    │   /api/models             │  backend/main.py             │
│                     │   /api/datasets           │                              │
│  - Dataset mgmt     │   /api/train              │  services/                   │
│  - Train & monitor  │   /api/predict            │   ├ registry.py   model registry │
│  - Inference test   │                           │   ├ datasets.py   dataset mgmt   │
│  - Model library    │                           │   ├ trainer.py    training jobs  │
└─────────────────────┘                           │   └ predictor.py  inference      │
                                                  └──────────┬───────────────────┘
                                                             │ ultralytics YOLO
                                                  ┌──────────▼───────────────────┐
                                                  │  models/                     │
                                                  │   ├ pretrained/  YOLOv11/v26  │
                                                  │   └ YOLOV26/     custom slot  │
                                                  │  datasets/       registered   │
                                                  │  runs/           train output │
                                                  └──────────────────────────────┘
```

## Project layout

```
yolo-platform/
├── backend/
│   ├── main.py              # FastAPI entry point & API routes
│   ├── schemas.py           # Pydantic request/response models
│   ├── core/
│   │   └── config.py        # paths & global settings
│   └── services/
│       ├── registry.py      # model registry (pretrained / trained / YOLOv26)
│       ├── datasets.py      # dataset registration, validation, listing
│       ├── trainer.py       # background training jobs (live metrics via callback)
│       ├── predictor.py     # inference (detections + annotated image)
│       └── fsutil.py        # folder size & path-traversal-safe deletion
├── frontend/
│   └── app.py               # Streamlit UI
├── models/
│   ├── pretrained/          # yolo11/yolo26 n/s/m/l/x.pt (auto-downloaded on first use)
│   └── YOLOV26/             # drop-in slot for custom yolo26*.pt / .yaml
├── datasets/                # registered datasets (data.yaml only; images stay in place)
├── runs/                    # training output (weights/best.pt, results.csv, curves)
├── docs/manual.md           # step-by-step user manual
├── start.bat / install.bat  # one-click launch / install (Windows)
├── requirements.txt
└── README.md / README.en.md
```

## Install & run

```powershell
# One-click on Windows: double-click install.bat, then start.bat
# Or manually:
pip install -r requirements.txt

# 1. Start the backend (window 1)
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 2. Start the frontend (window 2)
streamlit run frontend/app.py --server.address 127.0.0.1
```

The frontend connects to `http://127.0.0.1:8000` by default; override with the `YOLO_API_URL` env var.

> GPU note: `requirements.txt` does not pin PyTorch, so a plain `pip` install pulls the CPU build.
> For GPU acceleration, install a CUDA build, e.g. `pip install --index-url
> https://download.pytorch.org/whl/cu118 torch torchvision`. `install.bat` detects this and
> prints the right command.

## Dataset format

Point the platform at a **local folder** (no upload/copy — `data.yaml` just references it).
A standard ultralytics layout works directly:

```
my_dataset/
├── data.yaml        # train/val paths, nc, names  (optional — can be auto-generated)
├── images/          # or images/train + images/val
└── labels/          # YOLO .txt labels, same stem as each image
```

If there is no `data.yaml`, the platform generates one and lets you choose the train/val split
(random auto-split, or use the same folder for both). Class names are inferred from the labels
and can be overridden.

## Key features

- **Dataset management**: pick a local folder with the built-in folder browser (or paste a path);
  images are referenced, not copied. Choose auto train/val split or same folder.
- **Training**: preset modes (Quick / Standard / Fine / Custom) fill the form in one click;
  common hyperparameters are first-class, with advanced JSON pass-through. GPU is auto-detected.
  AutoBatch (`batch=-1`) avoids out-of-memory crashes.
- **Monitoring**: live mAP/loss curves and progress bar with a red/yellow/green verdict; shows the
  parameters used; **cancel anytime** (stops at the next epoch boundary, weights preserved);
  **resume** cancelled/interrupted/failed runs; records are persisted, so they **survive a backend
  restart**, and runs copied into `runs/` manually are listed automatically.
- **Inference**: folder-browsing mode, single or batch inference; copy hard cases to
  `need_to_train/` with one click.
- **Model library / management**: lists trained models with a mAP verdict; delete datasets
  (without touching your source images), delete runs, clean `need_to_train`; disk usage in the sidebar.

## YOLOv26 support

YOLOv26 is an official Ultralytics model loaded natively by `YOLO()`, so **no code changes are needed**:

- `yolo26n/s/m/l/x.pt` are in the auto-download pretrained list and appear under
  "official pretrained" in the model selector (requires `ultralytics >= 8.4`).
- You can also drop a custom `yolo26*.pt` (or a `.yaml` architecture) into `models/YOLOV26/`;
  the registry scans it and it shows up in the Training and Inference selectors.

If your installed `ultralytics` is too old to parse a YOLOv26 architecture, training fails with a
friendly hint telling you to run `pip install -U ultralytics`.

## License

This project is licensed under **AGPL-3.0** (see [LICENSE](LICENSE)).

This follows from its core dependency [`ultralytics`](https://github.com/ultralytics/ultralytics),
which is AGPL-3.0: derivative and integrated software must be released under the same terms when
distributed or offered as a network service. For closed-source or commercial deployment, obtain an
enterprise license from [Ultralytics](https://www.ultralytics.com/license), or replace the
inference/training core with a permissively licensed implementation.
