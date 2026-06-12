# YOLO Platform — 一站式 YOLO 訓練與測試平台

前後端分離架構：**FastAPI**（核心後端，調用 `ultralytics` 執行 YOLOv11）+ **Streamlit**（操作介面，透過 REST API 與後端溝通）。

> ⚠️ **安全警告：僅限本機單人使用**
>
> 本平台設計為個人工作站上的本機工具，**沒有任何身分驗證**，且 API 依設計可以
> 讀取伺服器上的任意檔案路徑（資料夾瀏覽推論、資料集註冊）、寫入檔案（need_to_train）
> 並執行訓練任務。請務必：
>
> - 只綁定 `127.0.0.1`（預設與 `start.bat` 皆是如此），**絕對不要**用
>   `--host 0.0.0.0` 或任何方式把 port 8000/8501 暴露到區域網路或網際網路
> - 不要在多人共用的主機上執行
> - 如需遠端使用，請自行加上反向代理 + 身分驗證，並改寫路徑相關 API 的存取範圍

## 系統架構

```
┌─────────────────────┐         REST API          ┌──────────────────────────────┐
│  Streamlit 前端      │  ◄──────────────────────► │  FastAPI 後端                 │
│  frontend/app.py    │   /api/models             │  backend/main.py             │
│                     │   /api/datasets           │                              │
│  - 資料集管理        │   /api/train              │  services/                   │
│  - 訓練啟動/監控     │   /api/predict            │   ├ registry.py  模型註冊中心 │
│  - 推論測試          │                           │   ├ datasets.py  資料集管理   │
│  - 模型庫            │                           │   ├ trainer.py   訓練任務管理 │
└─────────────────────┘                           │   └ predictor.py 推論服務     │
                                                  └──────────┬───────────────────┘
                                                             │ ultralytics YOLO
                                                  ┌──────────▼───────────────────┐
                                                  │  models/                     │
                                                  │   ├ pretrained/  YOLOv11預訓練│
                                                  │   └ YOLOV26/     ★YOLOv26預留 │
                                                  │  datasets/       上傳的資料集 │
                                                  │  runs/           訓練輸出     │
                                                  └──────────────────────────────┘
```

## 目錄結構

```
yolo-platform/
├── backend/
│   ├── main.py              # FastAPI 入口與 API 路由
│   ├── schemas.py           # Pydantic 請求/回應模型
│   ├── core/
│   │   └── config.py        # 路徑與全域設定
│   └── services/
│       ├── registry.py      # 模型註冊中心（YOLOv11 預訓練 / 已訓練 / YOLOv26）
│       ├── datasets.py      # 資料集上傳、驗證、列表
│       ├── trainer.py       # 背景訓練任務（callback 即時回報 metrics）
│       └── predictor.py     # 推論（回傳偵測結果 + 標註圖）
├── frontend/
│   └── app.py               # Streamlit 操作介面
├── models/
│   ├── pretrained/          # yolo11n/s/m/l/x.pt（首次使用自動下載）
│   └── YOLOV26/             # ★ 預留：Ultralytics YOLOv26 發布後放入 yolo26*.pt 即自動出現在介面
├── datasets/                # 資料集（上傳 zip 自動解壓，需含 data.yaml）
├── runs/                    # 訓練輸出（weights/best.pt、results.csv、曲線圖）
├── requirements.txt
└── README.md
```

## 安裝與啟動

```powershell
pip install -r requirements.txt

# 1. 啟動後端（視窗一）
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 2. 啟動前端（視窗二）
streamlit run frontend/app.py
```

前端預設連 `http://127.0.0.1:8000`，可用環境變數 `YOLO_API_URL` 覆寫。

## 資料集格式

上傳的 zip 需為標準 ultralytics 格式，內含 `data.yaml`：

```
my_dataset.zip
├── data.yaml        # 含 train/val 路徑、nc、names
├── images/train/  images/val/
└── labels/train/  labels/val/
```

後端解壓後會自動把 `data.yaml` 的 `path` 改寫為絕對路徑。

## 擴充 YOLOv26（Ultralytics 官方後續版本）

YOLOv26 同為 Ultralytics 官方模型，`YOLO()` 可原生載入，**接入不需改任何程式碼**：

1. 升級套件：`pip install -U ultralytics`
2. 將 `yolo26n.pt` 等權重（或 `.yaml` 架構設定）放入 `models/YOLOV26/`
3. 模型註冊中心會自動掃描，前端「訓練」與「推論」選單即出現 `YOLOV26/...` 選項

若要讓 YOLOv26 支援首次使用自動下載，把檔名加入
`backend/core/config.py` 的 `PRETRAINED_MODELS` 清單即可。

## 授權（License）

本專案採用 **AGPL-3.0** 授權（見 [LICENSE](LICENSE)）。

這是因為核心依賴 [`ultralytics`](https://github.com/ultralytics/ultralytics) 採用
AGPL-3.0：衍生與整合使用的軟體在散布或以網路服務形式提供時，必須以相同條款開源。
若需要閉源或商業部署，請向 [Ultralytics](https://www.ultralytics.com/license)
購買企業授權，或將推論/訓練核心替換為其他授權友善的實作。
