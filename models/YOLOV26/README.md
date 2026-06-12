# models/YOLOV26 — YOLOv26 擴充區（預留）

YOLOv26 與 YOLOv11 一樣是 **Ultralytics 官方**發布的模型，
同樣由 `ultralytics.YOLO()` 原生載入，因此接入不需要修改任何程式碼。

後端模型註冊中心（`backend/services/registry.py`）會自動掃描此資料夾：

| 放入的檔案 | 意義 | 出現位置 |
|---|---|---|
| `yolo26*.pt` | YOLOv26 官方預訓練 / 已訓練權重 | 訓練與推論頁的模型選單 |
| `*.yaml`     | YOLOv26 架構設定（從頭訓練）     | 訓練頁「基底模型」選單 |

模型 id 格式為 `YOLOV26/<檔名>`。

## 啟用步驟

1. 升級套件至支援 YOLOv26 的版本：`pip install -U ultralytics`
2. 將 `yolo26n.pt`（或 s/m/l/x 等規模）放入本資料夾
3. 重新整理前端頁面，模型選單即出現 `YOLOv26 預訓練｜yolo26n.pt`

若希望 YOLOv26 像 YOLOv11 一樣支援「首次使用自動下載」，
只需把對應檔名加入 `backend/core/config.py` 的 `PRETRAINED_MODELS` 清單。
