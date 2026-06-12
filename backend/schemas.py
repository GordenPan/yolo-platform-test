"""API 請求/回應的 Pydantic 模型。"""
from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    model: str = Field(..., description="model_id，例如 yolo11n.pt / YOLOV26/xxx.yaml / runs/<run>")
    dataset: str = Field(..., description="資料集名稱（datasets/ 下的資料夾名）")
    name: str | None = Field(None, description="run 名稱；留空用時間戳")
    epochs: int = Field(50, ge=1, le=10000)
    imgsz: int = Field(640, ge=32)
    batch: int = Field(16, ge=1)
    device: str | None = Field(None, description="留空自動偵測（有 GPU 用 GPU，否則 CPU）")
    # 常用訓練策略（預設值同 ultralytics）
    pretrained: bool = Field(True, description="是否使用預訓練權重")
    patience: int = Field(100, ge=0, description="early stopping 容忍 epochs")
    save_period: int = Field(-1, description="每 N epochs 存一次權重；-1 停用")
    close_mosaic: int = Field(10, ge=0, description="最後 N epochs 關閉 mosaic")
    # 常用資料增強
    degrees: float = Field(0.0, ge=-180, le=180, description="隨機旋轉角度範圍")
    flipud: float = Field(0.0, ge=0, le=1, description="上下翻轉機率")
    fliplr: float = Field(0.5, ge=0, le=1, description="左右翻轉機率")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="其餘 ultralytics train() 超參數透傳，例如 {'mosaic': 0, 'cache': 'disk'}",
    )


class TrainResponse(BaseModel):
    task_id: str
