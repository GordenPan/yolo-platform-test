"""檔案系統工具：資料夾大小計算、防穿越的安全子路徑解析。

刪除類 API 一律透過 safe_child() 取得目標路徑，確保不會被 ../ 或絕對路徑
穿越到平台目錄之外。
"""
from __future__ import annotations

from pathlib import Path


def dir_size(path: Path) -> int:
    """遞迴計算資料夾位元組數（容錯：略過無法存取的項目）。"""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def mb(num_bytes: int) -> float:
    return round(num_bytes / 1024 / 1024, 2)


def safe_child(base: Path, name: str) -> Path:
    """回傳 base 底下名為 name 的子路徑，並確保解析後仍在 base 內。

    name 必須是單一路徑元件——含路徑分隔或 .. 一律拒絕，避免穿越攻擊。
    """
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"非法名稱: {name}")
    base_r = base.resolve()
    target = (base_r / name).resolve()
    if base_r not in target.parents:
        raise ValueError(f"路徑超出允許範圍: {name}")
    return target
