"""用 Streamlit AppTest 在程序內無頭執行前端，確認五頁都不拋例外（需後端在跑）。"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "frontend" / "app.py")
PAGES = ["📁 資料集管理", "🏋️ 模型訓練", "📈 訓練監控", "🔍 推論測試", "📦 模型庫"]
FAILS = []


def render(page=None):
    at = AppTest.from_file(APP, default_timeout=30).run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


ASCII = ["default", "datasets", "train", "monitor", "inference", "models"]
for i, page in enumerate([None] + PAGES):
    label = ASCII[i]
    try:
        at = render(page)
        if at.exception:
            FAILS.append(label)
            print(f"FAIL | {label}: {[str(e) for e in at.exception]}".encode('ascii', 'replace').decode())
        else:
            print(f"PASS | {label} (exceptions=0, error_blocks={len(at.error)})")
    except Exception as e:
        FAILS.append(label)
        print(f"FAIL | {label}: {type(e).__name__}: {e}".encode('ascii', 'replace').decode())

print("\n" + ("有頁面渲染失敗: " + ", ".join(FAILS) if FAILS else "五頁全部正常渲染"))
sys.exit(1 if FAILS else 0)
