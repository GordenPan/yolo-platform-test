"""驗證 item 6 套餐切換：選「快速試跑」後 epochs/imgsz 應被填成 10/320。"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "frontend" / "app.py")
FAILS = []


def step(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name} {detail}")
    if not ok:
        FAILS.append(name)


at = AppTest.from_file(APP, default_timeout=30).run()
at.sidebar.radio[0].set_value("🏋️ 模型訓練").run()

# 預設應為「標準」：epochs 50 / imgsz 640
step("default preset = 標準 (epochs=50)", at.session_state["t_epochs"] == 50,
     f"(epochs={at.session_state['t_epochs']})")

# 切到「快速試跑」
at.radio(key="t_preset").set_value("快速試跑").run()
step("quick preset -> epochs 10", at.session_state["t_epochs"] == 10,
     f"(epochs={at.session_state['t_epochs']})")
step("quick preset -> imgsz 320", at.session_state["t_imgsz"] == 320,
     f"(imgsz={at.session_state['t_imgsz']})")

# 切到「精細」
at.radio(key="t_preset").set_value("精細").run()
step("fine preset -> epochs 300 / imgsz 960",
     at.session_state["t_epochs"] == 300 and at.session_state["t_imgsz"] == 960,
     f"(epochs={at.session_state['t_epochs']}, imgsz={at.session_state['t_imgsz']})")

step("no exception", not at.exception)

print("\n" + ("有測試失敗: " + ", ".join(FAILS) if FAILS else "套餐切換全部正常"))
sys.exit(1 if FAILS else 0)
