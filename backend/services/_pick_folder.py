"""獨立腳本：開啟原生資料夾選擇視窗，把選到的路徑印到 stdout。

由後端以子程序呼叫（subprocess），避免 tkinter 在 uvicorn 工作執行緒中
無法執行主迴圈的問題。僅適用於後端與使用者同一台、有桌面環境的本機情境。
"""
import sys


def main() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="請選擇資料集／影像資料夾")
        root.destroy()
        sys.stdout.write(path or "")
    except Exception:
        sys.stdout.write("")


if __name__ == "__main__":
    main()
