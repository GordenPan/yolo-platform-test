"""YOLO Platform 前端（Streamlit）。

僅透過 REST API 與後端溝通，不直接 import backend 程式碼，維持前後端分離。
啟動：streamlit run frontend/app.py
"""
import base64
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("YOLO_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="YOLO Platform", page_icon="🎯", layout="wide")


def api_get(path: str, params: dict | None = None):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    if r.status_code >= 400:
        try:
            raise RuntimeError(r.json().get("detail", r.text))
        except ValueError:
            raise RuntimeError(r.text)
    return r.json()


def api_post(path: str, **kwargs):
    r = requests.post(f"{API}{path}", timeout=300, **kwargs)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


def model_selector(label: str, types: tuple[str, ...]) -> str | None:
    """共用的模型下拉選單，依 type 過濾（pretrained / trained / yolov26）。"""
    models = [m for m in api_get("/api/models") if m["type"] in types]
    if not models:
        st.warning("目前沒有可用的模型")
        return None
    choice = st.selectbox(label, models, format_func=lambda m: m["label"])
    return choice["id"]


# ---------- 側邊欄 ----------

st.sidebar.title("🎯 YOLO Platform")
page = st.sidebar.radio("功能", ["📁 資料集管理", "🏋️ 模型訓練", "📈 訓練監控", "🔍 推論測試", "📦 模型庫"])

try:
    api_get("/api/health")
    st.sidebar.success(f"後端連線正常\n\n{API}")
except Exception:
    st.sidebar.error(f"無法連線後端 {API}\n\n請先啟動：\n`uvicorn backend.main:app --port 8000`")
    st.stop()

system = api_get("/api/system")
if system["cuda"]:
    st.sidebar.info(f"🖥️ GPU：{system['gpu_name']}\n\n訓練/推論自動使用 GPU")
else:
    st.sidebar.warning("🖥️ 未偵測到 GPU，自動使用 CPU")


# ---------- 資料集管理 ----------

if page == "📁 資料集管理":
    st.header("📁 資料集管理")

    with st.form("register_form"):
        folder = st.text_input(
            "資料集資料夾路徑",
            placeholder=r"例如 D:\Project\a027\Train_pic1（需含 images/ 與 labels/，或已含 data.yaml）",
            help="直接引用本機資料夾，不複製影像；資料夾更新後重新註冊即可",
        )
        name = st.text_input("資料集名稱（留空用資料夾名）", placeholder="例如 train_pic1")

        st.markdown("**train / val 切分方式**（僅在資料夾沒有 data.yaml 時生效）")
        split_label = st.radio(
            "切分方式", ["自動隨機切分 train/val", "train 與 val 使用同一資料夾"],
            label_visibility="collapsed",
            help="同一資料夾：驗證指標會偏樂觀（模型在看過的圖上驗證），但可用全部影像訓練",
        )
        val_ratio = st.slider("驗證集比例（自動切分時）", 0.05, 0.5, 0.2, 0.05)
        classes = st.text_input("類別名稱（選填，逗號分隔）", placeholder="例如 crack_label, label_2")

        if st.form_submit_button("註冊資料集", type="primary") and folder:
            with st.spinner("註冊中…"):
                try:
                    info = api_post(
                        "/api/datasets/register",
                        data={"folder": folder, "name": name,
                              "split": "same" if "同一" in split_label else "auto",
                              "val_ratio": val_ratio,
                              "classes": classes})
                    st.success(f"已註冊資料集 **{info['name']}**（{info['nc']} 類別，"
                               f"{'train=val 同資料夾' if info['split'] == 'same_folder' else '已切分 train/val'}）")
                except RuntimeError as e:
                    st.error(str(e))

    st.subheader("現有資料集")
    ds = api_get("/api/datasets")
    if ds:
        st.dataframe(pd.DataFrame(ds)[["name", "nc", "classes", "split"]], use_container_width=True)
    else:
        st.info("尚無資料集，請先上傳")


# ---------- 模型訓練 ----------

elif page == "🏋️ 模型訓練":
    st.header("🏋️ 模型訓練")

    ds = api_get("/api/datasets")
    if not ds:
        st.warning("請先到「資料集管理」上傳資料集")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        model_id = model_selector("基底模型 / 架構", ("pretrained", "yolov26", "trained"))
        dataset = st.selectbox("資料集", [d["name"] for d in ds])
        run_name = st.text_input("Run 名稱（留空用時間戳）", placeholder="例如 train_a027_0612")
    with col2:
        epochs = st.number_input("Epochs", 1, 10000, 50)
        imgsz = st.select_slider("影像尺寸", [320, 416, 512, 640, 768, 960, 1024], value=640)
        batch = st.number_input("Batch size", 1, 256, 16)

    st.markdown("##### 訓練策略")
    s1, s2, s3, s4 = st.columns(4)
    pretrained = s1.checkbox("使用預訓練權重", value=True)
    patience = s2.number_input("Patience（early stop）", 0, 100000, 100)
    save_period = s3.number_input("Save period（-1 停用）", -1, 10000, -1)
    close_mosaic = s4.number_input("Close mosaic（最後 N epochs）", 0, 10000, 10)

    st.markdown("##### 資料增強")
    a1, a2, a3 = st.columns(3)
    degrees = a1.number_input("旋轉角度 degrees", -180.0, 180.0, 0.0, 5.0)
    flipud = a2.slider("上下翻轉機率 flipud", 0.0, 1.0, 0.0, 0.05)
    fliplr = a3.slider("左右翻轉機率 fliplr", 0.0, 1.0, 0.5, 0.05)

    with st.expander("⚙️ 其他進階參數（JSON，透傳給 ultralytics train()）"):
        extra_text = st.text_area(
            "上方沒列出的參數寫在這裡：",
            value="{}",
            help='範例：{"mosaic": 0, "scale": 0.1, "erasing": 0, "cache": "disk"}',
        )

    if st.button("🚀 開始訓練", type="primary") and model_id:
        try:
            extra = json.loads(extra_text or "{}")
        except json.JSONDecodeError as e:
            st.error(f"進階參數不是合法 JSON：{e}")
            st.stop()
        try:
            resp = api_post("/api/train", json={
                "model": model_id, "dataset": dataset, "name": run_name or None,
                "epochs": int(epochs), "imgsz": int(imgsz), "batch": int(batch),
                "pretrained": pretrained, "patience": int(patience),
                "save_period": int(save_period), "close_mosaic": int(close_mosaic),
                "degrees": degrees, "flipud": flipud, "fliplr": fliplr,
                "extra": extra,
            })
            st.success(f"訓練任務已啟動：`{resp['task_id']}`，請到「訓練監控」查看進度")
        except RuntimeError as e:
            st.error(str(e))


# ---------- 訓練監控 ----------

elif page == "📈 訓練監控":
    st.header("📈 訓練監控")

    tasks = api_get("/api/train")
    if not tasks:
        st.info("尚無訓練任務")
        st.stop()

    status_icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
    task = st.selectbox(
        "選擇任務", tasks,
        format_func=lambda t: f"{status_icon.get(t['status'], '')} {t['id']}（{t['status']}）",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔃 重新整理"):
            st.rerun()
    with col2:
        auto = st.checkbox("自動更新（每 5 秒）", value=task["status"] == "running")

    task = api_get(f"/api/train/{task['id']}")  # 取最新狀態

    c1, c2, c3 = st.columns(3)
    c1.metric("狀態", task["status"])
    c2.metric("進度", f"{task['epoch']} / {task['total_epochs']} epochs")
    c3.metric("模型", task["params"]["model"])
    if task["total_epochs"]:
        st.progress(min(task["epoch"] / task["total_epochs"], 1.0))

    if task["error"]:
        st.error("訓練失敗")
        st.code(task["error"])

    if task["metrics_history"]:
        df = pd.DataFrame(task["metrics_history"]).set_index("epoch")
        map_cols = [c for c in df.columns if "mAP" in c]
        loss_cols = [c for c in df.columns if "loss" in c]
        g1, g2 = st.columns(2)
        if map_cols:
            g1.subheader("mAP")
            g1.line_chart(df[map_cols])
        if loss_cols:
            g2.subheader("Loss")
            g2.line_chart(df[loss_cols])
        with st.expander("完整指標表"):
            st.dataframe(df, use_container_width=True)

    if task["status"] == "completed" and task["run_dir"]:
        st.success(f"訓練完成！權重位置：`{task['run_dir']}\\weights\\best.pt`")

    if auto and task["status"] in ("pending", "running"):
        time.sleep(5)
        st.rerun()


# ---------- 推論測試 ----------

elif page == "🔍 推論測試":
    st.header("🔍 推論測試")

    folder = st.text_input("影像資料夾路徑",
                           value=st.session_state.get("inf_folder", ""),
                           placeholder=r"例如 D:\Project\a027\Train_pic1\images")
    model_id = model_selector("模型", ("pretrained", "trained", "yolov26"))
    conf = st.slider("信心閾值", 0.05, 0.95, 0.25, 0.05)

    if st.button("📂 載入資料夾") and folder:
        try:
            resp = api_get("/api/folder/images", params={"path": folder})
            if resp["count"] == 0:
                st.warning("此資料夾中沒有影像檔")
            else:
                st.session_state.inf_folder = folder
                st.session_state.inf_images = resp["images"]
                st.session_state.inf_idx = 0
                st.session_state.inf_results = {}
                st.rerun()
        except RuntimeError as e:
            st.error(str(e))

    images = st.session_state.get("inf_images", [])
    if not images:
        st.info("輸入資料夾路徑並按「📂 載入資料夾」開始")
    else:
        idx = st.session_state.get("inf_idx", 0)
        n = len(images)

        b1, b2, b3, b4, b5 = st.columns(5)
        if b1.button("⬅️ 上一張", use_container_width=True):
            st.session_state.inf_idx = (idx - 1) % n
            st.rerun()
        if b2.button("下一張 ➡️", use_container_width=True):
            st.session_state.inf_idx = (idx + 1) % n
            st.rerun()
        run_clicked = b3.button("🔍 推論", type="primary", use_container_width=True)
        run_all_clicked = b4.button("🔍🔍 推論全部", use_container_width=True)
        mark_clicked = b5.button("📋 複製到 need_to_train", use_container_width=True)

        idx = st.session_state.inf_idx
        current = images[idx]
        st.caption(f"第 {idx + 1} / {n} 張｜`{current}`")

        if run_clicked and model_id:
            with st.spinner("推論中…"):
                try:
                    st.session_state.inf_results[current] = api_post(
                        "/api/predict_path",
                        data={"model": model_id, "image_path": current, "conf": conf})
                except RuntimeError as e:
                    st.error(str(e))

        if run_all_clicked and model_id:
            prog = st.progress(0.0, text=f"批次推論中… 0 / {n}")
            failed = 0
            for i, img in enumerate(images):
                try:
                    st.session_state.inf_results[img] = api_post(
                        "/api/predict_path",
                        data={"model": model_id, "image_path": img, "conf": conf})
                except RuntimeError:
                    failed += 1
                prog.progress((i + 1) / n, text=f"批次推論中… {i + 1} / {n}")
            prog.empty()
            if failed:
                st.warning(f"批次推論完成，{n - failed} 張成功、{failed} 張失敗")
            else:
                st.success(f"批次推論完成，共 {n} 張")

        if mark_clicked:
            try:
                r = api_post("/api/mark_for_training", data={"image_path": current})
                msg = f"已複製到 `{r['copied_to']}`"
                if r["label_copied"]:
                    msg += "（含對應標註檔）"
                st.success(msg)
            except RuntimeError as e:
                st.error(str(e))

        left, right = st.columns(2)
        with left:
            st.image(current, caption="原始影像", use_container_width=True)
        with right:
            result = st.session_state.inf_results.get(current)
            if result:
                st.image(base64.b64decode(result["annotated_image_b64"]),
                         caption=f"偵測結果（{result['count']} 個物件）", use_container_width=True)
                if result["detections"]:
                    st.dataframe(pd.DataFrame(result["detections"]), use_container_width=True)
                else:
                    st.info("未偵測到任何物件，可嘗試降低信心閾值")
            else:
                st.info("尚未推論，按「🔍 推論」執行")

        done = {p: r for p, r in st.session_state.inf_results.items() if p in images}
        if len(done) > 1:
            with st.expander(f"📊 全部結果總覽（已推論 {len(done)} / {n} 張）", expanded=False):
                rows = [{"檔名": os.path.basename(p),
                         "物件數": r["count"],
                         "最高信心": max((d["confidence"] for d in r["detections"]), default=None)}
                        for p, r in done.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------- 模型庫 ----------

elif page == "📦 模型庫":
    st.header("📦 模型庫")
    models = api_get("/api/models")

    st.subheader("已訓練模型")
    trained = [m for m in models if m["type"] == "trained"]
    if trained:
        rows = [{"run": m["id"], **m.get("metrics", {})} for m in trained]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("尚無已訓練模型")

    st.subheader("YOLOv26 模型（擴充區）")
    yolov26 = [m for m in models if m["type"] == "yolov26"]
    if yolov26:
        st.dataframe(pd.DataFrame(yolov26)[["id", "label"]], use_container_width=True)
    else:
        st.caption("YOLOv26 為 Ultralytics 官方後續版本。發布後升級 ultralytics 套件，"
                   "將 yolo26*.pt 權重（或 .yaml 架構設定）放入 `models/YOLOV26/` 即會自動出現在此處與各模型選單")

    st.subheader("官方預訓練")
    st.dataframe(pd.DataFrame([m for m in models if m["type"] == "pretrained"])[["id", "label"]],
                 use_container_width=True)
