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


class ApiError(RuntimeError):
    """帶 HTTP 狀態碼的 API 錯誤，讓呼叫端可區分 409 等特定情況。"""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


def _raise_for_status(r):
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ApiError(r.status_code, detail)


def api_get(path: str, params: dict | None = None):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    _raise_for_status(r)
    return r.json()


def api_post(path: str, **kwargs):
    r = requests.post(f"{API}{path}", timeout=300, **kwargs)
    _raise_for_status(r)
    return r.json()


def api_delete(path: str, params: dict | None = None):
    r = requests.delete(f"{API}{path}", params=params, timeout=60)
    _raise_for_status(r)
    return r.json()


def map_verdict(m: float | None) -> str:
    """把 mAP50 翻成白話紅綠燈結論。"""
    if m is None:
        return ""
    if m < 0.3:
        return f"🔴 mAP50={m:.3f}｜模型還抓不太到目標，建議增加標註圖片或提高 epochs"
    if m < 0.7:
        return f"🟡 mAP50={m:.3f}｜尚可，仍有進步空間（可再加資料或延長訓練）"
    return f"🟢 mAP50={m:.3f}｜表現良好"


def model_selector(label: str, types: tuple[str, ...]) -> str | None:
    """共用的模型下拉選單，依 type 過濾（pretrained / trained / yolov26）。"""
    models = [m for m in api_get("/api/models") if m["type"] in types]
    if not models:
        st.warning("目前沒有可用的模型")
        return None
    choice = st.selectbox(label, models, format_func=lambda m: m["label"])
    return choice["id"]


def folder_picker(key: str) -> str:
    """用作業系統原生視窗選擇資料夾（僅本機）。回傳已選路徑；若選到沒有影像的
    資料夾會提醒使用者。另保留手動貼路徑作為後備（由呼叫頁的 expander 提供）。"""
    picked_key = f"{key}_picked"
    c1, c2 = st.columns([1, 4])
    if c1.button("📂 選擇資料夾…", key=f"{key}_browse", use_container_width=True):
        with st.spinner("已開啟選擇視窗，請在彈出的視窗中選擇資料夾…"):
            try:
                res = api_post("/api/fs/pick_folder")
            except RuntimeError as e:
                st.error(str(e))
                res = None
        if res and not res.get("cancelled"):
            st.session_state[picked_key] = res["path"]
            st.session_state[f"{key}_usable"] = res
            st.rerun()

    picked = st.session_state.get(picked_key, "")
    if picked:
        c2.text_input("已選擇的資料夾", value=picked, disabled=True)
        info = st.session_state.get(f"{key}_usable", {})
        if info.get("usable"):
            cnt = info.get("image_count", 0)
            extra = "（含 data.yaml）" if cnt == -1 else f"（約 {cnt} 張影像）"
            st.success(f"✅ {info.get('reason', '')} {extra}")
        elif info:
            st.warning(f"⚠️ {info.get('reason', '此資料夾可能無法使用')}——請改選含影像的資料夾。")
    else:
        c2.caption("尚未選擇；點左側按鈕開啟選擇視窗，或於下方進階手動輸入路徑。")
    return picked


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

try:
    su = api_get("/api/storage")
    st.sidebar.caption(
        f"💾 磁碟用量\n\n"
        f"- 訓練成果 runs：{su['runs_mb']:.0f} MB\n"
        f"- 資料集 datasets：{su['datasets_mb']:.0f} MB\n"
        f"- need_to_train：{su['need_to_train_mb']:.0f} MB")
except Exception:
    pass


# ---------- 資料集管理 ----------

if page == "📁 資料集管理":
    st.header("📁 資料集管理")

    st.markdown("##### 1. 選擇資料集資料夾")
    picked = folder_picker("ds")
    with st.expander("或直接輸入／貼上路徑（進階）"):
        manual = st.text_input(
            "資料夾路徑", value="", key="ds_manual",
            placeholder=r"例如 D:\Project\a027\Train_pic1（需含 images/ 與 labels/，或已含 data.yaml）")
    folder = manual.strip() or picked

    st.markdown("##### 2. 設定")
    name = st.text_input("資料集名稱（留空用資料夾名）", placeholder="例如 train_pic1")
    st.markdown("**train / val 切分方式**（僅在資料夾沒有 data.yaml 時生效）")
    split_label = st.radio(
        "切分方式", ["自動隨機切分 train/val", "train 與 val 使用同一資料夾"],
        label_visibility="collapsed",
        help="同一資料夾：驗證指標會偏樂觀（模型在看過的圖上驗證），但可用全部影像訓練",
    )
    val_ratio = st.slider("驗證集比例（自動切分時）", 0.05, 0.5, 0.2, 0.05)
    classes = st.text_input("類別名稱（選填，逗號分隔）", placeholder="例如 crack_label, label_2")

    def _do_register(overwrite: bool):
        data = {"folder": folder, "name": name,
                "split": "same" if "同一" in split_label else "auto",
                "val_ratio": val_ratio, "classes": classes, "overwrite": str(overwrite).lower()}
        try:
            info = api_post("/api/datasets/register", data=data)
            st.session_state.pop("ds_pending", None)
            st.success(f"已註冊資料集 **{info['name']}**（{info['nc']} 類別，"
                       f"{'train=val 同資料夾' if info['split'] == 'same_folder' else '已切分 train/val'}）")
        except ApiError as e:
            if e.status == 409:
                st.session_state["ds_pending"] = name or os.path.basename(folder.rstrip("\\/"))
                st.rerun()
            else:
                st.error(str(e))

    if st.button("註冊資料集", type="primary"):
        if not folder:
            st.warning("請先選擇或輸入資料夾路徑")
        else:
            with st.spinner("註冊中…"):
                _do_register(overwrite=False)

    if st.session_state.get("ds_pending"):
        st.warning(f"⚠️ 資料集「{st.session_state['ds_pending']}」已存在，覆蓋會刪除舊的註冊設定。")
        cc1, cc2, _ = st.columns([1, 1, 3])
        if cc1.button("✅ 確認覆蓋", type="primary"):
            with st.spinner("覆蓋中…"):
                _do_register(overwrite=True)
        if cc2.button("取消"):
            st.session_state.pop("ds_pending", None)
            st.rerun()

    st.subheader("現有資料集")
    ds = api_get("/api/datasets")
    if ds:
        st.dataframe(pd.DataFrame(ds)[["name", "nc", "classes", "split"]], use_container_width=True)
        with st.expander("🗑️ 刪除資料集"):
            st.caption("只會刪除平台的資料集註冊，不會動到你的來源影像資料夾。")
            target = st.selectbox("選擇要刪除的資料集", [d["name"] for d in ds], key="ds_del_sel")
            confirm = st.checkbox(f"我確定要刪除「{target}」", key="ds_del_confirm")
            if st.button("刪除", disabled=not confirm):
                try:
                    api_delete(f"/api/datasets/{target}")
                    st.success(f"已刪除資料集「{target}」")
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
    else:
        st.info("尚無資料集，請先註冊")


# ---------- 模型訓練 ----------

elif page == "🏋️ 模型訓練":
    st.header("🏋️ 模型訓練")

    ds = api_get("/api/datasets")
    if not ds:
        st.warning("請先到「資料集管理」上傳資料集")
        st.stop()

    # 預設套餐：選了非「自訂」就把對應欄位填好（仍可手動微調）
    PRESETS = {
        "快速試跑": {"t_epochs": 10, "t_imgsz": 320, "t_batch": 16, "t_patience": 50, "t_close_mosaic": 5},
        "標準":   {"t_epochs": 50, "t_imgsz": 640, "t_batch": 8, "t_patience": 100, "t_close_mosaic": 10},
        "精細":   {"t_epochs": 300, "t_imgsz": 960, "t_batch": 4, "t_patience": 200, "t_close_mosaic": 20},
    }
    for _k, _v in PRESETS["標準"].items():
        st.session_state.setdefault(_k, _v)

    preset = st.radio("訓練模式", ["快速試跑", "標準", "精細", "自訂"], index=1, horizontal=True,
                      key="t_preset",
                      help="快速試跑≈10分鐘看效果；標準=一般用途；精細=高解析長訓練；自訂=完全手動")
    if preset != "自訂" and st.session_state.get("_t_last_preset") != preset:
        for _k, _v in PRESETS[preset].items():
            st.session_state[_k] = _v
        st.session_state["_t_last_preset"] = preset
        st.rerun()
    st.session_state["_t_last_preset"] = preset

    col1, col2 = st.columns(2)
    with col1:
        model_id = model_selector("基底模型 / 架構", ("pretrained", "yolov26", "trained"))
        dataset = st.selectbox("資料集", [d["name"] for d in ds])
        run_name = st.text_input("Run 名稱（留空用時間戳）", placeholder="例如 train_a027_0612")
    with col2:
        epochs = st.number_input("Epochs", 1, 10000, key="t_epochs")
        imgsz = st.number_input("影像尺寸 imgsz", 100, 10000, step=32, key="t_imgsz",
                                help="ultralytics 會自動對齊到 32 的倍數；越大越吃 GPU 記憶體")
        auto_batch = st.checkbox("🛡️ 自動 batch（AutoBatch，避免 GPU 記憶體不足）", value=False,
                                 help="勾選後由 ultralytics 依 GPU 可用記憶體自動決定 batch size")
        batch = st.number_input("Batch size", 1, 256, key="t_batch", disabled=auto_batch,
                                help="過大會導致該訓練因 GPU 記憶體不足而失敗（不會影響系統），"
                                     "錯誤會顯示在訓練監控；記憶體有限時請調小或改用 AutoBatch")

    st.markdown("##### 訓練策略")
    s1, s2, s3, s4 = st.columns(4)
    pretrained = s1.checkbox("使用預訓練權重", value=True)
    patience = s2.number_input("Patience（early stop）", 0, 100000, key="t_patience")
    save_period = s3.number_input("Save period（-1 停用）", -1, 10000, -1)
    close_mosaic = s4.number_input("Close mosaic（最後 N epochs）", 0, 10000, key="t_close_mosaic")

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
                "epochs": int(epochs), "imgsz": int(imgsz),
                "batch": -1 if auto_batch else int(batch),
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

    status_icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌",
                   "cancelled": "⏹️", "interrupted": "⚠️"}
    task = st.selectbox(
        "選擇任務", tasks,
        format_func=lambda t: f"{status_icon.get(t['status'], '')} {t['id']}（{t['status']}）",
    )

    task = api_get(f"/api/train/{task['id']}")  # 取最新狀態
    is_active = task["status"] in ("pending", "running")

    col1, col2, col3 = st.columns([1, 2, 3])
    with col1:
        if st.button("🔃 重新整理"):
            st.rerun()
    can_resume = task["status"] in ("cancelled", "interrupted", "failed") and task.get("run_dir")
    with col2:
        if is_active:
            if task.get("cancel_requested"):
                st.info("⏳ 取消中…將在本 epoch 結束後停止")
            elif st.button("⏹ 取消訓練", type="secondary"):
                try:
                    api_post(f"/api/train/{task['id']}/cancel")
                    st.warning("已送出取消，將在本 epoch 結束後停止（權重會保留）")
                except RuntimeError as e:
                    st.error(str(e))
        elif can_resume:
            if st.button("▶️ 繼續訓練", type="primary"):
                try:
                    resp = api_post(f"/api/train/{task['id']}/resume")
                    st.success(f"已從上次的權重續訓，新任務：`{resp['task_id']}`")
                except RuntimeError as e:
                    st.error(str(e))
    with col3:
        auto = st.checkbox("自動更新（每 5 秒）", value=is_active)

    c1, c2, c3 = st.columns(3)
    c1.metric("狀態", task["status"])
    c2.metric("進度", f"{task['epoch']} / {task['total_epochs']} epochs")
    c3.metric("模型", task["params"].get("model", "?"))
    if task["total_epochs"]:
        st.progress(min(task["epoch"] / task["total_epochs"], 1.0))

    # 本次訓練使用的參數
    p = task["params"]
    data_path = p.get("data", "")
    ds_name = os.path.basename(os.path.dirname(data_path)) if data_path else "?"
    info_rows = {
        "資料集": ds_name,
        "基底模型": p.get("model", "?"),
        "Run 名稱": p.get("name") or task["id"],
        "Epochs": p.get("epochs"),
        "影像尺寸 imgsz": p.get("imgsz"),
        "Batch": "AutoBatch（自動）" if p.get("batch") == -1 else p.get("batch"),
        "使用預訓練": p.get("pretrained"),
        "Patience": p.get("patience"),
        "Save period": p.get("save_period"),
        "Close mosaic": p.get("close_mosaic"),
        "旋轉 degrees": p.get("degrees"),
        "上下翻轉 flipud": p.get("flipud"),
        "左右翻轉 fliplr": p.get("fliplr"),
    }
    if p.get("resumed_from"):
        info_rows["續訓自"] = p["resumed_from"]
    if p.get("extra"):
        info_rows["進階參數"] = json.dumps(p["extra"], ensure_ascii=False)
    with st.expander("🧾 本次訓練參數", expanded=task["status"] != "completed"):
        if data_path:
            st.caption(f"資料設定檔：`{data_path}`")
        st.table(pd.DataFrame([(k, str(v)) for k, v in info_rows.items() if v is not None],
                              columns=["參數", "值"]))

    if task["error"]:
        st.error(f"訓練失敗：{task['hint']}" if task.get("hint") else "訓練失敗")
        with st.expander("錯誤詳情"):
            st.code(task["error"])
        if can_resume:
            st.caption("修正設定後，也可按上方「▶️ 繼續訓練」從最後的權重接續。")

    if task["metrics_history"]:
        df = pd.DataFrame(task["metrics_history"]).set_index("epoch")
        map_cols = [c for c in df.columns if "mAP" in c]
        loss_cols = [c for c in df.columns if "loss" in c]

        # 紅綠燈判讀（取最後一個 epoch 的 mAP50）
        map50_col = next((c for c in df.columns if "mAP50" in c and "50-95" not in c), None)
        if map50_col:
            verdict = map_verdict(float(df[map50_col].iloc[-1]))
            (st.success if "🟢" in verdict else st.warning if "🟡" in verdict else st.error)(verdict)

        g1, g2 = st.columns(2)
        if map_cols:
            g1.subheader("mAP")
            g1.line_chart(df[map_cols])
        if loss_cols:
            g2.subheader("Loss")
            g2.line_chart(df[loss_cols])
        with st.expander("完整指標表"):
            st.dataframe(df, use_container_width=True)

    if task["status"] in ("completed", "cancelled") and task["run_dir"]:
        prefix = "訓練完成！" if task["status"] == "completed" else "已取消（保留已訓練的權重）"
        st.success(f"{prefix}權重位置：`{task['run_dir']}\\weights\\best.pt`")
    elif task["status"] == "interrupted":
        st.warning("此任務在後端重啟前還在執行，已標記為中斷。若有產生權重仍可在 runs/ 找到。")

    if auto and is_active:
        time.sleep(5)
        st.rerun()


# ---------- 推論測試 ----------

elif page == "🔍 推論測試":
    st.header("🔍 推論測試")

    picked = folder_picker("inf")
    with st.expander("或直接輸入／貼上路徑（進階）"):
        manual = st.text_input("影像資料夾路徑", value="", key="inf_manual",
                               placeholder=r"例如 D:\Project\a027\Train_pic1\images")
    folder = manual.strip() or picked
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
        rows = []
        for m in trained:
            metrics = m.get("metrics", {})
            map50 = metrics.get("metrics/mAP50(B)")
            rows.append({
                "run": m["id"],
                "判讀": map_verdict(map50).split("｜")[0] if map50 is not None else "—",
                **metrics,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("判讀燈號：🟢 表現良好（mAP50>0.7）／🟡 尚可（0.3–0.7）／🔴 偏低（<0.3）")
    else:
        st.info("尚無已訓練模型")

    with st.expander("🗂️ 訓練成果管理（檢視大小／刪除）"):
        runs = api_get("/api/runs")
        if runs:
            st.dataframe(pd.DataFrame([{"名稱": r["name"], "大小(MB)": r["size_mb"],
                                        "有權重": "✓" if r["has_weights"] else ""} for r in runs]),
                         use_container_width=True)
            target = st.selectbox("選擇要刪除的訓練成果", [r["name"] for r in runs], key="run_del_sel")
            confirm = st.checkbox(f"我確定要刪除「{target}」（含權重與圖表，無法復原）", key="run_del_confirm")
            if st.button("刪除訓練成果", disabled=not confirm):
                try:
                    api_delete(f"/api/runs/{target}")
                    st.success(f"已刪除「{target}」")
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
        else:
            st.caption("runs/ 內尚無訓練成果")

    with st.expander("📋 need_to_train 管理"):
        ntt = api_get("/api/need_to_train")
        st.caption(f"目前 {ntt['count']} 張影像，共 {ntt['total_mb']:.1f} MB")
        if ntt["files"]:
            st.dataframe(pd.DataFrame(ntt["files"]).rename(
                columns={"name": "檔名", "size_mb": "大小(MB)"}), use_container_width=True)

            st.markdown("**自選刪除**")
            chosen = st.multiselect("選擇要刪除的影像（可多選）",
                                    [f["name"] for f in ntt["files"]], key="ntt_pick")
            if st.button("刪除選取", disabled=not chosen):
                ok = 0
                for fn in chosen:
                    try:
                        api_delete("/api/need_to_train", params={"name": fn})
                        ok += 1
                    except RuntimeError as e:
                        st.error(f"{fn}: {e}")
                st.success(f"已刪除 {ok} 個檔案")
                st.rerun()

            st.markdown("**清空全部**")
            if st.checkbox("我確定要清空 need_to_train 的所有影像", key="ntt_clear_confirm"):
                if st.button("清空 need_to_train"):
                    try:
                        r = api_delete("/api/need_to_train")
                        st.success(f"已清空（刪除 {r['removed_files']} 個檔案）")
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

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
