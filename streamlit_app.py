"""
Qwen3 ASR 字幕生成器 - Streamlit Web 前端（PyTorch GPU 版）
Glass Morphism Dark UI | PyTorch CUDA 推理後端

啟動：python -m streamlit run streamlit_app.py
（由 start-gpu.bat 選擇 Streamlit 時自動啟動）
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import tempfile
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

# ── 確保同目錄 Python 模組可被 import ─────────────────────────────────
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── 路徑設定 ─────────────────────────────────────────────────────────
GPU_MODEL_DIR = BASE_DIR / "GPUModel"
OV_MODEL_DIR  = BASE_DIR / "ov_models"
SRT_DIR       = BASE_DIR / "subtitles"
SRT_DIR.mkdir(exist_ok=True)

ASR_MODEL_NAME = "Qwen3-ASR-1.7B"

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]

SAMPLE_RATE = 16000


# ══════════════════════════════════════════════════════════════════════
# CSS：Glass Morphism Dark Theme（與 Vulkan 版保持相同美術風格）
# ══════════════════════════════════════════════════════════════════════

_CSS = """
<style>
/* ---------- 全域背景 ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d0d20 50%, #12102a 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: rgba(15, 15, 35, 0.85);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(100, 100, 200, 0.15);
}

/* ---------- Glass Morphism 卡片 ---------- */
.glass-card {
    background: rgba(20, 20, 45, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(100, 100, 220, 0.2);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 30, 0.4);
}

/* ---------- 標題 ---------- */
.main-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #8888ff 0%, #aa88ff 50%, #88ccff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle { color: #8888bb; font-size: 0.92rem; margin-bottom: 1rem; }

/* ---------- 狀態指示器 ---------- */
.status-ready   { color: #58D68D; font-weight: 600; }
.status-loading { color: #F0B27A; font-weight: 600; }
.status-error   { color: #F1948A; font-weight: 600; }

/* ---------- 結果區塊 ---------- */
.result-block {
    background: rgba(10, 10, 30, 0.6);
    border: 1px solid rgba(80, 80, 180, 0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'Consolas', monospace;
    font-size: 0.88rem;
    color: #AAAACC;
    white-space: pre-wrap;
    max-height: 380px;
    overflow-y: auto;
}

/* ---------- 進度文字 ---------- */
.prog-text { color: #7799BB; font-size: 0.85rem; font-family: monospace; }

/* ---------- 按鈕覆寫 ---------- */
div.stButton > button {
    background: rgba(60, 60, 160, 0.35);
    border: 1px solid rgba(100, 100, 220, 0.4);
    color: #CCCCEE;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: rgba(80, 80, 200, 0.5);
    border-color: rgba(140, 140, 255, 0.6);
    color: #EEEEFF;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════
# 引擎快取（跨 rerun 保留）
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_engine(device: str):
    """載入 GPUASREngine（快取，reload 後複用）。"""
    try:
        from app_gpu import GPUASREngine, GPU_MODEL_DIR as _GPU_DIR
    except ImportError:
        # 如果無法 import app_gpu，嘗試直接 import
        from app_gpu import GPUASREngine
        _GPU_DIR = GPU_MODEL_DIR

    engine = GPUASREngine()
    msgs: list[str] = []
    engine.load(device=device, model_dir=_GPU_DIR, cb=lambda m: msgs.append(m))
    return engine, msgs


def _get_engine():
    """取得或載入 GPUASREngine，使用 session_state 記錄狀態。"""
    ss = st.session_state
    if "engine_loaded" not in ss:
        ss.engine_loaded = False
        ss.engine_error  = None

    if ss.engine_loaded:
        return st.session_state.get("engine_obj")

    device = ss.get("device_choice", "cuda")
    try:
        engine, load_msgs = _load_engine(device)
        ss.engine_loaded = True
        ss.engine_obj    = engine
        ss.engine_msgs   = load_msgs
        return engine
    except Exception as e:
        ss.engine_error = str(e)
        return None


# ══════════════════════════════════════════════════════════════════════
# Streamlit 頁面
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Qwen3 ASR GPU",
        page_icon="🎙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── 側欄：設定 ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="main-title">🎙 Qwen3 ASR</div>'
            '<div class="subtitle">PyTorch GPU 版 · 字幕生成器</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # 裝置選擇
        ss = st.session_state
        _cuda_available = False
        try:
            import torch
            _cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        device_options = (["cuda", "cpu"] if _cuda_available else ["cpu"])
        device_labels  = []
        for d in device_options:
            if d == "cuda":
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    device_labels.append(f"CUDA ({gpu_name[:20]}, {vram:.0f}GB)")
                except Exception:
                    device_labels.append("CUDA")
            else:
                device_labels.append("CPU")

        prev_device = ss.get("device_choice", device_options[0])
        sel_idx = device_options.index(prev_device) if prev_device in device_options else 0
        sel_label = st.selectbox("推理裝置", device_labels, index=sel_idx)
        ss.device_choice = device_options[device_labels.index(sel_label)]

        # 語系
        lang_options = ["自動偵測"] + SUPPORTED_LANGUAGES
        sel_lang = st.selectbox("語系", lang_options, index=0,
                                key="lang_sel")
        language = sel_lang if sel_lang != "自動偵測" else None

        # 說話者分離
        diarize = st.checkbox("說話者分離", value=False, key="diarize_chk")
        n_speakers = None
        if diarize:
            n_raw = st.selectbox("說話者人數", ["自動", "2", "3", "4", "5", "6"],
                                 key="n_spk_sel")
            n_speakers = int(n_raw) if n_raw.isdigit() else None

        st.divider()

        # 模型狀態
        st.markdown("**模型狀態**")
        engine = _get_engine()
        if engine and engine.ready:
            st.markdown(
                f'<span class="status-ready">✅ 就緒（{ss.device_choice.upper()}）</span>',
                unsafe_allow_html=True,
            )
        elif ss.get("engine_error"):
            st.markdown(
                f'<span class="status-error">❌ 載入失敗：{ss.engine_error[:60]}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-loading">⏳ 載入中…</span>',
                unsafe_allow_html=True,
            )

        # 重新載入按鈕
        if st.button("🔄 重新載入模型"):
            _load_engine.clear()
            ss.engine_loaded = False
            ss.engine_error  = None
            st.rerun()

        st.divider()
        st.markdown(
            f"<small style='color:#445566;'>模型目錄：<br>{GPU_MODEL_DIR}</small>",
            unsafe_allow_html=True,
        )

    # ── 主內容：Tab ─────────────────────────────────────────────────
    tab_file, tab_settings = st.tabs(["📁  音檔轉字幕", "⚙️  設定"])

    # ─────── 音檔轉字幕 tab ─────────────────────────────────────────
    with tab_file:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        # 辨識提示
        hint_text = st.text_area(
            "辨識提示（可選）",
            placeholder="貼入歌詞、關鍵字或背景說明，可提升辨識準確度…",
            height=80,
            key="hint_area",
        )
        context = hint_text.strip() or None

        # 檔案上傳
        uploaded = st.file_uploader(
            "上傳音訊 / 影片檔案",
            type=["mp3", "wav", "flac", "m4a", "ogg", "aac",
                  "mp4", "mkv", "avi", "mov", "wmv", "webm"],
            key="audio_uploader",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            col1, col2 = st.columns([3, 1])
            with col2:
                start_btn = st.button("▶  開始轉換", use_container_width=True,
                                      type="primary", key="start_btn")

            if start_btn:
                engine = _get_engine()
                if not engine or not engine.ready:
                    st.error("⚠️ 模型尚未載入，請等待或重新載入。")
                else:
                    # 儲存上傳檔案到臨時路徑
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = Path(tmp.name)

                    prog_placeholder = st.empty()
                    result_placeholder = st.empty()
                    log_lines: list[str] = []

                    def _prog_cb(done, total, msg):
                        pct = done / total if total > 0 else 0
                        log_lines.append(msg)
                        prog_placeholder.progress(pct, text=msg)

                    try:
                        # 影片檔案需先提取音軌
                        from ffmpeg_utils import VIDEO_EXTS, find_ffmpeg, extract_audio_to_wav
                        proc_path = tmp_path
                        tmp_wav   = None
                        if tmp_path.suffix.lower() in VIDEO_EXTS:
                            ffmpeg = find_ffmpeg()
                            if not ffmpeg:
                                st.error("❌ 需要 ffmpeg 才能處理影片。請先安裝 ffmpeg 並加入 PATH。")
                                tmp_path.unlink(missing_ok=True)
                                st.stop()
                            fd, wav_path = tempfile.mkstemp(suffix=".wav")
                            os.close(fd)
                            tmp_wav = Path(wav_path)
                            with st.spinner("🎬 提取音軌中…"):
                                extract_audio_to_wav(tmp_path, tmp_wav, ffmpeg)
                            proc_path = tmp_wav

                        with st.spinner("🔄 轉換中，請稍候…"):
                            t0  = time.perf_counter()
                            srt = engine.process_file(
                                proc_path,
                                progress_cb=_prog_cb,
                                language=language,
                                context=context,
                                diarize=diarize,
                                n_speakers=n_speakers,
                            )
                            elapsed = time.perf_counter() - t0

                        # 清理臨時檔
                        tmp_path.unlink(missing_ok=True)
                        if tmp_wav:
                            tmp_wav.unlink(missing_ok=True)

                        prog_placeholder.empty()

                        if srt:
                            st.success(f"✅ 完成！耗時 {elapsed:.1f}s")
                            srt_text = srt.read_text(encoding="utf-8")
                            result_placeholder.markdown(
                                f'<div class="result-block">{srt_text}</div>',
                                unsafe_allow_html=True,
                            )
                            st.download_button(
                                "💾 下載 SRT",
                                data=srt_text.encode("utf-8"),
                                file_name=srt.name,
                                mime="text/plain",
                                key="dl_btn",
                            )
                        else:
                            st.warning("⚠️ 未偵測到人聲，未產生字幕。")

                    except Exception as e:
                        tmp_path.unlink(missing_ok=True)
                        if "tmp_wav" in dir() and tmp_wav:
                            tmp_wav.unlink(missing_ok=True)
                        st.error(f"❌ 轉換失敗：{e}")

    # ─────── 設定 tab ────────────────────────────────────────────────
    with tab_settings:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 模型路徑")
        st.code(str(GPU_MODEL_DIR / ASR_MODEL_NAME))

        asr_ok = (GPU_MODEL_DIR / ASR_MODEL_NAME / "config.json").exists()
        if asr_ok:
            st.success(f"✅ 找到模型：{ASR_MODEL_NAME}")
        else:
            st.error(
                f"❌ 找不到模型：{GPU_MODEL_DIR / ASR_MODEL_NAME}\n\n"
                "請執行 `start-gpu.bat` 並選擇下載模型。"
            )

        st.markdown("### 說明")
        st.markdown(
            "- 此 Streamlit 前端使用 **PyTorch CUDA** 後端（`app-gpu.py`）\n"
            "- 支援音訊：mp3 / wav / flac / m4a / ogg / aac\n"
            "- 支援影片（需要 ffmpeg）：mp4 / mkv / avi / mov / wmv / webm\n"
            "- 字幕輸出至：`subtitles/` 目錄\n"
            "- 若需 Vulkan GPU 支援，請改用 `QwenASR.exe`"
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
