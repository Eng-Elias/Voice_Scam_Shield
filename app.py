# This script is the main entry point for the Voice Scam Shield application, a Streamlit web app.
# It handles the user interface, real-time microphone analysis, and file-based audio analysis.
# The app uses Whisper for transcription and can leverage Gemini for semantic scam detection.

import json
import time
from datetime import datetime
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io

from config.settings import get_settings, Settings
from src.audio_processor import AudioProcessor as WhisperEngine
from src.scam_detector import ScamDetector
from src.alert_system import render_alert
from src import file_handler


st.set_page_config(page_title="Voice Scam Shield", page_icon="🛡️", layout="wide")

# ---------- Styles ----------
CUSTOM_CSS = """
<style>
.small { font-size: 0.9rem; color: #666; }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9rem; }
.card { padding: 0.9rem; border-radius: 8px; border: 1px solid #eee; background: #fff; }
.badge { padding: 0.1rem 0.5rem; border-radius: 6px; background: #eef2ff; color: #334155; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "stream_scores" not in st.session_state:
    st.session_state.stream_scores: List[Dict[str, Any]] = []
if "stream_transcript" not in st.session_state:
    st.session_state.stream_transcript = ""
if "stream_wave" not in st.session_state:
    st.session_state.stream_wave: List[Dict[str, Any]] = []


# ---------- Settings & Engines ----------
settings = get_settings()

st.sidebar.header("Settings")
st.sidebar.caption("Adjust analysis sensitivity and runtime options.")
ui_sensitivity = st.sidebar.slider(
    "Scam sensitivity (higher = more sensitive)", 0.1, 0.95, float(settings.scam_sensitivity), 0.05
)
settings.scam_sensitivity = float(ui_sensitivity)

# Optional audio beeps on high-risk alerts
enable_beep = st.sidebar.checkbox("Audio beep on HIGH risk", value=True)

st.sidebar.write("\n")
st.sidebar.caption("Whisper model for transcription (set via .env)")
st.sidebar.code(f"WHISPER_MODEL={settings.whisper_model}", language="bash")
st.sidebar.caption("Gemini model (set via .env)")
st.sidebar.code(f"GEMINI_MODEL={settings.gemini_model}", language="bash")

whisper_engine = WhisperEngine(settings)
scam_detector = ScamDetector(settings)


# ---------- Header ----------
st.title("🛡️ Voice Scam Shield")
st.caption(
    "Detect potential scams real time from microphone streams or uploaded audio files. "
    "Transcribe with Whisper. Optionally analyze with Gemini for semantic cues."
)


# ---------- Mic Recorder Helpers ----------
def _decode_wav_bytes_to_float32(audio_bytes: bytes) -> (np.ndarray, int):
    """Decodes WAV audio bytes into a mono float32 numpy array and its sample rate."""
    """Decode WAV bytes to mono float32 numpy array in [-1, 1] and return (samples, sample_rate)."""
    bio = io.BytesIO(audio_bytes)
    seg = AudioSegment.from_file(bio, format="wav")
    # Ensure 16kHz mono 16-bit PCM for Whisper
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    sr = seg.frame_rate
    arr = np.array(seg.get_array_of_samples())
    x = (arr.astype(np.float32) / 32767.0).astype(np.float32)
    return x, sr


def _process_new_mic_audio(audio: Dict[str, Any], engine: WhisperEngine, detector: ScamDetector, settings: Settings) -> None:
    """Processes a new audio chunk from the microphone recorder."""
    """On receiving a new audio dict from mic_recorder, split into chunks and update transcript/analysis."""
    if not audio:
        return
    last_id = st.session_state.get("_mic_last_id", 0)
    if audio.get("id", 0) <= last_id:
        return
    st.session_state["_mic_last_id"] = audio.get("id", 0)

    if not engine.is_ready():
        return

    with st.spinner("Processing recorded audio..."):
        x, sr = _decode_wav_bytes_to_float32(audio["bytes"])  # sr will be 16000
        n = len(x)
        if n == 0:
            return

        chunk_samples = int(max(1, settings.chunk_seconds) * sr)

        # Amplitude envelope for visualization (0.1s steps)
        step = max(1, int(0.1 * sr))
        base_t = time.time()
        for i in range(0, n, step):
            seg = x[i : i + step]
            if len(seg) == 0:
                continue
            amp = float(np.clip(np.abs(seg).mean() * 5.0, 0.0, 1.0))
            st.session_state.stream_wave.append({"t": base_t + (i / sr), "amp": amp})
        st.session_state.stream_wave = st.session_state.stream_wave[-300:]

        # Transcribe each chunk and update analysis incrementally
        for s in range(0, n, chunk_samples):
            e = min(n, s + chunk_samples)
            chunk = x[s:e]
            if len(chunk) < int(0.5 * sr):
                # Skip too-short chunk to reduce overhead
                continue
            text = engine.transcribe_stream_chunk(chunk, sr)
            if text:
                st.session_state.stream_transcript = (st.session_state.stream_transcript + " " + text).strip()
                analysis = detector.analyze_text(st.session_state.stream_transcript[-2000:])
                st.session_state["_last_analysis"] = analysis
                st.session_state.stream_scores.append({
                    "t": time.time(),
                    "score": float(analysis.risk_score),
                    "level": analysis.risk_level,
                })


# ---------- Tabs ----------
stream_tab, file_tab, history_tab = st.tabs(["Stream Analysis", "File Analysis", "History"])


with stream_tab:
    st.subheader("🎙️ Live Microphone Analysis (Recorder)")
    if not whisper_engine.is_ready():
        st.warning(
            "Whisper is not available. Install dependencies from requirements.txt to enable transcription."
        )

    # Record audio using mic_recorder. We request WAV for simpler decoding.
    audio = mic_recorder(
        start_prompt="⏺️ Start recording",
        stop_prompt="⏹️ Stop",
        format="wav",
        key="vss_mic",
    )

    # When a new clip is received (after you press Stop), process it in CHUNK_SECONDS windows
    if audio:
        _process_new_mic_audio(audio, whisper_engine, scam_detector, settings)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Transcript (cumulative)")
        st.text_area(
            "Transcript",
            value=st.session_state.stream_transcript,
            height=200,
            label_visibility="collapsed",
        )
    with col2:
        st.caption("Audio level")
        lvl = 0.0
        if st.session_state.stream_wave:
            lvl = float(st.session_state.stream_wave[-1]["amp"])
        st.progress(lvl)

    # Mini waveform / level trace
    if st.session_state.stream_wave:
        st.caption("Waveform (relative level)")
        dfw = pd.DataFrame([
            {"Time": pd.to_datetime(x["t"], unit="s"), "Level": x["amp"]}
            for x in st.session_state.stream_wave
        ])
        wchart = (
            alt.Chart(dfw)
            .mark_line()
            .encode(x="Time:T", y=alt.Y("Level:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=120)
        )
        st.altair_chart(wchart, use_container_width=True)

    if st.session_state.get("_last_analysis"):
        st.caption("Live alert")
        render_alert(st, st.session_state["_last_analysis"], enable_audio=enable_beep)

    # Confidence over time chart
    if st.session_state.stream_scores:
        st.caption("Confidence over time")
        df = pd.DataFrame([
            {"Time": pd.to_datetime(x["t"], unit="s"), "Score": x["score"]}
            for x in st.session_state.stream_scores
        ])
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x="Time:T", y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)


with file_tab:
    st.subheader("📁 Audio File Analysis")
    uploaded = st.file_uploader(
        "Upload audio files (MP3, WAV, M4A, FLAC)",
        type=["mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True,
    )

    if uploaded:
        results: List[Dict[str, Any]] = []
        cleanup_dirs: List[str] = []
        prog = st.progress(0.0)
        for i, f in enumerate(uploaded, start=1):
            try:
                wav_path, meta = file_handler.process_uploaded_file(f)
                cleanup_dirs.append(wav_path.rsplit("/", 1)[0] if "/" in wav_path else wav_path.rsplit("\\", 1)[0])
                tr = whisper_engine.transcribe_file(wav_path) if whisper_engine.is_ready() else None
                text = tr.text if tr else ""
                analysis = scam_detector.analyze_text(text)

                with st.expander(f"{f.name} — Risk: {analysis.risk_level} ({analysis.risk_score:.2f})", expanded=True):
                    render_alert(st, analysis, enable_audio=enable_beep)
                    st.caption("Transcript")
                    st.write(text or "(Transcription disabled or empty)")

                results.append(
                    {
                        "file": f.name,
                        "duration_s": meta.duration_seconds,
                        "risk_score": analysis.risk_score,
                        "risk_level": analysis.risk_level,
                        "matched_patterns": analysis.matched_patterns,
                        "reasons": analysis.reasons,
                        "gemini_summary": analysis.gemini_summary,
                        "transcript": text,
                    }
                )
                st.session_state.history.append({**results[-1], "type": "file", "ts": time.time()})
            except Exception as e:
                st.error(f"Failed to process {getattr(f, 'name', 'file')}: {e}")
            finally:
                prog.progress(i / max(1, len(uploaded)))
        file_handler.cleanup_temp_paths(cleanup_dirs)

        st.download_button(
            "⬇️ Download analysis JSON",
            data=json.dumps({"generated_at": time.time(), "results": results}, indent=2),
            file_name="voice_scam_shield_results.json",
            mime="application/json",
        )


with history_tab:
    st.subheader("🧾 Session History")
    if not st.session_state.history:
        st.info("No history yet. Analyze a stream or upload files.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df.drop(columns=["transcript"]) if "transcript" in df.columns else df, use_container_width=True)
        st.download_button(
            "⬇️ Download history JSON",
            data=json.dumps(st.session_state.history, indent=2),
            file_name="voice_scam_shield_history.json",
            mime="application/json",
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.stream_scores = []
            st.session_state.stream_transcript = ""
            st.rerun()
