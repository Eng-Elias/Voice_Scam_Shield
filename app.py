import json
import time
from datetime import datetime
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)
import av

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
    "Detect potential scams in real time from microphone streams or uploaded audio files. "
    "Transcribe with Whisper. Optionally analyze with Gemini for semantic cues."
)


# ---------- Stream Processor ----------
class TranscriberProcessor(AudioProcessorBase):
    def __init__(self, engine: WhisperEngine, detector: ScamDetector, settings: Settings):
        self.engine = engine
        self.detector = detector
        self.settings = settings
        self.buffer = np.zeros(0, dtype=np.float32)
        self.sample_rate = 16000
        self.chunk_samples = int(self.settings.chunk_seconds * self.sample_rate)
        self.last_transcript = ""
        self.last_analysis = None
        self.timeline: List[Dict[str, Any]] = []
        self.amplitude = 0.0
        self.wave: List[Dict[str, Any]] = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to mono float32 [-1,1]
        samples = frame.to_ndarray()
        sr = int(frame.sample_rate)
        if samples.ndim == 2:  # (channels, samples)
            samples = samples.mean(axis=0)
        if samples.dtype != np.float32:
            # whisper expects float32 [-1,1]
            samples = samples.astype(np.float32)
        # Rough normalization if integer-like range
        if samples.max() > 1.5:
            samples = samples / 32768.0
        self.amplitude = float(np.clip(np.abs(samples).mean() * 5, 0.0, 1.0))

        if sr != self.sample_rate:
            self.sample_rate = sr
            self.chunk_samples = int(self.settings.chunk_seconds * self.sample_rate)
        self.buffer = np.concatenate([self.buffer, samples])

        # Track amplitude timeline locally (thread-safe; no Streamlit calls here)
        self.wave.append({"t": time.time(), "amp": float(self.amplitude)})
        self.wave = self.wave[-300:]

        if len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[: self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples :]
            if self.engine.is_ready():
                text = self.engine.transcribe_stream_chunk(chunk, self.sample_rate)
            else:
                text = ""
            if text:
                self.last_transcript = (self.last_transcript + " " + text).strip()
                analysis = self.detector.analyze_text(self.last_transcript[-2000:])
                self.last_analysis = analysis
                self.timeline.append({
                    "t": time.time(),
                    "score": float(analysis.risk_score),
                    "level": analysis.risk_level,
                })
        return frame


# ---------- Tabs ----------
stream_tab, file_tab, history_tab = st.tabs(["Stream Analysis", "File Analysis", "History"])


with stream_tab:
    st.subheader("🎙️ Live Microphone Analysis")
    if not whisper_engine.is_ready():
        st.warning(
            "Whisper is not available. Install dependencies from requirements.txt to enable transcription."
        )

    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    ctx = webrtc_streamer(
        key="vss-stream",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        frontend_rtc_configuration=rtc_config,
        async_processing=False,
        audio_processor_factory=lambda: TranscriberProcessor(whisper_engine, scam_detector, settings),
    )

    # Require the user to click Start (and grant mic permission)
    if not ctx or not ctx.state.playing:
        st.info("Click Start above and allow microphone access. Use the Local URL (localhost) for mic streaming.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Real-time transcript")
        st.text_area(
            "Transcript",
            value=(ctx.audio_processor.last_transcript if ctx and ctx.audio_processor else ""),
            height=200,
            label_visibility="collapsed",
        )
    with col2:
        st.caption("Audio level")
        lvl = st.session_state.get("_lvl", 0.0)
        if ctx and ctx.audio_processor:
            lvl = float(ctx.audio_processor.amplitude)
            st.session_state["_lvl"] = lvl
        st.progress(lvl)

    # Mini waveform / level trace
    if ctx and ctx.audio_processor and ctx.audio_processor.wave:
        st.caption("Waveform (relative level)")
        dfw = pd.DataFrame([
            {"Time": pd.to_datetime(x["t"], unit="s"), "Level": x["amp"]}
            for x in ctx.audio_processor.wave
        ])
        wchart = (
            alt.Chart(dfw)
            .mark_line()
            .encode(x="Time:T", y=alt.Y("Level:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=120)
        )
        st.altair_chart(wchart, use_container_width=True)

    if ctx and ctx.audio_processor and ctx.audio_processor.last_analysis:
        st.caption("Live alert")
        render_alert(st, ctx.audio_processor.last_analysis, enable_audio=enable_beep)

    # Confidence over time chart
    if ctx and ctx.audio_processor and ctx.audio_processor.timeline:
        st.caption("Confidence over time")
        df = pd.DataFrame([
            {"Time": pd.to_datetime(x["t"], unit="s"), "Score": x["score"]}
            for x in ctx.audio_processor.timeline
        ])
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x="Time:T", y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)

    st.info(
        "Tip: Keep the tab active and your microphone enabled. Transcripts update in chunks (\n"
        f"~{settings.chunk_seconds:.0f}s)."
    )


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
