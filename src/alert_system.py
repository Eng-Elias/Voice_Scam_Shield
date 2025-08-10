# This script manages the visual and audio alerting system for the Streamlit user interface.
# It defines different alert styles based on risk levels (low, medium, high) and provides recommendations.

from dataclasses import dataclass
from typing import List
import time
import base64
import io
import wave
import numpy as np

from config.settings import Settings
from .scam_detector import ScamAnalysis


@dataclass
class AlertStyle:
    """Represents the visual style for a specific alert level."""
    color: str
    bg: str
    emoji: str


LEVEL_STYLE = {
    "low": AlertStyle(color="#2e7d32", bg="#e8f5e9", emoji="✅"),
    "medium": AlertStyle(color="#f9a825", bg="#fff8e1", emoji="⚠️"),
    "high": AlertStyle(color="#c62828", bg="#ffebee", emoji="⛔"),
}


def recommendations(analysis: ScamAnalysis) -> List[str]:
    """Generates a list of security recommendations based on the analysis results."""
    recs: List[str] = []
    text = analysis.text.lower()
    if any(k in text for k in ["otp", "one-time password", "password"]):
        recs.append("Never share one-time passwords or account passwords.")
    if any(k in text for k in ["gift card", "bitcoin", "crypto", "wire transfer", "western union"]):
        recs.append("Do not send money via gift cards, crypto, or wire transfers on request.")
    if any(k in text for k in ["remote access", "teamviewer", "anydesk", "logmein"]):
        recs.append("Do not grant remote access to your device to unknown callers.")
    if any(k in text for k in ["verify your account", "confirm your identity", "bank account", "routing number", "ssn", "social security"]):
        recs.append("Never share personal or banking details over unsolicited calls.")
    if analysis.risk_level != "low":
        recs.append("Hang up and independently verify the caller using official contact channels.")
    if not recs:
        recs.append("Be cautious. Verify the caller and avoid sharing sensitive data.")
    return recs


def render_alert(st, analysis: ScamAnalysis, enable_audio: bool = False, throttle_seconds: float = 2.0) -> None:
    """Renders a visual alert component in the Streamlit interface based on the scam analysis."""
    style = LEVEL_STYLE.get(analysis.risk_level, LEVEL_STYLE["low"])
    st.markdown(
        f"<div style='padding:0.9rem;border-radius:8px;background:{style.bg};border-left:6px solid {style.color};'>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
        f"<span style='font-size:1.4rem'>{style.emoji}</span>"
        f"<div><strong style='color:{style.color};text-transform:uppercase;'>Risk: {analysis.risk_level}</strong> "
        f"<span style='color:#333'>(score: {analysis.risk_score:.2f})</span></div>"
        f"</div>"
        f"<div style='margin-top:0.5rem;color:#333;'>"
        f"{analysis.gemini_summary or 'Pattern-based evaluation'}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, max(0.0, analysis.risk_score)))

    cols = st.columns(2)
    with cols[0]:
        if analysis.matched_patterns:
            st.caption("Matched patterns")
            st.write(", ".join(sorted(set(analysis.matched_patterns))))
        if analysis.reasons:
            st.caption("Reasons (AI)")
            for r in analysis.reasons[:5]:
                st.write(f"- {r}")
    with cols[1]:
        st.caption("Recommendations")
        for rec in recommendations(analysis)[:5]:
            st.write(f"- {rec}")

    # Optional audio alert for HIGH risk
    if enable_audio and analysis.risk_level == "high":
        now = time.time()
        last = st.session_state.get("_last_beep_ts", 0.0)
        if now - last >= float(throttle_seconds):
            st.session_state["_last_beep_ts"] = now

            # Generate a short beep (sine wave) and autoplay it via hidden audio tag
            sr = 16000
            dur = 0.2
            freq = 880.0
            t = np.arange(int(sr * dur), dtype=np.float32)
            sig = (np.sin(2 * np.pi * freq * t / sr) * 0.25).astype(np.float32)
            pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            st.markdown(
                f"<audio autoplay style='display:none'><source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>",
                unsafe_allow_html=True,
            )
