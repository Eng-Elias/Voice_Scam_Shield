# Voice Scam Shield - Features

This document details the features implemented for both file-based analysis and live stream analysis. The goal is a robust, hackathon-friendly MVP with room to expand.

## 1) Real-time Stream Analysis
- Live microphone capture via WebRTC (browser) using `streamlit-webrtc`.
- Rolling-buffer transcription in short chunks (configurable, default 4–6s).
- Incremental scam risk analysis per chunk with aggregated confidence over time.
- Visual alerts with color coding and live confidence meter.
- Resilient to missing API keys: pattern detection works without Gemini.

## 2) File Upload & Batch Analysis
- Drag-and-drop upload of audio files: `MP3`, `WAV`, `M4A`, `FLAC`.
- File validation, conversion, and normalization to 16kHz mono WAV.
- Whisper-based transcription with pluggable model size (default `base`).
- Batch processing with progress indication and per-file result cards.
- Export combined analysis results as JSON.

## 3) AI-powered Scam Detection
- Pattern-based detection for common scam indicators:
  - “gift card”, “bank account”, “one-time password”, “social security”, “bitcoin”, “wire transfer”, “remote access”, “pay immediately”, etc.
- Gemini-based semantic analysis for nuanced contexts and intent.
- Confidence scoring algorithm that blends pattern hits and Gemini output.
- Risk levels: `low`, `medium`, `high`.

## 4) Transcription Engine
- OpenAI Whisper integration (local inference) for:
  - Audio file transcription.
  - Periodic chunk transcription in live mode.
- Recommended models for CPU-only: `tiny` or `base` for speed.

## 5) Alert & Recommendations
- Color-coded alerts: green (low), yellow (medium), red (high).
- Contextual recommendations (e.g., “Do not share OTP”, “Verify caller identity”).
- Optional audio alert for high-risk events (can be extended in UI).

## 6) Reporting & History
- Session history of transcripts and risk assessments.
- Export analysis results for auditing and sharing.

## 7) Configuration & Extensibility
- Centralized config in `config/settings.py` including:
  - Audio settings (sample rate, channels).
  - Pattern lists and sensitivity thresholds.
  - Gemini model and API key.
- Easy to add patterns or tweak weights and thresholds.

## 8) Notes & Limitations
- Whisper requires PyTorch; installation size can be large.
- Live streaming quality depends on mic/device/browser; if not supported, use file analysis.
- Gemini results depend on API quota and network.
- This is an MVP for a hackathon; security-hardening and extensive testing are future work.
