# Voice Scam Shield

Real-time voice scam detection for uploaded audio files and live microphone streams.

Detect common scam patterns and leverage Google Gemini for advanced, AI-powered analysis. Transcribe speech via OpenAI Whisper. Built with Streamlit for rapid prototyping and live dashboards.

## Key Features
- Real-time audio stream analysis (microphone)
- Audio file upload and batch analysis
- AI-powered scam detection via Google Gemini
- Real-time transcription with OpenAI Whisper
- Visual alerts with confidence scoring
- Support for MP3, WAV, M4A, FLAC
- Export analysis results

## Tech Stack
- Python
- Streamlit + streamlit-webrtc
- OpenAI Whisper (speech-to-text)
- Google Gemini API (semantic analysis)

## Quick Start

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

If PyTorch fails to install via the default index, install the CPU-only wheels explicitly:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

3) Configure environment

- Copy `.env.example` to `.env` and fill in your keys.
- At minimum, set `GOOGLE_API_KEY` to use Gemini analysis.

```bash
copy .env.example .env  # Windows
```

4) Run the app

```bash
streamlit run app.py
```

5) Optional: FFmpeg for wider audio support

This project uses pydub and a portable FFmpeg binary via `imageio-ffmpeg`. No system install is required. If you prefer your own FFmpeg install, ensure it's on PATH.

## Project Structure

```
Voice_Scam_Shield/
├─ app.py
├─ requirements.txt
├─ README.md
├─ FEATURES.md
├─ .env.example
├─ config/
│  ├─ __init__.py
│  └─ settings.py
└─ src/
   ├─ __init__.py
   ├─ audio_processor.py
   ├─ scam_detector.py
   ├─ alert_system.py
   └─ file_handler.py
```

## Notes & Constraints
- Whisper models require PyTorch; on CPU, start with the `tiny` or `base` model for speed.
- Live streaming quality depends on network and browser audio constraints.
- Gemini analysis requires a valid API key. The app will gracefully fallback to pattern-only detection if not configured.

## Help / Tips
- Use `.env` to set `GOOGLE_API_KEY` and adjust sensitivity.
- If streaming doesn't work, try a different browser or fallback to File Analysis.
- For faster transcription on CPU, set `WHISPER_MODEL=tiny` in `.env`.

## License
Hackathon / prototype use. Add your preferred license here.
