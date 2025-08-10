# Voice Scam Shield: 1-Page Project Report

---

### **Project:** Voice Scam Shield
### **Author:** [Your Name]
### **Date:** 2025-08-10

---

## 1. The Challenge

Voice-based scams and phishing attempts pose a significant and growing threat, especially to vulnerable populations. Scammers exploit trust and create false urgency to trick individuals into compromising personal information, making unauthorized payments, or granting remote access to their devices. The challenge is to provide a tool that is accessible, easy to use, and effective at detecting these threats in real-time, empowering users to protect themselves before they become victims.

## 2. Solution & Tools Used

**Voice Scam Shield** is a real-time audio analysis tool designed to identify potential scams during a conversation. It operates as a Streamlit web application, providing an intuitive interface for both live microphone analysis and batch processing of uploaded audio files.

The core of the solution is a dual-analysis pipeline:

1.  **Speech-to-Text Transcription**: Audio is first converted into text using **OpenAI's Whisper model**.
2.  **Hybrid Scam Detection**:
    *   **Pattern Matching**: The transcript is scanned against a curated list of high-risk keywords and phrases (e.g., "gift card," "urgent payment"). This provides a rapid, initial risk assessment.
    *   **AI-Powered Analysis**: For deeper contextual understanding, the transcript is sent to **Google's Gemini model**. The AI evaluates the conversation's intent, tone, and semantics to identify sophisticated scam tactics that keywords alone might miss.

The application then presents a clear, color-coded risk level (Low, Medium, High), a confidence score, and actionable recommendations to the user.

### **Technology Stack:**

*   **Programming Language**: Python 3.11
*   **Framework**: Streamlit (for the web interface)
*   **Speech-to-Text**: OpenAI Whisper (`base` model)
*   **AI Analysis**: Google Gemini API (`gemini-2.0-flash-lite` model)
*   **Audio Processing**: Pydub, NumPy
*   **Dependencies**: `streamlit-mic-recorder` for live audio capture, `python-dotenv` for environment management.

## 3. Successes

*   **Dual-Detection Method**: Combining rapid pattern matching with advanced AI analysis creates a robust system that is both fast and intelligent.
*   **Clear & Intuitive UI**: The Streamlit interface is clean and easy to navigate. Visual alerts, risk scores, and live charts make the analysis results immediately understandable.
*   **Flexible Input**: The tool supports both live audio streams and uploads of common audio formats (MP3, WAV, M4A, FLAC), making it versatile for different use cases.
