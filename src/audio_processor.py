# This script provides an audio processing and transcription engine using OpenAI's Whisper model.
# It is designed to handle both file-based transcription and real-time audio stream processing.
# The class encapsulates the logic for loading the Whisper model, resampling audio, and performing transcription.

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import Settings

try:
    import whisper  # type: ignore
    _WHISPER_AVAILABLE = True
except Exception:
    whisper = None  # type: ignore
    _WHISPER_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Dataclass to hold the results of a transcription task."""
    text: str
    segments: List[Dict[str, Any]]


class AudioProcessor:
    """
    A wrapper class for the Whisper transcription model.

    This class manages loading the Whisper model, and provides methods to transcribe
    both audio files and raw audio chunks from a stream.
    """
    """
    Audio processing and transcription using OpenAI Whisper.
    - File transcription
    - Streaming chunk transcription (numpy arrays)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_lock = threading.Lock()

    def is_ready(self) -> bool:
        """Checks if the Whisper library is installed and available."""
        return _WHISPER_AVAILABLE

    def _ensure_model(self) -> None:
        """Loads the Whisper model into memory if it hasn't been loaded yet."""
        if not _WHISPER_AVAILABLE:
            raise RuntimeError(
                "Whisper is not available. Please install dependencies from requirements.txt"
            )
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    # Load once, use CPU by default. Users can choose model via env.
                    self._model = whisper.load_model(self.settings.whisper_model)

    # ---------- Utilities ----------
    @staticmethod
    def _to_mono_float32(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            # channels x samples
            x = x.mean(axis=0)
        # convert to float32 in [-1, 1]
        if x.dtype == np.int16:
            x = (x.astype(np.float32)) / 32768.0
        elif x.dtype == np.float32:
            pass
        else:
            x = x.astype(np.float32)
            # assume already roughly [-1,1]
        # normalize small DC offset if any
        if len(x) > 0:
            x = x - np.mean(x)
            peak = np.max(np.abs(x)) + 1e-6
            x = x / max(1.0, peak)
        return x

    @staticmethod
    def _resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or len(x) == 0:
            return x
        duration = len(x) / float(orig_sr)
        n_target = int(round(duration * target_sr))
        if n_target <= 0:
            return np.zeros(0, dtype=np.float32)
        t_orig = np.linspace(0.0, duration, num=len(x), endpoint=False)
        t_target = np.linspace(0.0, duration, num=n_target, endpoint=False)
        y = np.interp(t_target, t_orig, x).astype(np.float32)
        return y

    # ---------- File transcription ----------
    def transcribe_file(self, wav_path: str) -> TranscriptionResult:
        """Transcribes a given audio file and returns the transcription result."""
        """Transcribe an audio file path (preferably 16kHz mono WAV)."""
        self._ensure_model()
        result = self._model.transcribe(wav_path, fp16=False)
        text = result.get("text", "").strip()
        segments = result.get("segments", []) or []
        return TranscriptionResult(text=text, segments=segments)

    # ---------- Streaming chunk transcription ----------
    def transcribe_stream_chunk(
            self, audio_chunk: np.ndarray, chunk_sample_rate: int
    ) -> str:
        """Transcribes a single chunk of audio from a stream."""
        """
        Transcribe a short chunk of audio from a live stream.
        audio_chunk: np array of shape (samples,) or (channels, samples)
        chunk_sample_rate: sample rate of the chunk
        Returns text for the chunk (can be empty).
        """
        self._ensure_model()
        x = self._to_mono_float32(audio_chunk)
        x = self._resample_linear(x, chunk_sample_rate, 16000)
        if len(x) < 16000 * 0.5:
            # too short (<0.5s), skip to reduce overhead
            return ""
        # Whisper transcribe accepts numpy array at 16kHz in float32 [-1,1]
        out = self._model.transcribe(x, fp16=False, language=None)
        return (out.get("text") or "").strip()
