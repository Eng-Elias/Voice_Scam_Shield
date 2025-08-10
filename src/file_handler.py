# This script provides utilities for handling audio files.
# It supports loading audio from uploaded files or local paths, converting them to a standard format (16kHz mono WAV),
# and managing temporary files to ensure the application remains clean.

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg


# Configure pydub to use a portable FFmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


SUPPORTED_INPUT_EXTS = {".mp3", ".wav", ".m4a", ".flac"}


@dataclass
class FileMeta:
    """Dataclass to store metadata about a processed audio file."""
    path: str
    original_name: str
    duration_seconds: float
    channels: int
    frame_rate: int
    sample_width: int


def _safe_suffix(name: str) -> str:
    base = os.path.basename(name)
    return base.replace(" ", "_")


def validate_extension(filename: str) -> bool:
    """Checks if a file has a supported audio extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_INPUT_EXTS


def load_audiosegment_from_uploaded(file) -> AudioSegment:
    """Loads a pydub AudioSegment from a file-like object (e.g., Streamlit's UploadedFile)."""
    # `file` can be a Streamlit UploadedFile or any file-like object
    raw = file.read() if hasattr(file, "read") else file
    bio = io.BytesIO(raw)
    return AudioSegment.from_file(bio)


def load_audiosegment_from_path(path: str) -> AudioSegment:
    """Loads a pydub AudioSegment from a local file path."""
    return AudioSegment.from_file(path)


def convert_to_wav_mono_16k(seg: AudioSegment) -> Tuple[str, FileMeta]:
    """Converts an AudioSegment to a 16kHz mono WAV file, saved in a temporary directory."""
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM
    tmpdir = tempfile.mkdtemp(prefix="vss_")
    out_path = os.path.join(tmpdir, "converted.wav")
    seg.export(out_path, format="wav")
    meta = FileMeta(
        path=out_path,
        original_name=os.path.basename(out_path),
        duration_seconds=len(seg) / 1000.0,
        channels=seg.channels,
        frame_rate=seg.frame_rate,
        sample_width=seg.sample_width,
    )
    return out_path, meta


def process_uploaded_file(file) -> Tuple[str, FileMeta]:
    """Processes an uploaded audio file, converting it to the standard WAV format."""
    if not hasattr(file, "name"):
        raise ValueError("Invalid uploaded file")
    if not validate_extension(file.name):
        raise ValueError("Unsupported file type. Allowed: MP3, WAV, M4A, FLAC")
    seg = load_audiosegment_from_uploaded(file)
    return convert_to_wav_mono_16k(seg)


def process_file_path(path: str) -> Tuple[str, FileMeta]:
    """Processes a local audio file path, converting it to the standard WAV format."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not validate_extension(path):
        raise ValueError("Unsupported file type. Allowed: MP3, WAV, M4A, FLAC")
    seg = load_audiosegment_from_path(path)
    return convert_to_wav_mono_16k(seg)


def audiosegment_from_ndarray(data: np.ndarray, sample_rate: int, channels: int) -> AudioSegment:
    """Creates a pydub AudioSegment from a NumPy array."""
    """
    data: float32 or int16 numpy array. Shape: (samples,) for mono or (channels, samples).
    Converts to 16-bit PCM AudioSegment.
    """
    if data.ndim == 2:
        # (channels, samples) -> mono by averaging
        data = data.mean(axis=0)
    # Normalize to int16
    if data.dtype != np.int16:
        # assume float32 in [-1, 1]
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767.0).astype(np.int16)
    raw = data.tobytes()
    seg = AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    return seg


def cleanup_temp_paths(paths: List[str]) -> None:
    """Removes temporary directories and files created during file processing."""
    for p in paths:
        try:
            if os.path.isdir(p):
                # remove directory tree
                for root, dirs, files in os.walk(p, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                os.rmdir(p)
            elif os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
