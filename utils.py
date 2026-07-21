"""
utils.py
--------
Shared utilities for the Emotion Recognition from Speech project:
  - RAVDESS filename parsing
  - Dataset loading
  - Audio feature extraction (time-sequence MFCC for the CNN+LSTM model,
    plus Mel Spectrogram / waveform for visualization)
  - Visualization helpers (waveform, MFCC, spectrogram, emotion distribution)
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# RAVDESS label mapping
# Filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
# Emotion code (3rd part): 01=neutral 02=calm 03=happy 04=sad 05=angry
#                           06=fearful 07=disgust 08=surprised
# ---------------------------------------------------------------------------
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SAMPLE_RATE = 22050
N_MFCC = 40
DURATION = 3  # seconds, audio is padded/truncated to this length
MAX_LEN = 130  # fixed number of MFCC time frames fed to the CNN+LSTM model


def emotion_from_filename(filename: str) -> str:
    """Extract the emotion label from a RAVDESS-formatted filename."""
    parts = os.path.basename(filename).split("-")
    if len(parts) < 3:
        return "unknown"
    code = parts[2]
    return RAVDESS_EMOTION_MAP.get(code, "unknown")


def load_dataset_manifest(dataset_dir: str) -> pd.DataFrame:
    """
    Walk `dataset_dir` for .wav files and build a manifest DataFrame with
    columns: [path, emotion].

    Supports RAVDESS folder structure (Actor_XX/*.wav) as well as a flat
    directory of .wav files following the RAVDESS naming convention.
    Also supports TESS/EMO-DB style folders named after the emotion
    (falls back to using the parent folder name as the label when the
    RAVDESS filename pattern does not match).
    """
    wav_files = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
    records = []
    for wav_path in wav_files:
        emotion = emotion_from_filename(wav_path)
        if emotion == "unknown":
            # Fallback: use parent directory name as label (TESS/EMO-DB style)
            emotion = os.path.basename(os.path.dirname(wav_path)).lower()
        records.append({"path": wav_path, "emotion": emotion})
    return pd.DataFrame(records)


def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """Load an audio file, resample, and pad/truncate to a fixed duration."""
    signal, sample_rate = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]
    return signal, sample_rate


def extract_mfcc_sequence(signal, sr, n_mfcc=N_MFCC, max_len=MAX_LEN) -> np.ndarray:
    """
    Extract a time-sequence of MFCC frames shaped (max_len, n_mfcc), suitable
    as input to a CNN+LSTM model (time axis first, like a short "image" of
    frequency-over-time). Pads with zeros or truncates to a fixed length.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, frames)
    mfcc = mfcc.T  # (frames, n_mfcc)
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc.astype("float32")


def extract_sequence_from_path(path: str) -> np.ndarray:
    """Load an audio file from disk and extract its MFCC sequence."""
    signal, sr = load_audio(path)
    return extract_mfcc_sequence(signal, sr)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_waveform(signal, sr, save_path):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mfcc(signal, sr, save_path):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr)
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spectrogram(signal, sr, save_path):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_emotion_distribution(df: pd.DataFrame, save_path, title="Emotion Distribution in Dataset"):
    plt.figure(figsize=(8, 5))
    df["emotion"].value_counts().plot(kind="bar", color="steelblue")
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Synthetic demo dataset (fallback when no real dataset is present)
# ---------------------------------------------------------------------------
EMOTIONS = list(RAVDESS_EMOTION_MAP.values())

_SYNTH_PARAMS = {
    "neutral": (180, 0.30),
    "calm": (150, 0.20),
    "happy": (260, 0.60),
    "sad": (140, 0.25),
    "angry": (300, 0.80),
    "fearful": (280, 0.50),
    "disgust": (200, 0.35),
    "surprised": (320, 0.70),
}


def synthesize_signal(emotion: str, seed: int, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """
    Generate a synthetic speech-like waveform for a given emotion label.

    This is used only as an offline fallback demo dataset when no real
    audio dataset (e.g. RAVDESS) is present in `dataset/`, so the training
    pipeline and Streamlit app can still be exercised end-to-end and
    produce real (not fabricated) plots and metrics from *some* data.
    Real MFCC sequences are still extracted from this signal using the
    same `extract_mfcc_sequence` function used for real audio.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration))
    base_freq, amp = _SYNTH_PARAMS.get(emotion, (200, 0.4))
    freq_jitter = base_freq + rng.normal(0, 8, size=t.shape).cumsum() * 0.001
    signal = amp * np.sin(2 * np.pi * freq_jitter * t) + 0.05 * rng.randn(len(t))
    return signal.astype("float32"), sr


def generate_synthetic_dataset(n_per_class: int = 40):
    """
    Build an in-memory synthetic demo dataset: for each emotion class,
    generate `n_per_class` synthetic waveforms and extract real MFCC
    time-sequence features from each using librosa.

    Returns:
        X (np.ndarray): shape (N, MAX_LEN, N_MFCC) feature sequences
        y (np.ndarray): emotion string labels
        sample_signal, sample_sr, sample_emotion: one example signal for
            waveform/MFCC/spectrogram visualization
    """
    X, y = [], []
    sample_signal, sample_sr, sample_emotion = None, None, None
    for emotion in EMOTIONS:
        for i in range(n_per_class):
            seed = abs(hash(f"{emotion}_{i}")) % (2**31)
            signal, sr = synthesize_signal(emotion, seed)
            X.append(extract_mfcc_sequence(signal, sr))
            y.append(emotion)
            if sample_signal is None and emotion == "happy":
                sample_signal, sample_sr, sample_emotion = signal, sr, emotion
    return np.array(X), np.array(y), sample_signal, sample_sr, sample_emotion
