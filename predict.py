"""
predict.py
----------
Load the trained emotion recognition model (PyTorch CNN+LSTM) and label
encoder to predict the emotion expressed in a given WAV audio file.

CLI usage:
    python predict.py path/to/audio.wav
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from model import EmotionCNNLSTM, get_device
from utils import extract_sequence_from_path

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn_lstm.pt")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

_model = None
_label_encoder = None
_device = get_device()


def _check_artifacts_exist():
    missing = [p for p in [MODEL_PATH, ENCODER_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing trained artifacts: {missing}. Run train.py first."
        )


def load_artifacts():
    """Load and cache the PyTorch model and label encoder."""
    global _model, _label_encoder
    if _model is None:
        _check_artifacts_exist()
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        _model = EmotionCNNLSTM(checkpoint["num_classes"])
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(_device)
        _model.eval()
        with open(ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
    return _model, _label_encoder


def predict_emotion(wav_path: str):
    """
    Predict the emotion expressed in the given WAV file.

    Returns:
        predicted_label (str), confidence (float), probabilities (dict[str, float])
    """
    model, label_encoder = load_artifacts()

    sequence = extract_sequence_from_path(wav_path)
    sequence = np.expand_dims(sequence, axis=0)  # (1, MAX_LEN, N_MFCC)

    with torch.no_grad():
        tensor = torch.from_numpy(sequence.astype("float32")).to(_device)
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_idx = int(np.argmax(probabilities))
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = float(probabilities[predicted_idx])

    prob_dict = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    return predicted_label, confidence, prob_dict


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"Error: file not found: {wav_path}")
        sys.exit(1)

    label, confidence, probs = predict_emotion(wav_path)

    print(f"Predicted Emotion: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Emotion-wise probabilities:")
    for emotion, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {p * 100:.2f}%")


if __name__ == "__main__":
    main()
