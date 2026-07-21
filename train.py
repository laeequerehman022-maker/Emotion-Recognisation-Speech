"""
train.py
--------
Trains a CNN + LSTM deep learning model (PyTorch) to classify emotions
from speech audio using the RAVDESS dataset. Saves the trained model and
label encoder to `models/`, and writes evaluation plots to `screenshots/`.

Architecture (Conv1D over MFCC time-frames feeding an LSTM):
    Conv1D(64) -> BatchNorm -> MaxPool -> Dropout
    Conv1D(128) -> BatchNorm -> MaxPool -> Dropout
    LSTM(128) -> Dropout
    Dense(64, relu) -> Dropout
    Dense(num_classes, softmax)

Each audio clip is represented as a (time_frames, n_mfcc) sequence of MFCC
coefficients extracted with librosa -- the CNN layers learn local
spectro-temporal patterns, and the LSTM layer models how those patterns
evolve over the length of the utterance.

Expected dataset layout (default RAVDESS structure):
    dataset/
      Actor_01/*.wav
      Actor_02/*.wav
      ...

If no .wav files are found in `dataset/`, the script automatically falls
back to an in-memory synthetic demo dataset (see utils.generate_synthetic_dataset)
so the full pipeline can still be run end-to-end and produce real plots
and metrics. Replace `dataset/` with actual RAVDESS/TESS/EMO-DB audio to
get authentic results.

Usage:
    python train.py --dataset_dir dataset
"""

import os
import argparse
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import EmotionCNNLSTM, get_device
from utils import (
    load_dataset_manifest,
    extract_sequence_from_path,
    generate_synthetic_dataset,
    plot_emotion_distribution,
    plot_waveform,
    plot_mfcc,
    plot_spectrogram,
    load_audio,
    N_MFCC,
    MAX_LEN,
)

MODELS_DIR = "models"
SCREENSHOTS_DIR = "screenshots"
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn_lstm.pt")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = get_device()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["loss"], label="Training Loss", color="crimson")
    axes[0].plot(history["val_loss"], label="Validation Loss", color="darkorange")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="Training Accuracy", color="seagreen")
    axes[1].plot(history["val_accuracy"], label="Validation Accuracy", color="royalblue")
    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "training_history.png"), dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def train_model(model, train_loader, val_loader, epochs=60, patience=8):
    """Train with Adam + CrossEntropyLoss, manual early stopping that
    restores the best validation-loss weights (mirrors Keras'
    EarlyStopping(monitor='val_loss', patience=..., restore_best_weights=True))."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            running_correct += (outputs.argmax(1) == yb).sum().item()
            running_total += xb.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (outputs.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} "
              f"- val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val_loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset",
                         help="Path to RAVDESS (or compatible) dataset directory")
    parser.add_argument("--epochs", type=int, default=60,
                         help="Max training epochs")
    args = parser.parse_args()

    print(f"Scanning dataset directory: {args.dataset_dir}")
    manifest = load_dataset_manifest(args.dataset_dir)

    used_synthetic = False
    if manifest.empty:
        used_synthetic = True
        print(f"No .wav files found in '{args.dataset_dir}'.")
        print("Falling back to an in-memory SYNTHETIC demo dataset so the "
              "pipeline can still run end-to-end. Add real RAVDESS/TESS/"
              "EMO-DB audio to 'dataset/' and re-run for authentic results.")
        X, y_raw, sample_signal, sample_sr, sample_emotion = generate_synthetic_dataset(n_per_class=40)
        dist_df = pd.DataFrame({"emotion": y_raw})
        plot_emotion_distribution(
            dist_df, os.path.join(SCREENSHOTS_DIR, "emotion_distribution.png"),
            title="Emotion Distribution (Synthetic Demo Dataset)",
        )
        plot_waveform(sample_signal, sample_sr, os.path.join(SCREENSHOTS_DIR, "waveform.png"))
        plot_mfcc(sample_signal, sample_sr, os.path.join(SCREENSHOTS_DIR, "mfcc.png"))
        plot_spectrogram(sample_signal, sample_sr, os.path.join(SCREENSHOTS_DIR, "spectrogram.png"))
    else:
        print(f"Found {len(manifest)} audio files across "
              f"{manifest['emotion'].nunique()} emotions.")
        plot_emotion_distribution(manifest, os.path.join(SCREENSHOTS_DIR, "emotion_distribution.png"))

        sample_path = manifest.iloc[0]["path"]
        sample_signal, sample_sr = load_audio(sample_path)
        plot_waveform(sample_signal, sample_sr, os.path.join(SCREENSHOTS_DIR, "waveform.png"))
        plot_mfcc(sample_signal, sample_sr, os.path.join(SCREENSHOTS_DIR, "mfcc.png"))
        plot_spectrogram(sample_signal, sample_sr, os.path.join(SCREENSHOTS_DIR, "spectrogram.png"))

        print("Extracting MFCC sequences for all samples (this may take a while)...")
        sequences = []
        for i, path in enumerate(manifest["path"]):
            sequences.append(extract_sequence_from_path(path))
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(manifest)}")
        X = np.array(sequences)
        y_raw = manifest["emotion"].values

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print("Building CNN+LSTM model...")
    model = EmotionCNNLSTM(num_classes).to(DEVICE)
    print(model)

    X_train_t = torch.from_numpy(X_train.astype("float32"))
    y_train_t = torch.from_numpy(y_train.astype("int64"))
    X_test_t = torch.from_numpy(X_test.astype("float32"))
    y_test_t = torch.from_numpy(y_test.astype("int64"))

    full_train_ds = TensorDataset(X_train_t, y_train_t)
    val_size = int(0.15 * len(full_train_ds))
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(
        full_train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    print("Training model...")
    history = train_model(model, train_loader, val_loader, epochs=args.epochs, patience=8)

    plot_training_history(history)

    print("Evaluating on test set...")
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            all_preds.append(outputs.argmax(1).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(SCREENSHOTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
        if used_synthetic:
            f.write(
                "\n\nNOTE: This run used an in-memory SYNTHETIC demo dataset "
                "because no real audio files were found in the dataset "
                "directory. Add real RAVDESS (or TESS/EMO-DB) .wav files to "
                "'dataset/' and re-run train.py for authentic results.\n"
            )

    plot_confusion(y_test, y_pred, class_names)

    print("Saving model and label encoder...")
    torch.save({"model_state_dict": model.state_dict(), "num_classes": num_classes}, MODEL_PATH)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print("Done. Model and evaluation artifacts saved.")


if __name__ == "__main__":
    main()
