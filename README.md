# 🎙️ Speech Emotion Recognition

A deep learning system for classifying human emotions from speech audio — Happy, Sad, Angry, Calm, Neutral, Fearful, Disgust, and Surprised — built with a CNN + LSTM architecture in PyTorch, trained on MFCC features extracted from the RAVDESS dataset. Includes an interactive Streamlit web application for real-time emotion prediction from uploaded audio.

Built as part of the CodeAlpha Machine Learning Internship.

---

## 📌 Overview

The pipeline extracts MFCC (Mel-Frequency Cepstral Coefficients) from speech recordings using Librosa, trains a CNN + LSTM model to classify emotions, and serves predictions through a Streamlit web interface.

Capabilities:
- Upload and play back a WAV audio file
- Predict the speaker's emotion
- View confidence scores across all emotion classes

---

## 🧠 Model Architecture

Each 3-second audio clip is converted into a (130 × 40) MFCC feature sequence and passed through the following network:

```
Input (130 × 40 MFCC)
        │
        ▼
Conv1D (64, kernel=5, ReLU) → BatchNorm → MaxPooling → Dropout (0.3)
        │
        ▼
Conv1D (128, kernel=5, ReLU) → BatchNorm → MaxPooling → Dropout (0.3)
        │
        ▼
LSTM (128) → Dropout (0.4)
        │
        ▼
Dense (64, ReLU) → Dropout (0.3)
        │
        ▼
Dense (8 classes, Softmax)
```

**Training configuration**

| Component | Detail |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Regularization | Batch normalization, dropout, early stopping |

---

## 🎵 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 professional actors
- 8 emotion classes: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

Expected folder structure:

```
dataset/
├── Actor_01/
├── Actor_02/
├── Actor_03/
...
```

Also supported: TESS and EMO-DB — place audio files inside emotion-named folders.

**Offline demo mode:** if no dataset is detected, the project automatically generates a synthetic dataset so the full training pipeline can still run end-to-end.

---

## 🎼 Feature Extraction

Each audio file is:
- Resampled to 22.05 kHz
- Padded or trimmed to 3 seconds
- Converted into 40 MFCC coefficients
- Fixed to 130 time frames

The pipeline also generates waveform, MFCC, and mel spectrogram visualizations for each sample.

---

## 📁 Project Structure

```text
CodeAlpha_EmotionRecognitionFromSpeech/
├── dataset/
├── models/
├── screenshots/
├── app.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
└── emotion_recognition.ipynb
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/CodeAlpha_EmotionRecognitionFromSpeech.git
cd CodeAlpha_EmotionRecognitionFromSpeech

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --dataset_dir dataset

# Launch the Streamlit app
streamlit run app.py
```

---

## 📊 Results

Running `train.py` automatically generates:
- Waveform, MFCC, and spectrogram plots
- Emotion distribution chart
- Training history curves
- Confusion matrix
- Classification report

The included demo uses a synthetic dataset and achieves 100% accuracy, since the generated samples are intentionally separable. For realistic performance, replace the dataset with RAVDESS, TESS, or EMO-DB.

**Expected accuracy on RAVDESS: 60–75%**

---

## 🖼️ Screenshots

The `screenshots/` directory includes:
- Waveform
- MFCC
- Spectrogram
- Training curves
- Confusion matrix
- Classification report

---

## 🔮 Future Improvements

- Combine RAVDESS, TESS, and EMO-DB datasets
- Audio data augmentation
- Attention mechanism
- Wav2Vec 2.0 feature extractor
- Cloud deployment
- Mobile-friendly interface

---

## 👨‍💻 Author

**Irtaza Hyder**
Machine Learning Intern at CodeAlpha
Bachelor of Science in Computer Science (BSCS)

---

⭐ If you found this project useful, consider giving the repository a star.
