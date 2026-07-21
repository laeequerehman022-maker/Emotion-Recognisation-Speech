"""
model.py
--------
PyTorch CNN+LSTM architecture for the Emotion Recognition from Speech
project. Shared by train.py, predict.py, and app.py so the exact same
architecture is used for training and inference.

Architecture (Conv1D over MFCC time-frames feeding an LSTM):
    Conv1D(64) -> BatchNorm -> MaxPool -> Dropout
    Conv1D(128) -> BatchNorm -> MaxPool -> Dropout
    LSTM(128) -> Dropout
    Dense(64, relu) -> Dropout
    Dense(num_classes, softmax)

Input to the model is (batch, MAX_LEN, N_MFCC) -- time-major, matching the
MFCC sequences produced by utils.extract_mfcc_sequence. Internally this is
transposed to PyTorch's channels-first Conv1d layout (batch, N_MFCC,
MAX_LEN) and back to (batch, seq_len, channels) before the LSTM.
"""

import torch
import torch.nn as nn

from utils import N_MFCC, MAX_LEN


class EmotionCNNLSTM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(N_MFCC, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.4)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, MAX_LEN, N_MFCC) -> (batch, N_MFCC, MAX_LEN) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        # (batch, channels, seq_len) -> (batch, seq_len, channels) for LSTM
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # last layer's final hidden state, like Keras LSTM(...) default return
        x = self.lstm_dropout(x)
        x = self.head(x)
        return x  # raw logits; use nn.CrossEntropyLoss / torch.softmax


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
