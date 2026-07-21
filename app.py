"""
app.py
------
Streamlit GUI for the Emotion Recognition from Speech project.

Users can:
  - Upload a WAV audio file
  - Play the uploaded audio
  - Get the predicted emotion, confidence score, and a full
    emotion-wise probability breakdown

Run with:
    streamlit run app.py
"""

import os
import tempfile
import pandas as pd
import streamlit as st

from predict import load_artifacts, predict_emotion

st.set_page_config(
    page_title="Emotion Recognition from Speech",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Emotion Recognition from Speech")
st.write(
    "A CNN + LSTM deep learning model (PyTorch), trained on MFCC "
    "features from the RAVDESS dataset, predicts the emotion expressed in "
    "a speech recording. Upload a WAV file below to try it out."
)

try:
    load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "Trained model artifacts not found. Please run `python train.py` "
        "first to generate the model and label encoder in `models/`."
    )

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔮 Predict Emotion", type="primary", disabled=not model_loaded):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            label, confidence, probs = predict_emotion(tmp_path)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Emotion", label.capitalize())
            with col2:
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            st.subheader("Probability Chart")
            prob_df = pd.DataFrame({
                "Emotion": [e.capitalize() for e in probs.keys()],
                "Probability": list(probs.values()),
            }).set_index("Emotion").sort_values("Probability", ascending=False)
            st.bar_chart(prob_df)

            st.subheader("Emotion-wise Probability Breakdown")
            st.dataframe(
                prob_df.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
            )
        finally:
            os.remove(tmp_path)

st.divider()
st.caption("CodeAlpha Machine Learning Internship — Emotion Recognition from Speech")
