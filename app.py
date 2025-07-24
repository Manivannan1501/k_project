import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os
from sklearn.preprocessing import StandardScaler

# Load the trained model pipeline
model = joblib.load("voice_gender_classifier_all_features.pkl")

# Expected feature names (must match those used during training)
feature_names = model.named_steps['classifier'].n_features_in_

st.set_page_config(page_title="Voice Gender Classifier", layout="wide")
st.title("🤖 Voice Gender Classification using SVM")

st.sidebar.header("📁 Upload Options")
input_mode = st.sidebar.radio("Choose input method", ("Manual Input", "Upload CSV", "Upload Audio (.wav)"))

# Function to extract features from audio file
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        features = {}
        features['mean_pitch'] = np.mean(librosa.yin(y, fmin=50, fmax=300))
        features['std_pitch'] = np.std(librosa.yin(y, fmin=50, fmax=300))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['rms_energy'] = np.mean(librosa.feature.rms(y))
        features['log_energy'] = np.log(np.sum(y ** 2))
        features['mean_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        for i in range(1, 6):
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i - 1])

        # Fill missing features with 0 or placeholder
        for feat in feature_names:
            if feat not in features:
                features[feat] = 0.0

        return features
    except Exception as e:
        st.error(f"Failed to process audio file: {e}")
        return None

if input_mode == "Manual Input":
    st.subheader("🎛️ Enter Feature Values Manually")
    input_data = {}
    col1, col2 = st.columns(2)
    for i, feat in enumerate(feature_names):
        with (col1 if i % 2 == 0 else col2):
            input_data[feat] = st.number_input(f"{feat}", value=0.0)

    if st.button("🔍 Predict Gender"):
        try:
            input_df = pd.DataFrame([input_data])[feature_names]
            prediction = model.predict(input_df)[0]
            label = "Female" if prediction == 1 else "Male"
            st.success(f"🧠 Predicted Gender: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif input_mode == "Upload CSV":
    st.subheader("📄 Upload a CSV file with features")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            if not all(feat in df.columns for feat in feature_names):
                missing = set(feature_names) - set(df.columns)
                st.error(f"Missing required features: {missing}")
            else:
                predictions = model.predict(df[feature_names])
                df['Predicted Gender'] = ["Female" if p == 1 else "Male" for p in predictions]
                st.dataframe(df)
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", data=csv_output, file_name="gender_predictions.csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

elif input_mode == "Upload Audio (.wav)":
    st.subheader("🎙️ Upload an Audio File (.wav)")
    audio_file = st.file_uploader("Upload .wav file", type=["wav"])

    if audio_file:
        try:
            temp_path = os.path.join("temp.wav")
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())

            features = extract_audio_features(temp_path)
            if features:
                input_df = pd.DataFrame([features])[feature_names]
                prediction = model.predict(input_df)[0]
                label = "Female" if prediction == 1 else "Male"
                st.audio(temp_path, format="audio/wav")
                st.success(f"🧠 Predicted Gender: {label}")
        except Exception as e:
            st.error(f"Failed to process audio file: {e}")
