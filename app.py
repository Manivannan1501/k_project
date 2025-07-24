# Full Streamlit app with:
# - SMOTE oversampling
# - Audio file support
# - Feature selection & hyperparameter tuning

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Voice Gender Classifier", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

@st.cache_resource
def load_model():
    return joblib.load("voice_gender_classifier_all_features.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

df = load_data()
model = load_model()
scaler = load_scaler()

menu = st.sidebar.radio("Navigate", ["Introduction", "Classification", "Audio Upload", "Model Retraining"])

if menu == "Introduction":
    st.title("🎙️ Voice Gender Classification")
    st.write("This app classifies audio as male or female using ML techniques.")

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification")

    st.sidebar.subheader("Feature Selection & SVM Tuning")
    top_n = st.sidebar.slider("Number of Features", 5, len(df.columns)-1, 10)
    selector_method = st.sidebar.selectbox("Feature Selector", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C_val = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0)
    gamma_val = st.sidebar.selectbox("Gamma", ["scale", "auto"])

    X = df.drop(columns=["label"])
    y = df["label"]

    # Apply SMOTE
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Feature selection
    if selector_method == "ANOVA F-test":
        selector = SelectKBest(score_func=f_classif, k=top_n)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=top_n)

    X_selected = selector.fit_transform(X_resampled, y_resampled)
    selected_features = X.columns[selector.get_support()].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

    clf = SVC(kernel=kernel, C=C_val, gamma=gamma_val, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Test Accuracy: {acc:.2%}")

    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax, display_labels=["Female", "Male"], cmap="Blues")
    st.pyplot(fig)

    st.subheader("Predict from Manual Input")
    user_input = []
    for col in X.columns:
        if col in selected_features:
            val = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()), step=0.01)
            user_input.append(val)
        else:
            user_input.append(df[col].mean())

    if st.button("Predict Gender"):
        scaled_input = scaler.transform([user_input])
        selected_input = selector.transform(scaled_input)
        pred = clf.predict(selected_input)[0]
        gender = "👨 Male" if pred == 1 else "👩 Female"
        st.success(f"Predicted: {gender}")

elif menu == "Audio Upload":
    st.title("🎤 Predict Gender from Audio")
    audio_file = st.file_uploader("Upload a .wav file", type=["wav"])

    def extract_features(y, sr):
        features = {
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
            "rms_energy": np.mean(librosa.feature.rms(y=y)),
            "log_energy": np.log(np.sum(y ** 2) + 1e-6),
            "mean_pitch": np.mean(librosa.piptrack(y=y, sr=sr)[0]),
            "std_pitch": np.std(librosa.piptrack(y=y, sr=sr)[0]),
            "mean_spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        }
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        for i in range(mfccs.shape[0]):
            features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        return features

    if audio_file is not None:
        y, sr = librosa.load(audio_file, sr=None)
        try:
            feat = extract_features(y, sr)
            feature_df = pd.DataFrame([feat])
            feature_df_full = pd.DataFrame(columns=df.drop(columns=["label"]).columns)
            for col in feature_df.columns:
                feature_df_full[col] = feature_df[col]
            feature_df_full = feature_df_full.fillna(df.mean())
            input_scaled = scaler.transform(feature_df_full)
            prediction = model.predict(input_scaled)
            label = "👨 Male" if prediction[0] == 1 else "👩 Female"
            st.success(f"Predicted Gender: **{label}**")
        except Exception as e:
            st.error(f"Failed to process audio: {e}")

elif menu == "Model Retraining":
    st.title("🔄 Model Retraining")
    st.write("This will retrain the SVM classifier with selected options.")
    if st.button("Retrain Model"):
        try:
            X = df.drop(columns=["label"])
            y = df["label"]
            X_scaled = scaler.fit_transform(X)
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
            clf.fit(X_resampled, y_resampled)
            joblib.dump(clf, "voice_gender_classifier_all_features.pkl")
            st.success("Model retrained and saved successfully.")
        except Exception as e:
            st.error(f"Retraining failed: {e}")
