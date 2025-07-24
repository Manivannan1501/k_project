# Combined app.py with Audio Upload & Retraining

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import librosa
import os
import tempfile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Human Voice Classification & Clustering", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Load model & scaler if available
model_path = "voice_gender_classifier_all_features.pkl"
scaler_path = "scaler.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    model = None
    scaler = None

menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering", "Audio Upload", "Retrain Model"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.write("""
    This project analyzes human voices to classify gender and perform clustering using audio features.
    Features can be extracted from CSV or WAV files.
    """)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df['gender'] = df['label'].map({0: 'Female', 1: 'Male'})
    st.subheader("Gender Distribution")
    st.bar_chart(df['gender'].value_counts())
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification using SVM")
    top_n = st.sidebar.slider("Select Top N Features", 5, len(df.columns)-1, 10)
    selector_method = st.sidebar.selectbox("Feature Selection", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C_value = st.sidebar.slider("C Value", 0.01, 10.0, 1.0)
    gamma_value = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_grid = st.sidebar.checkbox("Use Grid Search")

    X = df.drop(columns=["label"])
    y = df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if selector_method == "ANOVA F-test":
        selector = SelectKBest(score_func=f_classif, k=top_n)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=top_n)

    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

    if use_grid:
        param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["linear", "rbf"]}
        grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, cv=3)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = SVC(kernel=kernel, C=C_value, gamma=gamma_value, class_weight="balanced")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Accuracy: {acc:.2%}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Female", "Male"], ax=ax)
    st.pyplot(fig)

    st.subheader("🎛️ Predict with Manual Input")
    input_values = []
    for feature in X.columns:
        val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        input_values.append(val)

    if st.button("🔍 Predict"):
        try:
            input_scaled = scaler.transform([input_values])
            input_selected = selector.transform(input_scaled)
            result = model.predict(input_selected)
            gender = "Male" if result[0] == 1 else "Female"
            st.success(f"Predicted Gender: {gender}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

elif menu == "Audio Upload":
    st.title("🎤 Predict Gender from Audio")
    audio_file = st.file_uploader("Upload WAV file", type=[".wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None)
            features = {
                "mean_pitch": np.mean(librosa.piptrack(y=y, sr=sr)[0]),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "rms_energy": np.mean(librosa.feature.rms(y=y)),
                "log_energy": np.mean(np.log1p(librosa.feature.rms(y=y))),
                "mfcc_1_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr)[0]),
            }
            input_df = pd.DataFrame([features])
            input_scaled = scaler.transform(input_df)
            input_selected = selector.transform(input_scaled)
            prediction = model.predict(input_selected)
            gender = "Male" if prediction[0] == 1 else "Female"
            st.success(f"Predicted Gender: {gender}")
        except Exception as e:
            st.error(f"Failed to process audio file: {e}")

elif menu == "Retrain Model":
    st.title("🔁 Retrain SVM Classifier")
    st.write("Retrain using full dataset with SMOTE and selected features")

    sm = SMOTE()
    X = df.drop("label", axis=1)
    y = df["label"]
    X_res, y_res = sm.fit_resample(X, y)

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    selector = SelectKBest(score_func=f_classif, k=15)
    X_res_selected = selector.fit_transform(X_res_scaled, y_res)

    X_train, X_test, y_train, y_test = train_test_split(X_res_selected, y_res, test_size=0.2, stratify=y_res, random_state=42)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.success(f"Retrained Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    st.success("Model and Scaler saved successfully.")
