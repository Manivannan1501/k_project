# Final complete version of app.py with:
# - SMOTE balancing
# - Audio upload and prediction
# - Classification with feature selection and tuning
# - Clustering analysis with visualizations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib
import os
import tempfile

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Voice Gender Classification & Clustering", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Sidebar menu
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering", "Audio Upload"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("---")
    st.header("🧩 Introduction")
    st.write("""
    This app explores human voice characteristics for **gender classification** and **clustering** using ML.
    """)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("1. Gender Distribution")
    df['gender'] = df['label'].map({0: 'Female', 1: 'Male'})
    st.bar_chart(df['gender'].value_counts())

    st.subheader("2. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("3. Feature Distribution")
    features = ['mean_pitch', 'zero_crossing_rate', 'rms_energy', 'mfcc_1_mean']
    for f in features:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=f, hue='gender', fill=True, ax=ax)
        st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification")

    X = df.drop(columns=['label'])
    y = df['label']
    st.sidebar.subheader("⚙️ Feature Selection & Tuning")
    top_n = st.sidebar.slider("Select Top N Features", 5, X.shape[1], 10)
    method = st.sidebar.selectbox("Feature Selection", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_grid = st.sidebar.checkbox("Grid Search", False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

    if method == "ANOVA F-test":
        selector = SelectKBest(score_func=f_classif, k=top_n)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=top_n)

    X_selected = selector.fit_transform(X_resampled, y_resampled)
    selected_features = X.columns[selector.get_support()]

    st.subheader("⭐ Top Selected Features")
    st.write(selected_features.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, stratify=y_resampled)

    if use_grid:
        params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["linear", "rbf"]}
        model = GridSearchCV(SVC(class_weight="balanced"), params, cv=3)
    else:
        model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Female", "Male"], ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.subheader("🎛️ Predict Gender with Input")
    input_vals = []
    for col in X.columns:
        if col in selected_features.tolist():
            val = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_vals.append(val)
        else:
            input_vals.append(float(df[col].mean()))

    if st.button("Predict Gender"):
        try:
            scaled_input = scaler.transform([input_vals])
            input_selected = selector.transform(scaled_input)
            pred = model.predict(input_selected)
            label = "👨 Male" if pred[0] == 1 else "👩 Female"
            st.success(f"Predicted Gender: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif menu == "Clustering":
    st.title("🔍 Voice Clustering")
    X = df.drop(columns=['label'])
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_results = {}

    models = {
        "KMeans": KMeans(n_clusters=2, random_state=42),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=2),
        "GMM": GaussianMixture(n_components=2, random_state=42)
    }

    for name, algo in models.items():
        try:
            labels = algo.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
            cluster_results[name] = {"labels": labels, "score": score}
        except Exception:
            cluster_results[name] = {"labels": np.zeros(len(X)), "score": -1}

    st.subheader("📈 Clustering Results")
    for name, result in cluster_results.items():
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=result["labels"], cmap="coolwarm")
        ax[0].set_title(f"{name} Clustering")
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm")
        ax[1].set_title("True Labels")
        st.pyplot(fig)
        st.info(f"{name} Silhouette Score: {result['score']:.3f}")

elif menu == "Audio Upload":
    st.title("🎤 Upload Audio File")

    def extract_features(file_path):
        try:
            y, sr = librosa.load(file_path)
            features = {
                "mean_pitch": np.mean(librosa.yin(y, fmin=50, fmax=300)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "rms_energy": np.mean(librosa.feature.rms(y=y)),
                "log_energy": np.log1p(np.mean(librosa.feature.rms(y=y))),
                "mfcc_1_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr)[0])
            }
            return features
        except Exception as e:
            st.error(f"Failed to extract features: {e}")
            return None

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path)
        feats = extract_features(tmp_path)

        if feats:
            input_vector = []
            for col in df.drop(columns=['label']).columns:
                if col in feats:
                    input_vector.append(feats[col])
                else:
                    input_vector.append(df[col].mean())

            scaled_input = StandardScaler().fit_transform(df.drop(columns=['label']))
            scaler = StandardScaler().fit(scaled_input)
            try:
                sm = SMOTE(random_state=42)
                X_resampled, y_resampled = sm.fit_resample(scaled_input, df['label'])
                selector = SelectKBest(score_func=f_classif, k=10).fit(X_resampled, y_resampled)
                model = SVC(kernel="rbf", class_weight="balanced").fit(selector.transform(X_resampled), y_resampled)

                input_scaled = scaler.transform([input_vector])
                input_selected = selector.transform(input_scaled)
                pred = model.predict(input_selected)
                st.success("Predicted Gender: 👨 Male" if pred[0] == 1 else "Predicted Gender: 👩 Female")
            except Exception as e:
                st.error(f"Failed to predict: {e}")
