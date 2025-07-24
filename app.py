# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import librosa
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Set Streamlit config
st.set_page_config(page_title="Voice Gender Classification", layout="wide")

@st.cache_data

def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Load models
model = joblib.load("voice_gender_classifier_all_features.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.write("""
        Classify and cluster human voice samples by gender using ML.
    """)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Gender Classification using SVM")

    top_n = st.sidebar.slider("Select Top N Features", 5, df.shape[1]-1, 10)
    method = st.sidebar.selectbox("Feature Selection", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C_val = st.sidebar.slider("C", 0.1, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_grid = st.sidebar.checkbox("Use Grid Search")

    X = df.drop(columns=["label"])
    y = df["label"]

    # Balance classes
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    X_scaled = scaler.fit_transform(X)

    if method == "ANOVA F-test":
        selector = SelectKBest(score_func=f_classif, k=top_n)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=top_n)

    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()].tolist()

    st.subheader("🔍 Selected Features")
    st.write(selected_features)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, stratify=y, test_size=0.2, random_state=42)

    if use_grid:
        grid = GridSearchCV(SVC(class_weight="balanced"), {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }, cv=3)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = SVC(kernel=kernel, C=C_val, gamma=gamma, class_weight="balanced")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.success(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.code(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.subheader("🎛️ Predict on Custom Input")
    input_data = []
    for feat in selected_features:
        val = st.number_input(f"{feat}", value=float(df[feat].mean()))
        input_data.append(val)

    if st.button("Predict Gender"):
        try:
            input_scaled = scaler.transform([input_data])
            input_selected = selector.transform(input_scaled)
            pred = model.predict(input_selected)[0]
            label = "👩 Female" if pred == 0 else "👨 Male"
            st.success(f"Predicted Gender: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.subheader("📁 Predict from CSV")
    csv_file = st.file_uploader("Upload CSV with Features", type=["csv"])
    if csv_file:
        try:
            df_csv = pd.read_csv(csv_file)
            df_csv_scaled = scaler.transform(df_csv)
            df_selected = selector.transform(df_csv_scaled)
            preds = model.predict(df_selected)
            pred_labels = ["Female" if p == 0 else "Male" for p in preds]
            df_csv["Predicted Gender"] = pred_labels
            st.dataframe(df_csv)
        except Exception as e:
            st.error(f"CSV Prediction failed: {e}")

    st.subheader("🎧 Predict from Audio File")
    audio_file = st.file_uploader("Upload WAV File", type=["wav"])
    if audio_file:
        try:
            y, sr = librosa.load(audio_file, sr=None)
            features = [
                librosa.feature.zero_crossing_rate(y).mean(),
                librosa.feature.rms(y=y).mean(),
                librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
                librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
                librosa.feature.spectral_contrast(y=y, sr=sr).mean(),
                librosa.feature.spectral_flatness(y=y).mean(),
                librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
                librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).mean(),
                np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist()
            ]
            flat = []
            for f in features:
                flat.extend(f if isinstance(f, list) else [f])
            flat = flat[:len(X.columns)]  # truncate or pad
            flat += [0]*(len(X.columns)-len(flat))
            audio_scaled = scaler.transform([flat])
            audio_selected = selector.transform(audio_scaled)
            pred = model.predict(audio_selected)[0]
            st.success(f"Audio Gender: {'👩 Female' if pred == 0 else '👨 Male'}")
        except Exception as e:
            st.error(f"Failed to process audio: {e}")

elif menu == "Clustering":
    st.title("🔍 Voice Clustering")
    X = df.drop(columns=["label"])
    y = df["label"]
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    clusterers = {
        "KMeans": KMeans(n_clusters=2),
        "DBSCAN": DBSCAN(eps=1.5),
        "Agglomerative": AgglomerativeClustering(n_clusters=2),
        "GMM": GaussianMixture(n_components=2)
    }

    for name, algo in clusterers.items():
        if name == "GMM":
            labels = algo.fit_predict(X_scaled)
        else:
            labels = algo.fit_predict(X_scaled)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm")
        ax[0].set_title(f"{name} Clusters")
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm")
        ax[1].set_title("Actual Labels")
        st.pyplot(fig)
