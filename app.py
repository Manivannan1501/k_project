import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import joblib
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# ===================== AUDIO FEATURE EXTRACTION =====================
def extract_features_from_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = []

        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(np.mean(librosa.feature.rms(y=y)))
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_flatness(y=y)))
        features.append(np.mean(librosa.feature.rolloff(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)))
        features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features.append(np.mean(mfccs[i]))

        return np.array(features)
    except Exception as e:
        st.error(f"Failed to extract features: {e}")
        return None

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

# ===================== MAIN =====================
st.set_page_config(page_title="Voice Gender Classifier", layout="wide")
df = load_data()

st.title("🎙️ Human Voice Gender Classification and Clustering")

menu = st.sidebar.radio("Navigate", ["EDA", "Classification", "Upload Audio", "Clustering"])

# ===================== EDA =====================
if menu == "EDA":
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Gender Distribution")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ===================== CLASSIFICATION =====================
elif menu == "Classification":
    X = df.drop(columns=["label"])
    y = df["label"]

    st.sidebar.subheader("Feature Selection and Tuning")
    k_features = st.sidebar.slider("Number of Features", 5, X.shape[1], 10)
    method = st.sidebar.selectbox("Selection Method", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_smote = st.sidebar.checkbox("Use SMOTE for class balancing")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "ANOVA F-test":
        selector = SelectKBest(score_func=f_classif, k=k_features)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=k_features)

    X_selected = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("Model Evaluation")
    st.text(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
    st.pyplot(fig)

    # Save model and scaler for reuse
    joblib.dump(clf, "final_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selector, "selector.pkl")

# ===================== UPLOAD AUDIO =====================
elif menu == "Upload Audio":
    st.subheader("🎧 Upload Audio File for Gender Prediction")
    uploaded_file = st.file_uploader("Upload a WAV file", type=[".wav"])

    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        features = extract_features_from_audio("temp.wav")

        if features is not None:
            try:
                clf = joblib.load("final_model.pkl")
                scaler = joblib.load("scaler.pkl")
                selector = joblib.load("selector.pkl")

                features_scaled = scaler.transform([features])
                features_selected = selector.transform(features_scaled)
                pred = clf.predict(features_selected)[0]
                label = "👨 Male" if pred == 1 else "👩 Female"
                st.success(f"Predicted Gender: {label}")
            except Exception as e:
                st.error(f"Model failed: {e}")

# ===================== CLUSTERING =====================
elif menu == "Clustering":
    st.subheader("🔍 Clustering of Voice Samples")
    features = df.drop(columns=["label"])
    true_labels = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_models = {
        "KMeans": KMeans(n_clusters=2, random_state=42),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "GMM": GaussianMixture(n_components=2, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=2)
    }

    for name, model in cluster_models.items():
        if name == "DBSCAN":
            labels = model.fit_predict(X_scaled)
            mask = labels != -1
            score = silhouette_score(X_scaled[mask], labels[mask]) if np.any(mask) else -1
        else:
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

        st.write(f"**{name} Silhouette Score:** {score:.3f}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
        ax[0].set_title(f"{name} Clustering")
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap="coolwarm", alpha=0.7)
        ax[1].set_title("True Gender Labels")
        st.pyplot(fig)
