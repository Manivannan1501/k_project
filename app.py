# Make sure all dependencies are in place.
# This updated app.py includes:
# - SMOTE integration
# - Audio file support
# - Feature selection & hyperparameter tuning
# - Clustering analysis and visualizations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import joblib
import tempfile

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, silhouette_score

from imblearn.over_sampling import SMOTE

# Set Streamlit config
st.set_page_config(page_title="Voice Gender App", layout="wide")

@st.cache_data

def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Load pre-trained model and scaler
model = joblib.load("voice_gender_classifier_all_features.pkl")
scaler = joblib.load("scaler.pkl")

# Audio feature extractor

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = []
        features.append(np.mean(librosa.pitch_tuning(librosa.yin(y, fmin=50, fmax=300, sr=sr))))
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(np.mean(librosa.feature.rms(y=y)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_flatness(y=y)))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for mfcc in mfccs:
            features.append(np.mean(mfcc))
        return pd.DataFrame([features])
    except Exception as e:
        raise RuntimeError(f"Failed to extract features: {e}")

# Navigation
menu = st.sidebar.radio("Menu", ["Introduction", "EDA", "Classification", "Clustering"])

if menu == "Introduction":
    st.title("🎤 Human Voice Gender Recognition")
    st.write("This app classifies speaker gender and analyzes voice data using ML and clustering algorithms.")

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df["gender"] = df["label"].map({0: "Female", 1: "Male"})

    st.subheader("Gender Distribution")
    st.bar_chart(df["gender"].value_counts())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Gender Classification with SVM")

    # Feature selection and SVM params
    top_n = st.sidebar.slider("Number of Features", 5, df.shape[1]-1, 10)
    selector_type = st.sidebar.selectbox("Selector", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    C_val = st.sidebar.slider("C", 0.1, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_grid = st.sidebar.checkbox("Use GridSearchCV")

    # Prepare data
    X = df.drop(columns=["label"])
    y = df["label"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    selector = SelectKBest(score_func=f_classif if selector_type == "ANOVA F-test" else mutual_info_classif, k=top_n)
    X_selected = selector.fit_transform(X_scaled, y_resampled)
    selected_features = X.columns[selector.get_support()]

    st.subheader("Top Features")
    feat_scores = pd.DataFrame({"Feature": selected_features, "Score": selector.scores_[selector.get_support()]})
    st.dataframe(feat_scores.sort_values("Score", ascending=False))

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, stratify=y_resampled)

    if use_grid:
        grid = GridSearchCV(SVC(class_weight="balanced"), {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }, cv=3)
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
    else:
        clf = SVC(kernel=kernel, C=C_val, gamma=gamma, class_weight="balanced")
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=["Female", "Male"], ax=ax)
    st.pyplot(fig)

    st.subheader("🎧 Predict from Audio File")
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            features = extract_features(tmp_path)
            features.columns = X.columns[:features.shape[1]]
            features_scaled = scaler.transform(features)
            features_selected = selector.transform(features_scaled)
            prediction = clf.predict(features_selected)
            st.success(f"Predicted Gender: {'👩 Female' if prediction[0] == 0 else '👨 Male'}")
        except Exception as e:
            st.error(f"Failed to process audio file: {e}")

elif menu == "Clustering":
    st.title("🔍 Clustering Analysis")
    X = df.drop(columns=["label"])
    y = df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    models = {
        "KMeans": KMeans(n_clusters=2, random_state=42),
        "DBSCAN": DBSCAN(eps=2, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=2),
        "GMM": GaussianMixture(n_components=2, random_state=42)
    }

    results = {}
    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        results[name] = (labels, score)

    result_df = pd.DataFrame({"Model": list(results.keys()), "Silhouette Score": [v[1] for v in results.values()]})
    st.subheader("📈 Clustering Scores")
    st.dataframe(result_df.sort_values("Silhouette Score", ascending=False))

    for name, (labels, _) in results.items():
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
        ax[0].set_title(f"{name} Clustering")
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        ax[1].set_title("True Labels")
        st.pyplot(fig)
