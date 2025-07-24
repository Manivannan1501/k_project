# Human Voice Gender Classification & Clustering App with Audio Support, SMOTE, Hyperparameter Tuning, Feature Selection, and Visualizations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib
import tempfile
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

# Set page layout
st.set_page_config(page_title="Human Voice App", layout="wide")

# Load CSV Dataset
@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Sidebar Navigation
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])

# Utility: Feature Extraction from Audio
@st.cache_data(experimental_allow_widgets=True)
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        features = {}

        features['mean_pitch'] = np.mean(librosa.yin(y, fmin=50, fmax=300, sr=sr))
        features['std_pitch'] = np.std(librosa.yin(y, fmin=50, fmax=300, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['rms_energy'] = np.mean(librosa.feature.rms(y=y)[0])
        features['log_energy'] = np.log1p(features['rms_energy'])
        features['mean_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(1, 6):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])

        return pd.DataFrame([features])
    except Exception as e:
        st.warning(f"Failed to extract features: {e}")
        return None

# Introduction Page
if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("""
    This app classifies human voice gender and clusters voice samples using audio features.
    
    Features:
    - Upload CSV or audio to predict gender
    - Experiment with feature selection and hyperparameter tuning
    - Visualize clusters using PCA and export results
    """)

# EDA Page
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    gender_counts = df['label'].value_counts().rename({0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Classification Page
elif menu == "Classification":
    st.title("🤖 Voice Gender Classification")

    X = df.drop("label", axis=1)
    y = df["label"]

    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    st.sidebar.subheader("🔧 Feature Selection & Tuning")
    k = st.sidebar.slider("Select Top K Features", 5, X.shape[1], 10)
    method = st.sidebar.selectbox("Selection Method", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    C_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    selector = SelectKBest(score_func=f_classif if method == "ANOVA F-test" else mutual_info_classif, k=k)
    X_sel = selector.fit_transform(X_scaled, y_bal)
    selected_features = X.columns[selector.get_support()]

    st.write("### Selected Features")
    st.write(selected_features.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X_sel, y_bal, test_size=0.2, stratify=y_bal)

    model = SVC(kernel=kernel, C=C_val, gamma=gamma, class_weight="balanced", probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred))

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Female", "Male"], ax=ax)
    st.pyplot(fig)

    # Predict with audio
    st.subheader("🎤 Predict with Audio File")
    audio = st.file_uploader("Upload WAV File", type=['wav'])
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio.read())
            tmp_path = tmp.name
        features = extract_features(tmp_path)
        if features is not None:
            input_scaled = scaler.transform(features[selected_features])
            prediction = model.predict(input_scaled)[0]
            st.success("Predicted Gender: **{}**".format("Male" if prediction == 1 else "Female"))

# Clustering Page
elif menu == "Clustering":
    st.title("🔍 Clustering Voice Samples")

    features = df.drop("label", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    models = {
        "KMeans": KMeans(n_clusters=2, random_state=42),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=2),
        "GMM": GaussianMixture(n_components=2, random_state=42)
    }

    st.subheader("📈 Clustering Visualizations")
    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        ax[0].set_title(f"{name} Clusters")

        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['label'], cmap='coolwarm')
        ax[1].set_title("Actual Gender")

        st.pyplot(fig)

        export = st.checkbox(f"Export {name} clustering results as CSV")
        if export:
            export_df = df.copy()
            export_df[f"{name}_cluster"] = labels
            csv = export_df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, f"{name}_clusters.csv")
