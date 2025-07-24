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

# Set page config
st.set_page_config(page_title="Human Voice Classification & Clustering", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Load model and scaler
with open("voice_gender_classifier_all_features.pkl", 'rb') as f:
    model = joblib.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = joblib.load(f)

# Sidebar menu
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("---")
    st.header("🧩 Introduction")
    st.write("""
    This project aims to explore how human voice characteristics can be analyzed using machine learning to **classify gender** and **group similar voice patterns** using **clustering techniques**.
    """)
    st.header("❗ Problem Statement")
    st.write("""
    There is a need for a **lightweight, feature-based ML system** that:
    - Classifies a voice sample's gender.
    - Clusters unlabeled voices into meaningful groups.
    - Is easy to deploy in web apps.
    """)
    st.header("💡 Proposed Solution")
    st.write("""
    1. **Classification** using SVM
    2. **Clustering** using K-Means, DBSCAN, etc.
    """)
    st.header("🛠️ Technologies Used")
    st.table({
        "Component": ["Language", "Data Analysis", "Visualization", "ML", "Interface", "Deployment"],
        "Technology": ["Python", "pandas, NumPy", "Matplotlib, Seaborn", "scikit-learn", "Streamlit", "Pickle"]
    })

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("1. Gender Class Distribution")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)

    st.subheader("2. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("3. Feature Distributions by Gender")
    df['gender'] = df['label'].map({0: 'Female', 1: 'Male'})
    features = ['mean_pitch', 'zero_crossing_rate', 'rms_energy', 'log_energy', 'mfcc_1_mean']
    for feature in features:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=feature, hue='gender', fill=True, ax=ax)
        st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification using SVM")

    st.sidebar.subheader("⚙️ Feature Selection & SVM Tuning")
    top_n = st.sidebar.slider("Number of Features to Select", 5, len(df.columns)-1, 10)
    selector_method = st.sidebar.selectbox("Feature Selection Method", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C_value = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    gamma_value = st.sidebar.selectbox("Gamma", ["scale", "auto"])

    use_grid = st.sidebar.checkbox("Use Grid Search (Slow)", value=False)

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

    st.subheader("⭐ Top Selected Features")
    feature_scores = selector.scores_[selector.get_support()]
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Score": feature_scores
    }).sort_values(by="Score", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x="Score", y="Feature", ax=ax)
    st.pyplot(fig)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    if use_grid:
        param_grid = {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }
        grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, cv=3)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        st.success(f"Best Parameters: {grid.best_params_}")
    else:
        best_model = SVC(kernel=kernel, C=C_value, gamma=gamma_value, class_weight="balanced")
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Accuracy on Test Set: **{acc:.2%}**")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Female", "Male"], ax=ax, cmap="Blues")
    st.pyplot(fig)

    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.subheader("🎛️ Upload Audio File for Prediction")
    audio_file = st.file_uploader("Upload WAV file", type=["wav"])
    if audio_file is not None:
        try:
            y, sr = librosa.load(audio_file, sr=None)
            features = [
                np.mean(librosa.feature.zero_crossing_rate(y)),
                np.mean(librosa.feature.rms(y=y)),
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                np.mean(librosa.feature.spectral_flatness(y=y)),
                np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist()
            ]
            features_flat = [val for sublist in features for val in (sublist if isinstance(sublist, list) else [sublist])]
            input_audio = scaler.transform([features_flat + [0] * (X.shape[1] - len(features_flat))])
            input_selected = selector.transform(input_audio)
            pred = best_model.predict(input_selected)[0]
            gender = "Male 👨" if pred == 1 else "Female 👩"
            st.success(f"Predicted Gender: **{gender}**")
        except Exception as e:
            st.error(f"Failed to process audio file: {e}")

elif menu == "Clustering":
    st.title("🔍 Voice Clustering Analysis")

    # Separate features and labels
    features = df.drop(columns=["label"])
    true_labels = df["label"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Dictionary to store clustering results
    cluster_outputs = {}

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    cluster_outputs["KMeans"] = {"labels": kmeans_labels, "score": kmeans_score}

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if np.any(dbscan_labels != -1) else -1
    cluster_outputs["DBSCAN"] = {"labels": dbscan_labels, "score": dbscan_score}

    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(X_scaled)
    agg_score = silhouette_score(X_scaled, agg_labels)
    cluster_outputs["Agglomerative"] = {"labels": agg_labels, "score": agg_score}

    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_score = silhouette_score(X_scaled, gmm_labels)
    cluster_outputs["GMM"] = {"labels": gmm_labels, "score": gmm_score}

    st.subheader("📊 Clustering Silhouette Scores")
    score_df = pd.DataFrame({
        "Model": list(cluster_outputs.keys()),
        "Silhouette Score": [v["score"] for v in cluster_outputs.values()]
    }).sort_values(by="Silhouette Score", ascending=False)
    st.dataframe(score_df)

    st.subheader("📈 PCA Clustering Visualizations (Compared to True Gender)")

    for name, result in cluster_outputs.items():
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Clustering result
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=result["labels"], cmap="coolwarm", alpha=0.7)
        ax[0].set_title(f"{name} Clustering")

        # True label visualization
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap="coolwarm", alpha=0.7)
        ax[1].set_title("Actual Gender Labels")

        st.pyplot(fig)
