import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os
import joblib
import tempfile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except ImportError:
    smote_available = False

st.set_page_config(page_title="Human Voice Classification & Clustering", layout="wide")

@st.cache_data

def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

menu = st.sidebar.radio("Navigation", ["Introduction", "EDA", "Classification", "Clustering", "Audio Upload & Prediction", "Model Retraining"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("---")
    st.write("""
    This project explores voice-based gender classification and clustering using machine learning.
    Features:
    - Audio upload and prediction
    - Feature selection and tuning
    - Clustering analysis
    - Model retraining with CSV
    """)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df['gender'] = df['label'].map({0: 'Female', 1: 'Male'})
    st.subheader("Label Distribution")
    st.bar_chart(df['gender'].value_counts())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification")

    st.sidebar.subheader("Feature Selection & SVM Tuning")
    top_k = st.sidebar.slider("Select top K features", 5, len(df.columns)-1, 10)
    method = st.sidebar.selectbox("Feature Selection Method", ["ANOVA", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_grid = st.sidebar.checkbox("Use GridSearchCV")

    X = df.drop(columns=["label"])
    y = df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif if method=="ANOVA" else mutual_info_classif, k=top_k)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]

    if smote_available:
        smote = SMOTE()
        X_selected, y = smote.fit_resample(X_selected, y)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

    if use_grid:
        grid = GridSearchCV(SVC(class_weight="balanced"), {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }, cv=3)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        st.success(f"Best Parameters: {grid.best_params_}")
    else:
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Test Accuracy: {acc:.2%}")
    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Female", "Male"], ax=ax, cmap="Blues")
    st.pyplot(fig)

    st.subheader("🎛 Predict on Custom Input")
    user_input = []
    for col in X.columns:
        val = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        user_input.append(val)

    if st.button("🔍 Predict Gender"):
        try:
            input_scaled = scaler.transform([user_input])
            input_selected = selector.transform(input_scaled)
            pred = model.predict(input_selected)[0]
            gender = "👩 Female" if pred == 0 else "👨 Male"
            st.success(f"Predicted Gender: {gender}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif menu == "Audio Upload & Prediction":
    st.title("🔊 Upload Audio File for Prediction")
    audio = st.file_uploader("Upload .wav File", type=["wav"])

    def extract_audio_features(file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            features = [
                np.mean(librosa.feature.zero_crossing_rate(y).T),
                np.mean(librosa.feature.rms(y=y).T),
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T),
                np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T),
                np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T),
                np.mean(librosa.feature.spectral_flatness(y=y).T),
                np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0).tolist()
            ]
            flat_features = [item for sublist in features if isinstance(sublist, list) for item in sublist]
            return np.array(flat_features).flatten()
        except Exception as e:
            st.error(f"Failed to extract features: {e}")
            return None

    if audio:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio.read())
            tmp_path = tmp.name

        features = extract_audio_features(tmp_path)
        if features is not None:
            try:
                input_scaled = scaler.transform([features])
                input_selected = selector.transform(input_scaled)
                pred = model.predict(input_selected)[0]
                label = "👩 Female" if pred == 0 else "👨 Male"
                st.success(f"Prediction: {label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif menu == "Clustering":
    st.title("🔍 Clustering Voice Data")
    X = df.drop(columns=["label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2).fit_transform(X_scaled)
    models = {
        "KMeans": KMeans(n_clusters=2, random_state=0),
        "Agglomerative": AgglomerativeClustering(n_clusters=2),
        "GMM": GaussianMixture(n_components=2),
        "DBSCAN": DBSCAN(eps=2.5)
    }

    for name, model_c in models.items():
        labels = model_c.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        st.subheader(f"{name} Clustering (Score: {score:.2f})")
        fig, ax = plt.subplots()
        ax.scatter(pca[:, 0], pca[:, 1], c=labels, cmap="coolwarm")
        st.pyplot(fig)

elif menu == "Model Retraining":
    st.title("🔁 Retrain Model with Custom CSV")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df_new = pd.read_csv(uploaded)
            st.write("Data Preview:", df_new.head())
            df_new.dropna(inplace=True)
            X = df_new.drop(columns=["label"])
            y = df_new["label"]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = SVC(kernel="rbf", class_weight="balanced")
            model.fit(X_scaled, y)
            joblib.dump(model, "retrained_model.pkl")
            st.success("Model retrained and saved as retrained_model.pkl")
        except Exception as e:
            st.error(f"Failed to retrain: {e}")
