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

@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Sidebar menu
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])

if menu == "Introduction":
    st.title("🎹 Human Voice Classification and Clustering")
    st.markdown("---")
    st.header("🧩 Introduction")
    st.write("""
    Explore how human voice characteristics can be analyzed using machine learning to classify gender and group similar voice patterns using clustering techniques.
    """)
    st.header("❗ Problem Statement")
    st.write("""
    There is a need for a lightweight, feature-based ML system that:
    - Classifies a voice sample's gender.
    - Clusters unlabeled voices into meaningful groups.
    - Accepts manual input, audio upload, and CSV batch input.
    """)
    st.header("💡 Technologies Used")
    st.table({
        "Component": ["Language", "Data Analysis", "Visualization", "ML", "Interface", "Deployment"],
        "Technology": ["Python", "pandas, NumPy", "Matplotlib, Seaborn", "scikit-learn", "Streamlit", "Pickle"]
    })

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification")

    st.sidebar.header("⚙️ Model Tuning")
    top_n = st.sidebar.slider("Select Top N Features", 5, len(df.columns)-1, 10)
    selector_method = st.sidebar.selectbox("Feature Selection Method", ["ANOVA F-test", "Mutual Info"])
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C_value = st.sidebar.slider("Regularization C", 0.01, 10.0, 1.0)
    gamma_value = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    use_grid = st.sidebar.checkbox("Use Grid Search")

    X = df.drop(columns=['label'])
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif if selector_method == "ANOVA F-test" else mutual_info_classif, k=top_n)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()].tolist()

    st.subheader("⭐ Top Selected Features")
    st.dataframe(pd.DataFrame({"Feature": selected_features}))

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y)

    if use_grid:
        grid = GridSearchCV(SVC(class_weight='balanced'), {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }, cv=3)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        st.success(f"Best Params: {grid.best_params_}")
    else:
        model = SVC(kernel=kernel, C=C_value, gamma=gamma_value, class_weight='balanced')
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    st.subheader("Predict Custom Input")
    full_input = [
        st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()), step=0.01)
        if col in selected_features else float(df[col].mean()) for col in X.columns
    ]
    if st.button("Predict Gender"):
        input_scaled = scaler.transform([full_input])
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)[0]
        st.success("Predicted: Male" if prediction == 1 else "Predicted: Female")

    st.subheader("📂 Batch Prediction from CSV")
    csv_file = st.file_uploader("Upload CSV with same columns as training features", type=["csv"])
    if csv_file:
        try:
            input_df = pd.read_csv(csv_file)
            input_scaled = scaler.transform(input_df)
            input_selected = selector.transform(input_scaled)
            preds = model.predict(input_selected)
            input_df['Predicted Gender'] = ['Male' if p == 1 else 'Female' for p in preds]
            st.dataframe(input_df)
        except Exception as e:
            st.error(f"Failed to predict: {e}")

    st.subheader("🎵 Predict from Audio File")
    audio_file = st.file_uploader("Upload WAV file", type=["wav"])
    if audio_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            y_audio, sr = librosa.load(tmp_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y_audio))
            energy = np.mean(librosa.feature.rms(y_audio))
            log_energy = np.log1p(np.sum(y_audio ** 2))
            pitch = librosa.yin(y_audio, fmin=50, fmax=500)
            mean_pitch = np.mean(pitch)
            std_pitch = np.std(pitch)
            features_audio = [
                mean_pitch, zcr, energy, log_energy,
                *[np.mean(mfccs[i]) for i in range(13)],
                std_pitch
            ]
            input_audio_df = pd.DataFrame([features_audio], columns=[
                "mean_pitch", "zero_crossing_rate", "rms_energy", "log_energy",
                *[f"mfcc_{i+1}_mean" for i in range(13)], "std_pitch"
            ])
            input_audio_scaled = scaler.transform(input_audio_df)
            input_audio_selected = selector.transform(input_audio_scaled)
            prediction = model.predict(input_audio_selected)[0]
            st.success("Predicted Gender: Male" if prediction == 1 else "Predicted Gender: Female")
        except Exception as e:
            st.error(f"Failed to process audio file: {e}")

elif menu == "Clustering":
    st.title("Clustering Analysis")
    features = df.drop(columns=['label'])
    true_labels = df['label']
    X_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    models = {
        "KMeans": KMeans(n_clusters=2),
        "DBSCAN": DBSCAN(eps=1.5),
        "Agglomerative": AgglomerativeClustering(n_clusters=2),
        "GMM": GaussianMixture(n_components=2)
    }
    for name, clusterer in models.items():
        try:
            labels = clusterer.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
            st.subheader(f"{name} Clustering (Score: {score:.2f})")
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm')
            ax[0].set_title("Cluster Output")
            ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='coolwarm')
            ax[1].set_title("Actual Gender")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"{name} failed: {e}")
