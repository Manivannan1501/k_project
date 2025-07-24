import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import librosa
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

st.set_page_config(page_title="Human Voice Classification & Clustering", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

with open("voice_gender_classifier_all_features.pkl", 'rb') as f:
    model = joblib.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = joblib.load(f)

menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])

if menu == "Introduction":
    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("---")
    st.header("🧩 Introduction")
    st.write("""
    Analyze human voice characteristics to classify gender and group voice patterns using clustering.
    """)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("1. Gender Distribution")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)

    st.subheader("2. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("3. Feature Distributions")
    df['gender'] = df['label'].map({0: 'Female', 1: 'Male'})
    features = ['mean_pitch', 'zero_crossing_rate', 'rms_energy', 'log_energy', 'mfcc_1_mean']
    for feature in features:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=feature, hue='gender', fill=True, ax=ax)
        st.pyplot(fig)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification using SVM")

    st.sidebar.subheader("⚙️ Feature Selection & Tuning")
    top_n = st.sidebar.slider("Select Top N Features", 5, len(df.columns)-1, 10)
    kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C_value = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    gamma_value = st.sidebar.selectbox("Gamma", ["scale", "auto"])

    X = df.drop(columns=["label"])
    y = df["label"]

    # Balance classes (Optional but helps when Male/Female are imbalanced)
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=top_n)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    model = SVC(kernel=kernel, C=C_value, gamma=gamma_value, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Female", "Male"], ax=ax, cmap="Blues")
    st.pyplot(fig)

    st.subheader("🎛️ Predict with Custom Input")
    full_input = []
    for col in X.columns:
        if col in selected_features:
            val = st.slider(
                label=col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean()),
                step=0.01
            )
            full_input.append(val)
        else:
            full_input.append(float(df[col].mean()))

    if st.button("🔍 Predict Gender"):
        try:
            input_scaled = scaler.transform([full_input])
            input_selected = selector.transform(input_scaled)
            prediction = model.predict(input_selected)
            label = "👨 Male" if prediction[0] == 1 else "👩 Female"
            st.success(f"Predicted Gender: **{label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif menu == "Clustering":
    st.title("🔍 Voice Clustering Analysis")
    features = df.drop(columns=["label"])
    true_labels = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_outputs = {}

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    cluster_outputs["KMeans"] = {"labels": kmeans_labels, "score": silhouette_score(X_scaled, kmeans_labels)}

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    db_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if np.any(dbscan_labels != -1) else -1
    cluster_outputs["DBSCAN"] = {"labels": dbscan_labels, "score": db_score}

    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(X_scaled)
    cluster_outputs["Agglomerative"] = {"labels": agg_labels, "score": silhouette_score(X_scaled, agg_labels)}

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    cluster_outputs["GMM"] = {"labels": gmm_labels, "score": silhouette_score(X_scaled, gmm_labels)}

    st.subheader("📊 Clustering Scores")
    st.dataframe(pd.DataFrame({"Model": list(cluster_outputs.keys()), "Score": [v["score"] for v in cluster_outputs.values()]}))

    st.subheader("📈 PCA Visualization")
    for name, result in cluster_outputs.items():
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=result["labels"], cmap="coolwarm", alpha=0.7)
        ax[0].set_title(f"{name} Clustering")
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap="coolwarm", alpha=0.7)
        ax[1].set_title("True Labels")
        st.pyplot(fig)
