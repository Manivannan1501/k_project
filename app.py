import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

    top_10 = [
        'mfcc_1_mean', 'mean_pitch', 'mfcc_3_mean', 'mfcc_5_mean', 
        'zero_crossing_rate', 'rms_energy', 'mean_spectral_centroid',
        'std_pitch', 'mfcc_2_mean', 'log_energy'
    ]

    st.subheader("🎛️ Enter Feature Values Manually")
    input_data = []
    for col in top_10:
        val = st.slider(
            label=col,
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean()),
            step=0.01
        )
        input_data.append(val)

    if st.button("Predict from Manual Input"):
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        label = "👨 Male" if prediction[0] == 1 else "👩 Female"
        st.success(f"Predicted Gender: **{label}**")

    st.subheader("📂 Upload CSV to Predict Gender")
    uploaded_file = st.file_uploader("Upload a CSV file with required features", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing = [col for col in top_10 if col not in uploaded_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                input_df = uploaded_df[top_10]
                input_scaled = scaler.transform(input_df)
                predictions = model.predict(input_scaled)
                labels = ['👩 Female' if p == 0 else '👨 Male' for p in predictions]
                uploaded_df['Predicted Gender'] = labels
                st.success("✅ Prediction completed!")
                st.dataframe(uploaded_df[['Predicted Gender'] + top_10])
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif menu == "Clustering":
    st.title("🔍 Voice Clustering Analysis")
    X = df.drop(columns=["label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_outputs = {}
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_outputs["KMeans"] = {
        "labels": kmeans.fit_predict(X_scaled),
        "score": silhouette_score(X_scaled, kmeans.labels_)
    }

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if np.any(dbscan_labels != -1) else -1
    cluster_outputs["DBSCAN"] = {"labels": dbscan_labels, "score": dbscan_score}

    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(X_scaled)
    cluster_outputs["Agglomerative"] = {
        "labels": agg_labels,
        "score": silhouette_score(X_scaled, agg_labels)
    }

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    cluster_outputs["GMM"] = {
        "labels": gmm_labels,
        "score": silhouette_score(X_scaled, gmm_labels)
    }

    st.subheader("📊 Clustering Model Silhouette Scores")
    score_df = pd.DataFrame({
        "Model": list(cluster_outputs.keys()),
        "Silhouette Score": [v["score"] for v in cluster_outputs.values()]
    }).sort_values(by="Silhouette Score", ascending=False)
    st.dataframe(score_df)

    st.subheader("📈 PCA Visualizations")
    for name, result in cluster_outputs.items():
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=result["labels"], cmap="viridis")
        ax.set_title(f"{name} Clustering")
        st.pyplot(fig)
