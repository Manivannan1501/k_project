import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(page_title="Human Voice Classification & Clustering", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

df = load_data()

# Sidebar menu
menu = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Classification", "Clustering"])
if menu == "Introduction":

    st.title("🎙️ Human Voice Classification and Clustering")
    st.markdown("---")

    st.header("🧩 Introduction")
    st.write("""
    This project aims to explore how human voice characteristics can be analyzed using machine learning to **classify gender** and **group similar voice patterns** using **clustering techniques**.

    By analyzing extracted audio features like MFCCs, pitch, energy, and spectral properties, the system can recognize patterns that distinguish male and female voices, and also cluster similar-sounding voices without labels.
    """)

    st.header("❗ Problem Statement")
    st.write("""
    Traditional voice classification systems often rely on direct audio analysis or deep signal processing, which may require complex setups or large raw datasets. These approaches may not generalize well or offer easy deployment in lightweight applications.

    There is a need for a **lightweight, feature-based machine learning system** that:
    - Classifies a voice sample's gender based on extracted audio features.
    - Clusters unlabeled voices into meaningful groups (e.g., by pitch, tone).
    - Is easy to deploy in web apps like Streamlit for user-friendly interaction.
    """)

    st.header("💡 Proposed Solution")
    st.markdown("""
    We propose a two-part ML system:

    1. **Classification**:
        - Predicts whether the speaker is male or female using **SVM** and other models based on selected features (e.g., pitch, MFCCs, energy).

    2. **Clustering**:
        - Uses **K-Means** and other models to group similar voice samples based on spectral and pitch-related features.

    Additional features include:
    - Feature importance analysis to select the best parameters.
    - Scaled and preprocessed data to improve model performance.
    - Streamlit-based interface for non-technical users.
    """)

    st.header("🛠️ Technologies and Languages Used")
    st.table({
        "Component": [
            "Programming Language", "Data Analysis", "Visualization",
            "Machine Learning", "Web App Interface", "Model Deployment"
        ],
        "Technology": [
            "Python", "pandas, NumPy", "Matplotlib, Seaborn",
            "scikit-learn", "Streamlit", "Pickle"
        ]
    })

    st.header("🤖 Machine Learning Models Used")
    st.markdown("### ✅ Classification Models")
    st.markdown("""
    - **Support Vector Machine (SVM)** — selected for final deployment  
    - Random Forest Classifier  
    - K-Nearest Neighbors (KNN)  
    - Gradient Boosting Classifier (e.g., XGBoost or GradientBoostingClassifier)  
    - Neural Networks (Optional extension)
    """)

    st.markdown("### ✅ Clustering Models")
    st.markdown("""
    - **K-Means Clustering**  
    - **DBSCAN**
    - **Agglomerative Clustering**
    - **Gaussian Mixture Model**
    """)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Explore the dataset: class distribution, feature relationships, and audio characteristics.")


    # 1. Class Distribution
    st.subheader("1. 🧑‍🤝‍🧑 Gender Class Distribution")
    gender_counts = df['label'].value_counts().rename(index={0: 'Female', 1: 'Male'})
    st.bar_chart(gender_counts)

    # 2. Correlation with Target
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # 3. KDE Distribution of Selected Features
    st.subheader("3. 📈 Distribution of Important Features by Gender")

    important_features = ['mean_pitch', 'zero_crossing_rate', 'rms_energy', 'log_energy', 'mfcc_1_mean']
    label_map = {0: 'Female', 1: 'Male'}
    df['gender'] = df['label'].map(label_map)

    for feature in important_features:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=feature, hue='gender', fill=True, ax=ax)
        ax.set_title(f"{feature} distribution by gender")
        st.pyplot(fig)

    # 4. Boxplots for Outlier Detection
    st.subheader("4. 🧪 Outliers in Key Features")

    for feature in important_features:
        fig, ax = plt.subplots()
        sns.boxplot(x='gender', y=feature, data=df, ax=ax)
        ax.set_title(f"{feature} by Gender")
        st.pyplot(fig)

    # 5. MFCC Feature Distributions
    st.subheader("5. 🎼 MFCC Feature Distributions")

    mfcc_cols = [col for col in df.columns if 'mfcc' in col and '_mean' in col]
    fig, ax = plt.subplots(figsize=(12, 8))
    df[mfcc_cols].hist(bins=20, figsize=(15, 10), layout=(5, 3), ax=ax if len(mfcc_cols) == 1 else None)
    plt.suptitle("MFCC Mean Feature Distributions", fontsize=16)
    st.pyplot(plt)

elif menu == "Classification":
    st.title("🤖 Voice Gender Classification")

    # Model and scaler
    with open("voice_gender_classifier_all_features.pkl", "rb") as f:
        model = joblib.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)

    input_df = pd.DataFrame([features])[[
    'mfcc_1_mean', 'mean_pitch', 'mfcc_3_mean', 'mfcc_5_mean',
    'zero_crossing_rate', 'rms_energy', 'mean_spectral_centroid',
    'std_pitch', 'mfcc_2_mean', 'log_energy'
]]

    import librosa

    def extract_features(audio_data, sr):
        features = {}

        try:
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            features['mean_pitch'] = np.mean(pitches) if len(pitches) > 0 else 0
            features['std_pitch'] = np.std(pitches) if len(pitches) > 0 else 0

            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=6)
            features['mfcc_1_mean'] = np.mean(mfcc[1])
            features['mfcc_2_mean'] = np.mean(mfcc[2])
            features['mfcc_3_mean'] = np.mean(mfcc[3])
            features['mfcc_5_mean'] = np.mean(mfcc[5])

            rms = librosa.feature.rms(y=audio_data)[0]
            features['rms_energy'] = np.mean(rms)
            features['log_energy'] = np.log(np.sum(audio_data**2) + 1e-6)

            zcr = librosa.feature.zero_crossing_rate(y=audio_data)[0]
            features['zero_crossing_rate'] = np.mean(zcr)

            sc = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['mean_spectral_centroid'] = np.mean(sc)

        except Exception as e:
            st.error(f"Error in feature extraction: {e}")
            return None

        return features

    # Choose input method
    method = st.radio("Choose Input Method:", ["🎧 Audio File Upload", "📄 CSV Upload"])

    if method == "🎧 Audio File Upload":
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            audio_data, sr = librosa.load(audio_file, sr=None)

            features = extract_features(audio_data, sr)

            if features:
                st.write("📊 Extracted Features:")
                st.json(features)

                input_df = pd.DataFrame([features])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)

                label = "👩 Female" if prediction[0] == 0 else "👨 Male"
                st.success(f"Predicted Gender: **{label}**")

    elif method == "📄 CSV Upload":
        st.markdown("""
        Upload a `.csv` file with the following 10 features:

        - mfcc_1_mean, mean_pitch, mfcc_3_mean, mfcc_5_mean  
        - zero_crossing_rate, rms_energy, mean_spectral_centroid  
        - std_pitch, mfcc_2_mean, log_energy
        """)

        csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if csv_file is not None:
            df_csv = pd.read_csv(csv_file)

            required_cols = [
                'mfcc_1_mean', 'mean_pitch', 'mfcc_3_mean', 'mfcc_5_mean',
                'zero_crossing_rate', 'rms_energy', 'mean_spectral_centroid',
                'std_pitch', 'mfcc_2_mean', 'log_energy'
            ]

            missing = [col for col in required_cols if col not in df_csv.columns]
            if missing:
                st.error(f"Missing columns in CSV: {missing}")
            else:
                st.dataframe(df_csv.head())

                input_scaled = scaler.transform(df_csv[required_cols])
                predictions = model.predict(input_scaled)

                df_csv["Predicted_Label"] = ["Female" if p == 0 else "Male" for p in predictions]
                st.success("✅ Prediction Complete")
                st.dataframe(df_csv)

                # Option to download
                csv_download = df_csv.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions CSV", data=csv_download, file_name="predictions.csv", mime="text/csv")



elif menu == "Clustering":
    st.title("🔍 Voice Clustering Analysis")
    st.markdown("Compare the clustering results of multiple unsupervised models using PCA visualizations and Silhouette Scores.")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Step 1: Prepare data
    X = df.drop(columns=["label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 2: Fit models and calculate silhouette scores
    cluster_outputs = {}

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    cluster_outputs["KMeans"] = {
        "labels": kmeans_labels,
        "score": silhouette_score(X_scaled, kmeans_labels)
    }

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if np.sum(dbscan_labels != -1) > 1 else -1
    cluster_outputs["DBSCAN"] = {
        "labels": dbscan_labels,
        "score": dbscan_score
    }

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(X_scaled)
    cluster_outputs["Agglomerative"] = {
        "labels": agg_labels,
        "score": silhouette_score(X_scaled, agg_labels)
    }

    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    cluster_outputs["GMM"] = {
        "labels": gmm_labels,
        "score": silhouette_score(X_scaled, gmm_labels)
    }


    # Step 3: Score Table
    st.subheader("📊 Clustering Model Silhouette Scores")
    scores_df = pd.DataFrame({
        "Model": list(cluster_outputs.keys()),
        "Silhouette Score": [v["score"] for v in cluster_outputs.values()]
    }).sort_values(by="Silhouette Score", ascending=False)
    st.dataframe(scores_df)

    # Step 4: Visualize All Models
    st.subheader("📈 Clustering PCA Visualizations")

    for model_name, result in cluster_outputs.items():
        st.markdown(f"### {model_name} (Silhouette Score: {result['score']:.4f})")
        labels = result["labels"]

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"{model_name} Clustering")
        st.pyplot(fig)

