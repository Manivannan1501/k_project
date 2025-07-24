
# Human Voice Clustering and Classification

## Problem Statement
Develop a machine learning–based model to classify and cluster human voice samples based on extracted audio features. The system will preprocess the dataset, apply clustering and classification models, and evaluate their performance. The final application will provide an interface for uploading audio samples and receiving predictions, deployed via Streamlit.

---

## Business Use Cases

1. **Speaker Identification**  
   Identify individuals based on their voice features.

2. **Gender Classification**  
   Classify voices as male or female for applications like call center analytics.

3. **Speech Analytics**  
   Extract insights from audio data for industries such as media, security, and customer service.

4. **Assistive Technologies**  
   Improve accessibility solutions by analyzing voice patterns.

---

## Approach

**Data Preparation:**  
- Use the provided dataset containing extracted voice features  
- Split into training, validation, and test sets

**Data Cleaning and Preprocessing:**  
- Handle missing or inconsistent data  
- Normalize numerical features for consistency

**Exploratory Data Analysis (EDA):**  
- Visualize feature distributions and correlations  
- Analyze patterns in spectral and pitch-related features

**Model Development:**  
- Apply clustering techniques (KMeans, DBSCAN)  
- Train classification models (Random Forest, SVM, Neural Networks, XGBoost)  
- Experiment with feature selection and hyperparameter tuning

**Model Evaluation and Comparison:**  
- Accuracy, precision, recall, F1-score  
- Silhouette score (for clustering)  
- Confusion matrix

**Application Development:**  
- Build a Streamlit-based interface to upload audio samples and receive predictions

---

## Data Flow and Architecture

**Data Collection & Preprocessing:**  
- Use extracted data with normalized spectral, pitch, and MFCC features from audio files

**Processing Pipeline:**  
- Implement feature engineering, dimensionality reduction, and normalization

**Model Training:**  
- Train machine learning models using scikit-learn or TensorFlow  
- Save trained models for deployment

**Deployment:**  
- Create a Streamlit-based front-end for user interaction

---

## Dataset

**File:** `human_voice_clustering.zip` (unpacked to CSV)  
This dataset consists of extracted audio features from human voice recordings, including:  

- mean_spectral_centroid  
- std_spectral_centroid  
- mean_spectral_bandwidth  
- std_spectral_bandwidth  
- mean_spectral_contrast  
- mean_spectral_flatness  
- mean_spectral_rolloff  
- zero_crossing_rate  
- rms_energy  
- mean_pitch, min_pitch, max_pitch, std_pitch  
- spectral_skew, spectral_kurtosis  
- energy_entropy, log_energy  
- mfcc_1_mean to mfcc_13_mean  
- mfcc_1_std to mfcc_13_std  
- **label** (0 = female, 1 = male)

---

## Results

By the end of this project, learners achieve:  

✅ A preprocessed and structured dataset  
✅ Multiple clustering and classification models trained and evaluated  
✅ A deployed Streamlit voice classification system with an audio feature upload interface

---

## Project Evaluation Metrics

- **Data Preprocessing Quality**: missing data handling, normalization  
- **Model Performance**: accuracy, precision, recall, F1-score for classification  
- **Clustering Performance**: silhouette score, cluster purity  
- **Application Functionality**: user-friendly Streamlit interface

---

## Technical Stack

- Python  
- scikit-learn  
- TensorFlow/Keras  
- Streamlit  
- Feature engineering  
- Audio feature extraction with librosa

---

## Deliverables

- Cleaned and preprocessed dataset  
- Trained models (Random Forest, SVM, MLP, XGBoost)  
- Performance metrics  
- Deployed Streamlit web app  
- Database logging with SQLite  
- Logs viewer with Streamlit

---

## How to Run

```bash
pip install -r requirements.txt
python init_db.py
streamlit run app.py
