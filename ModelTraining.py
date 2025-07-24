import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline
import joblib

# === LOAD DATA ===
df = pd.read_csv("vocal_gender_features_new.csv")

# === FEATURES AND LABEL SPLIT ===
X = df.drop(columns=["label"])
y = df["label"].values

# === SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SPLIT DATA ===
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# === EVALUATION FUNCTION ===
def evaluate_classifier(model, model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n🔍 {model_name} Evaluation:")
    print(f"✅ Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"🎯 Precision : {precision_score(y_test, preds):.4f}")
    print(f"🔁 Recall    : {recall_score(y_test, preds):.4f}")
    print(f"🏅 F1 Score  : {f1_score(y_test, preds):.4f}")
    if probs is not None:
        print(f"📈 ROC AUC   : {roc_auc_score(y_test, probs):.4f}")
    print("\n🧾 Report:\n", classification_report(y_test, preds))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# === TEST MULTIPLE MODELS ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True, C=2, gamma='scale', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, solver='lbfgs'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
    "MLP (Baseline)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

for name, model in models.items():
    evaluate_classifier(model, name)

# === HYPERPARAMETER TUNING (MLP) ===
print("\n🔧 Tuning MLPClassifier...")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (128, 64), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [300]
}
grid = GridSearchCV(MLPClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)
print("🏁 Best MLP Parameters:", grid.best_params_)
evaluate_classifier(grid.best_estimator_, "Tuned MLP")

# === FINAL PIPELINE ===
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', grid.best_estimator_)
])
final_pipeline.fit(np.vstack([X_train, X_val, X_test]), np.concatenate([y_train, y_val, y_test]))
joblib.dump(final_pipeline, 'voice_gender_classifier_all_features.pkl')
print("\n✅ Final model saved as 'voice_gender_classifier_all_features.pkl'")
