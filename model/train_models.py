import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Paths
BASE_DIR       = "/home/cloud/Desktop/ML Assignment 2/heart-disease-classification"
DATASET_PATH   = os.path.join(BASE_DIR, "heart.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
MODEL_DIR      = os.path.join(BASE_DIR, "model")

print("Dataset Path :", DATASET_PATH)
print("Model Dir    :", MODEL_DIR)
print("Dataset exists?", os.path.exists(DATASET_PATH))

# Load Data
def load_and_preprocess():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"Shape: {df.shape}")
    print(f"Target distribution:")
    print(df["HeartDisease"].value_counts())

    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    return X, y, label_encoders, df

# Metrics
def compute_metrics(model_name, y_true, y_pred, y_prob):
    return {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "AUC"      : round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC"      : round(matthews_corrcoef(y_true, y_pred), 4),
    }

# Train Models
def train_all_models(X_train, X_test, y_train, y_test, scaler):
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
        "kNN"                : KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes"        : GaussianNB(),
        "Random Forest"      : RandomForestClassifier(n_estimators=200, random_state=42),
    }

    needs_scaling  = {"Logistic Regression", "kNN", "Naive Bayes"}
    all_metrics    = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        if name in needs_scaling:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(name, y_test, y_pred, y_prob)
        all_metrics.append(metrics)
        trained_models[name] = model

        print(f"  Accuracy : {metrics['Accuracy']}")
        print(f"  AUC      : {metrics['AUC']}")
        print(f"  Precision: {metrics['Precision']}")
        print(f"  Recall   : {metrics['Recall']}")
        print(f"  F1       : {metrics['F1']}")
        print(f"  MCC      : {metrics['MCC']}")
        print(f"  Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred))

    return all_metrics, trained_models

# Save Models
def save_artifacts(trained_models, scaler, label_encoders):
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree"      : "decision_tree.pkl",
        "kNN"                : "knn.pkl",
        "Naive Bayes"        : "naive_bayes.pkl",
        "Random Forest"      : "random_forest.pkl",
    }
    for name, model in trained_models.items():
        path = os.path.join(MODEL_DIR, filename_map[name])
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {path}")

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Saved: scaler.pkl")

    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    print("Saved: label_encoders.pkl")

# Main
if __name__ == "__main__":
    print("=" * 55)
    print("HEART DISEASE CLASSIFICATION - MODEL TRAINING")
    print("=" * 55)

    X, y, label_encoders, df = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape}")
    print(f"Test size : {X_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)

    test_df = X_test.copy()
    test_df["HeartDisease"] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nSaved test_data.csv: {TEST_DATA_PATH}")

    all_metrics, trained_models = train_all_models(
        X_train, X_test, y_train, y_test, scaler
    )

    print("\n" + "=" * 55)
    print("FINAL COMPARISON TABLE")
    print("=" * 55)
    results_df = pd.DataFrame(all_metrics).set_index("Model")
    print(results_df.to_string())

    save_artifacts(trained_models, scaler, label_encoders)

    print("\n" + "=" * 55)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 55)
