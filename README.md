# Heart Disease Classification - ML Assignment 2

## a. Problem Statement

Cardiovascular disease is one of the leading causes of death globally. This project predicts whether a patient has heart disease based on 11 clinical features using 5 different Machine Learning classification models.

**Problem Type:** Binary Classification
- 0 = No Heart Disease
- 1 = Heart Disease Present

Early prediction of heart disease enables timely medical intervention, reduces mortality, and helps in preventive healthcare decisions.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository / Kaggle
- **Dataset Link:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Total Instances:** 918 rows
- **Total Features:** 12 (11 input features + 1 target)
- **Target Variable:** HeartDisease (0 = No Disease, 1 = Disease)
- **Class Distribution:** 508 Disease (55.3%) | 410 No Disease (44.7%)
- **Missing Values:** None
- **Type:** Binary Classification

### Feature List

| Feature          | Type        | Description                             |
|------------------|-------------|-----------------------------------------|
| Age              | Numeric     | Age in years                            |
| Sex              | Categorical | M = Male, F = Female                    |
| ChestPainType    | Categorical | ATA / NAP / ASY / TA                    |
| RestingBP        | Numeric     | Resting Blood Pressure (mm Hg)          |
| Cholesterol      | Numeric     | Serum Cholesterol (mm/dl)               |
| FastingBS        | Binary      | Fasting Blood Sugar > 120 mg/dl         |
| RestingECG       | Categorical | Normal / ST / LVH                       |
| MaxHR            | Numeric     | Maximum Heart Rate Achieved             |
| ExerciseAngina   | Categorical | Exercise Induced Angina: Y / N          |
| Oldpeak          | Numeric     | ST depression induced by exercise       |
| ST_Slope         | Categorical | Slope of peak exercise ST: Up/Flat/Down |
| HeartDisease     | Target      | 1 = Disease, 0 = No Disease             |

## c. GitHub Repository Link

https://github.com/soura8145/heart-disease-classification

## d. Models Used

Implemented 5 classification models on the same dataset with 80/20 train-test split:

1. **Logistic Regression** - Linear classifier using sigmoid function
2. **Decision Tree** - Tree-based classifier with gini criterion
3. **K-Nearest Neighbors (kNN)** - Instance-based classifier with k=7
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** - Ensemble of 200 decision trees

### Comparison Table - Evaluation Metrics

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.8696   | 0.8971 | 0.8482    | 0.9314 | 0.8879 | 0.7374 |
| Decision Tree       | 0.8098   | 0.8582 | 0.8252    | 0.8333 | 0.8293 | 0.6146 |
| kNN                 | 0.8913   | 0.9277 | 0.8942    | 0.9118 | 0.9029 | 0.7797 |
| Naive Bayes         | 0.8913   | 0.9280 | 0.8942    | 0.9118 | 0.9029 | 0.7797 |
| Random Forest       | 0.8913   | 0.9298 | 0.8868    | 0.9216 | 0.9038 | 0.7797 |

### Observations - Model Performance

| ML Model Name       | Observation about Model Performance                                                                                                                                                                        |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Strong baseline with 86.96% accuracy. High recall (0.9314) means it correctly identifies most disease cases. Good AUC (0.8971) shows strong probability calibration. Slightly lower precision suggests some false positives. Suitable for linear decision boundaries. |
| Decision Tree       | Weakest performer at 80.98% accuracy. Prone to overfitting despite max_depth=5 constraint. Lowest AUC (0.8582) and MCC (0.6146) among all models. Most interpretable model with clear if-then rules, but weakest generalization on unseen data. |
| kNN                 | Strong performer at 89.13% accuracy. Benefits greatly from feature scaling using StandardScaler. High AUC (0.9277) shows excellent ranking ability. k=7 balances bias-variance tradeoff well. Sensitive to feature magnitude and distance metric. |
| Naive Bayes         | Surprisingly strong at 89.13% accuracy despite the naive feature independence assumption. Best AUC among non-ensemble models (0.9280). Gaussian distributions fit clinical features reasonably well. Fastest training and inference time. |
| Random Forest       | Best overall performer with highest AUC (0.9298) and F1 (0.9038). Ensemble of 200 trees reduces variance effectively. High recall (0.9216) is critical for medical diagnosis. Most robust and reliable model for real-world clinical deployment. |
| **Overall Winner for this Dataset** | **Random Forest (Ensemble)** - Achieved highest AUC (0.9298), highest F1 score (0.9038), and best MCC (0.7797). The ensemble approach effectively handles nonlinear feature interactions and reduces overfitting through bagging. Ideal for clinical decision-making. |

## Streamlit App

**Live Demo:** https://heart-disease-souradeep.streamlit.app

### Features

- CSV upload option (upload test_data.csv)
- Model selection dropdown (5 models)
- Display of all 6 evaluation metrics
- Confusion matrix visualization
- ROC curve for each model
- Classification report
- Feature importance chart (for tree-based models)
- All models comparison view
- Data explorer with statistics and correlation heatmap

## How to Run Locally

```bash
# Clone repository
git clone https://github.com/soura8145/heart-disease-classification.git
cd heart-disease-classification

# Install dependencies
pip install -r requirements.txt

# Train models (if not already trained)
python model/train_models.py

# Run Streamlit app
streamlit run app.py