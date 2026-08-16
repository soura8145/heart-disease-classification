# Heart Disease Classification - ML Assignment

## a. Problem Statement
Predict whether a patient has heart disease based on clinical features.
Binary Classification: 0 = No Disease, 1 = Disease.
Early prediction enables timely medical intervention and saves lives.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository / Kaggle
- **Link**: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Instances**: 918 rows
- **Features**: 12 (11 input features + 1 target)
- **Target**: HeartDisease (0 = No Disease, 1 = Disease)
- **Type**: Binary Classification
- **Class Distribution**: 508 Disease (55.3%) | 410 No Disease (44.7%)

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
https://github.com/YOUR_USERNAME/heart-disease-classification

## d. Models Used

### Comparison Table
| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.8696   | 0.8971 | 0.8482    | 0.9314 | 0.8879 | 0.7374 |
| Decision Tree       | 0.8098   | 0.8582 | 0.8252    | 0.8333 | 0.8293 | 0.6146 |
| kNN                 | 0.8913   | 0.9277 | 0.8942    | 0.9118 | 0.9029 | 0.7797 |
| Naive Bayes         | 0.8913   | 0.9280 | 0.8942    | 0.9118 | 0.9029 | 0.7797 |
| Random Forest       | 0.8913   | 0.9298 | 0.8868    | 0.9216 | 0.9038 | 0.7797 |

### Observations
| ML Model Name       | Observation                                                             |
|---------------------|-------------------------------------------------------------------------|
| Logistic Regression | Strong baseline at 86.96% accuracy. High recall (0.9314) means it      |
|                     | correctly identifies most disease cases. Good AUC (0.8971) shows       |
|                     | strong probability calibration. Slightly lower precision suggests       |
|                     | some false positives.                                                   |
| Decision Tree       | Weakest performer at 80.98% accuracy. Prone to overfitting despite     |
|                     | max_depth=5 constraint. Lowest AUC (0.8582) and MCC (0.6146).         |
|                     | Most interpretable model but weakest generalization.                   |
| kNN                 | Strong performer at 89.13% accuracy. Benefits greatly from feature     |
|                     | scaling. High AUC (0.9277) shows excellent ranking ability. k=7       |
|                     | balances bias-variance tradeoff well on this dataset.                  |
| Naive Bayes         | Surprisingly strong at 89.13% accuracy despite feature independence    |
|                     | assumption. Best AUC among non-ensemble models (0.9280). Gaussian      |
|                     | distributions fit clinical features well. Fastest training model.      |
| Random Forest       | Best overall with highest AUC (0.9298) and F1 (0.9038). Ensemble      |
|                     | approach reduces variance effectively. High recall (0.9216) is         |
|                     | critical for medical diagnosis. Most robust for deployment.            |
| Overall Winner      | Random Forest - Best AUC (0.9298), Best F1 (0.9038),                  |
|                     | Best Recall (0.9216). Handles nonlinear feature interactions           |
|                     | better than single models. Ideal for clinical deployment.              |

## Streamlit App Link
https://your-app-name.streamlit.app

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py
