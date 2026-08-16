import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #C0392B;
        text-align: center;
        padding: 10px 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #C0392B;
        padding: 12px;
        border-radius: 8px;
        margin: 6px 0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #C0392B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2C3E50;
        border-bottom: 2px solid #C0392B;
        padding-bottom: 5px;
        margin: 15px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "Logistic Regression": os.path.join(BASE_DIR, "model", "logistic_regression.pkl"),
    "Decision Tree"      : os.path.join(BASE_DIR, "model", "decision_tree.pkl"),
    "kNN"                : os.path.join(BASE_DIR, "model", "knn.pkl"),
    "Naive Bayes"        : os.path.join(BASE_DIR, "model", "naive_bayes.pkl"),
    "Random Forest"      : os.path.join(BASE_DIR, "model", "random_forest.pkl"),
}

SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
NEEDS_SCALING = {"Logistic Regression", "kNN", "Naive Bayes"}

FEATURE_COLS = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
                "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
TARGET_COL   = "HeartDisease"

# ── Load Artifacts ─────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    path = MODEL_FILES[model_name]
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

# ── Preprocess Uploaded Data ───────────────────────────────
def preprocess(df):
    df = df.copy()
    mappings = {
        "Sex"            : {"M": 1, "F": 0},
        "ChestPainType"  : {"ATA": 1, "NAP": 2, "ASY": 0, "TA": 3},
        "RestingECG"     : {"Normal": 1, "ST": 2, "LVH": 0},
        "ExerciseAngina" : {"N": 0, "Y": 1},
        "ST_Slope"       : {"Up": 2, "Flat": 1, "Down": 0},
    }
    for col, mapping in mappings.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(mapping).fillna(df[col])
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df

# ── Compute Metrics ────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "AUC"      : round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC"      : round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ── Plots ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["No Disease","Disease"],
                yticklabels=["No Disease","Disease"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_prob, auc_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color="#C0392B", lw=2,
            label=f"AUC = {auc_score:.4f}")
    ax.plot([0,1],[0,1],"k--", lw=1.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#C0392B")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_metrics_bar(metrics, model_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#C0392B","#E74C3C","#E67E22","#F39C12","#27AE60","#2980B9"]
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, width=0.5)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Metrics - {model_name}", fontsize=11, fontweight="bold")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_all_comparison(all_results):
    metrics_list = ["Accuracy","AUC","Precision","Recall","F1","MCC"]
    x      = np.arange(len(metrics_list))
    width  = 0.15
    colors = ["#C0392B","#2980B9","#27AE60","#F39C12","#8E44AD"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (result, color) in enumerate(zip(all_results, colors)):
        vals   = [result[m] for m in metrics_list]
        offset = (i - len(all_results)/2) * width
        ax.bar(x + offset + width/2, vals, width,
               label=result["Model"], color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("All Models - Metric Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, model_name):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    features    = [FEATURE_COLS[i] for i in indices]
    vals        = importances[indices]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features, vals, color="#C0392B", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(f"Feature Importance - {model_name}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Disease Classifier")
    st.markdown("---")

    st.markdown("### 📁 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"],
        help="Upload the test_data.csv from the GitHub repository"
    )

    st.markdown("---")
    st.markdown("### 🤖 Select Model")
    selected_model = st.selectbox(
        "Choose Classification Model",
        list(MODEL_FILES.keys())
    )

    st.markdown("---")
    compare_all = st.checkbox("📊 Compare All Models", value=False)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset Info")
    st.markdown("""
    - **Source**: UCI Heart Disease
    - **Instances**: 918
    - **Features**: 12
    - **Target**: HeartDisease (0/1)
    - **Type**: Binary Classification
    """)

# ── MAIN HEADER ────────────────────────────────────────────
st.markdown('<div class="main-header">❤️ Heart Disease Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Binary Classification | UCI Heart Disease Dataset | 918 instances | 12 features</div>', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🔬 Model Evaluation", "📊 All Models", "📋 Data Explorer"])

# ── TAB 1: OVERVIEW ────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📌 Problem Statement</div>', unsafe_allow_html=True)
    st.markdown("""
    Cardiovascular disease is one of the leading causes of death globally.
    This app predicts **heart disease presence** using clinical measurements.
    We compare **5 ML classification models** on the UCI Heart Disease dataset.
    """)

    st.markdown('<div class="section-header">📋 Feature Description</div>', unsafe_allow_html=True)
    feature_df = pd.DataFrame({
        "Feature"    : ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
                        "FastingBS","RestingECG","MaxHR","ExerciseAngina",
                        "Oldpeak","ST_Slope","HeartDisease"],
        "Type"       : ["Numeric","Categorical","Categorical","Numeric","Numeric",
                        "Binary","Categorical","Numeric","Categorical",
                        "Numeric","Categorical","Target"],
        "Description": [
            "Age in years",
            "Sex: M=Male, F=Female",
            "Chest Pain: ATA/NAP/ASY/TA",
            "Resting Blood Pressure (mm Hg)",
            "Serum Cholesterol (mm/dl)",
            "Fasting Blood Sugar > 120 mg/dl",
            "Resting ECG: Normal/ST/LVH",
            "Maximum Heart Rate Achieved",
            "Exercise Induced Angina: Y/N",
            "ST depression induced by exercise",
            "ST Slope: Up/Flat/Down",
            "Target: 1=Disease, 0=No Disease"
        ]
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">📊 Model Results Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Model"    : ["Logistic Regression","Decision Tree","kNN","Naive Bayes","Random Forest"],
        "Accuracy" : [0.8696, 0.8098, 0.8913, 0.8913, 0.8913],
        "AUC"      : [0.8971, 0.8582, 0.9277, 0.9280, 0.9298],
        "Precision": [0.8482, 0.8252, 0.8942, 0.8942, 0.8868],
        "Recall"   : [0.9314, 0.8333, 0.9118, 0.9118, 0.9216],
        "F1"       : [0.8879, 0.8293, 0.9029, 0.9029, 0.9038],
        "MCC"      : [0.7374, 0.6146, 0.7797, 0.7797, 0.7797],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.success("🏆 Overall Winner: Random Forest (Best AUC: 0.9298, Best F1: 0.9038)")

# ── TAB 2: MODEL EVALUATION ────────────────────────────────
with tab2:
    if uploaded_file is None:
        st.info("📂 Please upload test_data.csv using the sidebar to evaluate models.")
        st.markdown("**Expected CSV format:**")
        sample = pd.DataFrame({
            "Age":[40,49],"Sex":["M","F"],"ChestPainType":["ATA","NAP"],
            "RestingBP":[140,160],"Cholesterol":[289,180],"FastingBS":[0,0],
            "RestingECG":["Normal","Normal"],"MaxHR":[172,156],
            "ExerciseAngina":["N","N"],"Oldpeak":[0.0,1.0],
            "ST_Slope":["Up","Flat"],"HeartDisease":[0,1]
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
    else:
        # Load and preprocess
        df_test  = pd.read_csv(uploaded_file)
        df_proc  = preprocess(df_test)
        X_test   = df_proc[FEATURE_COLS]
        y_test   = df_proc[TARGET_COL]

        st.success(f"✅ Loaded: {df_test.shape[0]} rows × {df_test.shape[1]} columns")

        # Load model and predict
        model  = load_model(selected_model)
        scaler = load_scaler()

        if model is None:
            st.error(f"❌ Model not found: {MODEL_FILES[selected_model]}")
        else:
            if selected_model in NEEDS_SCALING and scaler is not None:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test.values

            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)

            # Metrics cards
            st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
            cols = st.columns(6)
            indicators = {"Accuracy":"🎯","AUC":"📈","Precision":"🔍","Recall":"📡","F1":"⚖️","MCC":"🧮"}
            for col, (metric, val) in zip(cols, metrics.items()):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val:.4f}</div>
                        <div class="metric-label">{indicators[metric]} {metric}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Charts
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Confusion Matrix**")
                fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            with c2:
                st.markdown("**ROC Curve**")
                fig_roc = plot_roc(y_test, y_prob, metrics["AUC"], selected_model)
                st.pyplot(fig_roc, use_container_width=True)
                plt.close(fig_roc)
            with c3:
                st.markdown("**Metrics Bar Chart**")
                fig_bar = plot_metrics_bar(metrics, selected_model)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                st.markdown("---")
                fi1, fi2 = st.columns(2)
                with fi1:
                    st.markdown("**Feature Importance**")
                    fig_fi = plot_feature_importance(model, selected_model)
                    if fig_fi:
                        st.pyplot(fig_fi, use_container_width=True)
                        plt.close(fig_fi)
                with fi2:
                    st.markdown("**Importance Values**")
                    fi_df = pd.DataFrame({
                        "Feature"   : FEATURE_COLS,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False)
                    fi_df["Importance"] = fi_df["Importance"].round(4)
                    fi_df.insert(0, "Rank", range(1, len(fi_df)+1))
                    st.dataframe(fi_df, use_container_width=True, hide_index=True)

            # Classification report
            st.markdown("---")
            st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)
            report = classification_report(y_test, y_pred,
                                           target_names=["No Disease","Disease"],
                                           output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

            # Prediction preview
            st.markdown("---")
            st.markdown('<div class="section-header">🔍 Prediction Preview (first 20 rows)</div>', unsafe_allow_html=True)
            preview = X_test.copy().head(20)
            preview["Actual"]        = y_test.values[:20]
            preview["Predicted"]     = y_pred[:20]
            preview["Prob(Disease)"] = y_prob[:20].round(4)
            preview["Correct?"]      = preview.apply(
                lambda r: "✅" if r["Actual"] == r["Predicted"] else "❌", axis=1
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)

# ── TAB 3: ALL MODELS COMPARISON ───────────────────────────
with tab3:
    if uploaded_file is None:
        st.info("📂 Upload test_data.csv in sidebar to compare all models.")
    else:
        st.markdown('<div class="section-header">📊 All Models Comparison</div>', unsafe_allow_html=True)

        with st.spinner("Running all models..."):
            all_results = []
            scaler      = load_scaler()

            for m_name in MODEL_FILES.keys():
                m = load_model(m_name)
                if m is None:
                    continue
                try:
                    uploaded_file.seek(0)
                    df_c    = preprocess(pd.read_csv(uploaded_file))
                    Xc      = df_c[FEATURE_COLS]
                    yc      = df_c[TARGET_COL]
                    Xc_eval = scaler.transform(Xc) if m_name in NEEDS_SCALING and scaler else Xc.values
                    yc_pred = m.predict(Xc_eval)
                    yc_prob = m.predict_proba(Xc_eval)[:, 1]
                    m_met   = compute_metrics(yc, yc_pred, yc_prob)
                    m_met["Model"] = m_name
                    all_results.append(m_met)
                except Exception as e:
                    st.warning(f"{m_name} failed: {e}")

        if all_results:
            comp_df = pd.DataFrame(all_results).set_index("Model")

            def highlight_best(s):
                return ["background-color: #ABEBC6; font-weight:bold"
                        if v == s.max() else "" for v in s]

            st.dataframe(
                comp_df.style.apply(highlight_best, axis=0).format("{:.4f}"),
                use_container_width=True
            )
            st.caption("🟩 Green = Best value per metric")

            st.markdown("---")
            fig_comp = plot_all_comparison(all_results)
            st.pyplot(fig_comp, use_container_width=True)
            plt.close(fig_comp)

            best  = comp_df["F1"].idxmax()
            st.success(f"🏆 Best Model: {best} | "
                       f"Accuracy: {comp_df.loc[best,'Accuracy']:.4f} | "
                       f"F1: {comp_df.loc[best,'F1']:.4f} | "
                       f"AUC: {comp_df.loc[best,'AUC']:.4f}")

            # All ROC curves
            st.markdown("---")
            st.markdown("**ROC Curves - All Models**")
            colors  = ["#C0392B","#2980B9","#27AE60","#F39C12","#8E44AD"]
            fig_roc_all, ax = plt.subplots(figsize=(7, 5))
            for m_name, color in zip(MODEL_FILES.keys(), colors):
                m = load_model(m_name)
                if m is None:
                    continue
                try:
                    uploaded_file.seek(0)
                    df_r    = preprocess(pd.read_csv(uploaded_file))
                    Xr      = df_r[FEATURE_COLS]
                    yr      = df_r[TARGET_COL]
                    Xr_eval = scaler.transform(Xr) if m_name in NEEDS_SCALING and scaler else Xr.values
                    yr_prob = m.predict_proba(Xr_eval)[:, 1]
                    fpr, tpr, _ = roc_curve(yr, yr_prob)
                    auc_sc  = roc_auc_score(yr, yr_prob)
                    ax.plot(fpr, tpr, color=color, lw=2,
                            label=f"{m_name} (AUC={auc_sc:.3f})")
                except Exception:
                    pass
            ax.plot([0,1],[0,1],"k--", lw=1.5)
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.set_title("ROC Curves - All Models", fontsize=13, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_roc_all, use_container_width=True)
            plt.close(fig_roc_all)

# ── TAB 4: DATA EXPLORER ───────────────────────────────────
with tab4:
    if uploaded_file is None:
        st.info("📂 Upload test_data.csv to explore data.")
    else:
        uploaded_file.seek(0)
        df_exp = pd.read_csv(uploaded_file)

        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Rows",          df_exp.shape[0])
        e2.metric("Columns",       df_exp.shape[1])
        e3.metric("Heart Disease+", int(df_exp["HeartDisease"].sum()) if "HeartDisease" in df_exp.columns else "N/A")
        e4.metric("Missing Values", int(df_exp.isnull().sum().sum()))

        st.markdown("---")
        st.markdown("**Raw Data (first 20 rows)**")
        st.dataframe(df_exp.head(20), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Statistical Summary**")
        st.dataframe(df_exp.describe().round(3), use_container_width=True)

        st.markdown("---")
        st.markdown("**Correlation Heatmap**")
        df_num = preprocess(df_exp).select_dtypes(include=[np.number])
        fig_corr, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)
