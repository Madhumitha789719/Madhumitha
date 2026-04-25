# ============================================================
# A1 to A5 Full Correct Code
# Dataset: Transmission Line Fault Detection
# ============================================================

# Install required packages:
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost lime openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer

# SHAP optional
try:
    import shap
    shap_available = True
except:
    shap_available = False
    print("SHAP is not installed. SHAP part will be skipped.")

# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = r"C:\Users\gkzee\OneDrive\Desktop\classData.xlsx"

df = pd.read_excel(DATASET_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# CORRECT TARGET AND FEATURE SELECTION
# ============================================================

target_columns = ['G', 'C', 'B', 'A']
feature_columns = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

X = df[feature_columns]

# Combine G C B A into one fault label
df["Fault_Type"] = df[target_columns].astype(str).agg("".join, axis=1)

y = df["Fault_Type"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nFeatures used:")
print(feature_columns)

print("\nTarget used:")
print("Fault_Type created from G, C, B, A")

print("\nFault classes:")
for i, name in enumerate(label_encoder.classes_):
    print(i, "=", name)

# ============================================================
# TRAIN TEST SPLIT
# No stratify used because some classes may have only 1 sample
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, Xtr, Xte, ytr, yte, title):
    print("\n===================================================")
    print(title)
    print("===================================================")

    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    acc = accuracy_score(yte, y_pred)

    print("Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(yte, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(yte, y_pred))

    return acc, model

# ============================================================
# A1 FEATURE CORRELATION ANALYSIS
# ============================================================

print("\n\n================ A1: FEATURE CORRELATION ANALYSIS ================")

corr_matrix = X.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("A1: Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nHighly correlated feature pairs:")

threshold = 0.85

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            print(
                corr_matrix.columns[i],
                "and",
                corr_matrix.columns[j],
                "Correlation =",
                round(corr_matrix.iloc[i, j], 3)
            )

baseline_model = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42
)

baseline_acc, baseline_model = evaluate_model(
    baseline_model,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "Baseline XGBoost Model Without Feature Reduction"
)

# ============================================================
# A2 PCA WITH 99% EXPLAINED VARIANCE
# ============================================================

print("\n\n================ A2: PCA WITH 99% EXPLAINED VARIANCE ================")

pca_99 = PCA(n_components=0.99)

X_train_pca_99 = pca_99.fit_transform(X_train_scaled)
X_test_pca_99 = pca_99.transform(X_test_scaled)

print("Original number of features:", X_train_scaled.shape[1])
print("PCA components retained:", X_train_pca_99.shape[1])
print("Explained variance:", round(sum(pca_99.explained_variance_ratio_) * 100, 2), "%")

pca99_model = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42
)

pca99_acc, pca99_model = evaluate_model(
    pca99_model,
    X_train_pca_99,
    X_test_pca_99,
    y_train,
    y_test,
    "A2: XGBoost After PCA 99% Variance"
)

# ============================================================
# A3 PCA WITH 95% EXPLAINED VARIANCE
# ============================================================

print("\n\n================ A3: PCA WITH 95% EXPLAINED VARIANCE ================")

pca_95 = PCA(n_components=0.95)

X_train_pca_95 = pca_95.fit_transform(X_train_scaled)
X_test_pca_95 = pca_95.transform(X_test_scaled)

print("Original number of features:", X_train_scaled.shape[1])
print("PCA components retained:", X_train_pca_95.shape[1])
print("Explained variance:", round(sum(pca_95.explained_variance_ratio_) * 100, 2), "%")

pca95_model = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42
)

pca95_acc, pca95_model = evaluate_model(
    pca95_model,
    X_train_pca_95,
    X_test_pca_95,
    y_train,
    y_test,
    "A3: XGBoost After PCA 95% Variance"
)

# ============================================================
# A4 SEQUENTIAL FEATURE SELECTION
# ============================================================

print("\n\n================ A4: SEQUENTIAL FEATURE SELECTION ================")

sfs_estimator = RandomForestClassifier(random_state=42)

sfs = SequentialFeatureSelector(
    estimator=sfs_estimator,
    n_features_to_select=3,
    direction="forward",
    scoring="accuracy",
    cv=3
)

sfs.fit(X_train_scaled, y_train)

selected_features = X.columns[sfs.get_support()]

print("Selected features:")
print(selected_features.tolist())

X_train_sfs = sfs.transform(X_train_scaled)
X_test_sfs = sfs.transform(X_test_scaled)

sfs_model = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42
)

sfs_acc, sfs_model = evaluate_model(
    sfs_model,
    X_train_sfs,
    X_test_sfs,
    y_train,
    y_test,
    "A4: XGBoost After Sequential Feature Selection"
)

# ============================================================
# A5 LIME EXPLANATION
# ============================================================

print("\n\n================ A5: LIME EXPLANATION ================")

lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_columns,
    class_names=[str(c) for c in label_encoder.classes_],
    mode="classification"
)

sample_index = 0

lime_exp = lime_explainer.explain_instance(
    X_test_scaled[sample_index],
    baseline_model.predict_proba,
    num_features=6
)

print("\nLIME explanation for one test sample:")

for feature, value in lime_exp.as_list():
    print(feature, ":", value)

lime_exp.save_to_file("lime_explanation.html")

print("\nLIME explanation saved as lime_explanation.html")

# ============================================================
# A5 SHAP EXPLANATION
# ============================================================

print("\n\n================ A5: SHAP EXPLANATION ================")

if shap_available:
    shap_explainer = shap.TreeExplainer(baseline_model)
    shap_values = shap_explainer.shap_values(X_test_scaled)

    print("SHAP values calculated successfully")

    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_columns
    )

    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_columns,
        plot_type="bar"
    )
else:
    print("SHAP skipped because shap module is not installed.")
    print("To install SHAP, run:")
    print("pip install shap")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n\n================ FINAL COMPARISON TABLE ================")

results = pd.DataFrame({
    "Method": [
        "Baseline - No Reduction",
        "PCA - 99% Variance",
        "PCA - 95% Variance",
        "Sequential Feature Selection"
    ],
    "Features / Components": [
        X_train_scaled.shape[1],
        X_train_pca_99.shape[1],
        X_train_pca_95.shape[1],
        X_train_sfs.shape[1]
    ],
    "Accuracy (%)": [
        round(baseline_acc * 100, 2),
        round(pca99_acc * 100, 2),
        round(pca95_acc * 100, 2),
        round(sfs_acc * 100, 2)
    ]
})

print(results)

plt.figure(figsize=(10, 5))
sns.barplot(data=results, x="Method", y="Accuracy (%)")
plt.title("Accuracy Comparison of Feature Reduction Methods")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

results.to_excel("feature_reduction_results.xlsx", index=False)

print("\nResults saved as feature_reduction_results.xlsx")
print("\nAll A1 to A5 completed successfully.")
