# ============================================================
# 22AIE213 - Lab Session 07
# Project: Transmission Line Fault Detection
# A2: Hyperparameter tuning using RandomizedSearchCV
# A3: Classification using XGBoost and tabulation of results
# Dataset path: C:\Users\gkzee\OneDrive\Desktop\classData.xlsx
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# Safe import for XGBoost
# -----------------------------
try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost is not installed.")
    print("Open Command Prompt and run:")
    print("python -m pip install xgboost")
    sys.exit()

# ============================================================
# 1. LOAD DATASET
# ============================================================
file_path = r"C:\Users\gkzee\OneDrive\Desktop\classData.xlsx"

# fallback path for other environments
if not os.path.exists(file_path):
    file_path = r"/mnt/data/classData.xlsx"

if not os.path.exists(file_path):
    print("ERROR: Dataset file not found.")
    sys.exit()

df = pd.read_excel(file_path)

print("Dataset loaded successfully.")
print("Shape of dataset:", df.shape)
print("\nColumns in dataset:")
print(df.columns.tolist())

# ============================================================
# 2. BASIC CLEANING
# ============================================================
df = df.drop_duplicates()
df = df.dropna()

print("\nShape after cleaning:", df.shape)

# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================
required_columns = ['G', 'C', 'B', 'A', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

for col in required_columns:
    if col not in df.columns:
        print(f"ERROR: Required column '{col}' not found in dataset.")
        sys.exit()

# ============================================================
# 4. CREATE TARGET LABEL
# Combine G, C, B, A to form class label
# Example: 1 0 0 1 -> "1001"
# ============================================================
df["Fault_Type"] = df[["G", "C", "B", "A"]].astype(str).agg(''.join, axis=1)

print("\nFault type counts:")
print(df["Fault_Type"].value_counts())

# Features and target
X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].copy()
y = df["Fault_Type"].copy()

# Encode target manually
class_names = sorted(y.unique())
class_to_num = {cls: i for i, cls in enumerate(class_names)}
num_to_class = {i: cls for cls, i in class_to_num.items()}
y_encoded = y.map(class_to_num)

print("\nClass encoding:")
for cls, num in class_to_num.items():
    print(f"{cls} --> {num}")

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ============================================================
# 6. A2 - RANDOMIZEDSEARCHCV FOR XGBOOST
# ============================================================
xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=1
)

# Smaller and safer parameter search space
param_dist = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "min_child_weight": [1, 3, 5]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=1
)

print("\nRunning RandomizedSearchCV...")
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\n==================== A2 OUTPUT ====================")
print("Best Parameters:")
print(random_search.best_params_)
print("Best Cross-Validation Accuracy:", round(random_search.best_score_, 4))

# ============================================================
# 7. A3 - TRAIN AND TEST PREDICTIONS
# ============================================================
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Train metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average="weighted", zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average="weighted", zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

# ============================================================
# 8. TABULATED RESULTS
# ============================================================
results_table = pd.DataFrame({
    "Performance Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Training Result": [
        round(train_accuracy, 4),
        round(train_precision, 4),
        round(train_recall, 4),
        round(train_f1, 4)
    ],
    "Testing Result": [
        round(test_accuracy, 4),
        round(test_precision, 4),
        round(test_recall, 4),
        round(test_f1, 4)
    ]
})

print("\n==================== A3 OUTPUT ====================")
print("\nTabulated Train and Test Results:")
print(results_table.to_string(index=False))

# Save table
results_table.to_csv("A2_A3_XGBoost_Results.csv", index=False)
results_table.to_excel("A2_A3_XGBoost_Results.xlsx", index=False)

# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================
target_names = [num_to_class[i] for i in range(len(class_names))]

print("\nClassification Report on Test Data:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=target_names,
    zero_division=0
))

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.show()

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

importance_df.to_csv("XGBoost_Feature_Importance.csv", index=False)
importance_df.to_excel("XGBoost_Feature_Importance.xlsx", index=False)

plt.figure(figsize=(8, 5))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importance - XGBoost")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================
pred_df = X_test.copy()
pred_df["Actual_Class_Number"] = y_test.values
pred_df["Predicted_Class_Number"] = y_test_pred
pred_df["Actual_Class_Label"] = [num_to_class[i] for i in y_test.values]
pred_df["Predicted_Class_Label"] = [num_to_class[i] for i in y_test_pred]

pred_df.to_csv("XGBoost_Test_Predictions.csv", index=False)
pred_df.to_excel("XGBoost_Test_Predictions.xlsx", index=False)

# ============================================================
# 13. OBSERVATIONS
# ============================================================
print("\n==================== OBSERVATIONS ====================")

if train_accuracy - test_accuracy > 0.05:
    print("1. Training accuracy is noticeably higher than testing accuracy, so slight overfitting may be present.")
else:
    print("1. Training and testing accuracy are close, so the model generalizes well.")

print("2. RandomizedSearchCV was used to tune the XGBoost hyperparameters.")
print("3. Accuracy, Precision, Recall, and F1-Score were used to evaluate the classifier.")
print("4. XGBoost is effective for transmission line fault detection using current and voltage values.")
print("5. Feature importance indicates which electrical parameters influence fault classification most.")

print("\nFiles saved successfully:")
print("1. A2_A3_XGBoost_Results.csv")
print("2. A2_A3_XGBoost_Results.xlsx")
print("3. XGBoost_Feature_Importance.csv")
print("4. XGBoost_Feature_Importance.xlsx")
print("5. XGBoost_Test_Predictions.csv")
print("6. XGBoost_Test_Predictions.xlsx")
