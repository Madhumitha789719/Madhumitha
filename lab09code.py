# ============================================================
# A1, A2, A3 - STACKING + PIPELINE + LIME
# Full modular code as per given lab instructions
# Dataset path included
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# FUNCTION 1: Load dataset from Excel
# ============================================================
def load_dataset(file_path):
    data_frame = pd.read_excel(file_path)
    return data_frame


# ============================================================
# FUNCTION 2: Clean dataset
# ============================================================
def clean_dataset(data_frame):
    data_frame = data_frame.copy()
    data_frame = data_frame.drop_duplicates()
    data_frame.columns = data_frame.columns.str.strip()
    return data_frame


# ============================================================
# FUNCTION 3: Split features and target
# ============================================================
def split_features_target(data_frame, target_column):
    x_data = data_frame.drop(columns=[target_column])
    y_data = data_frame[target_column]
    return x_data, y_data


# ============================================================
# FUNCTION 4: Get numeric and categorical columns
# ============================================================
def get_column_types(x_data):
    num_cols = x_data.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = x_data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


# ============================================================
# FUNCTION 5: Build preprocessor
# ============================================================
def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor


# ============================================================
# FUNCTION 6: Build stacking model
# ============================================================
def build_stacking_model(problem_type):
    if problem_type == "classification":
        base_models = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=100, random_state=42))
        ]

        final_model = LogisticRegression(max_iter=2000)

        stack_model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_model,
            cv=5
        )

    elif problem_type == "regression":
        base_models = [
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingRegressor(random_state=42)),
            ("et", ExtraTreesRegressor(n_estimators=100, random_state=42))
        ]

        final_model = LinearRegression()

        stack_model = StackingRegressor(
            estimators=base_models,
            final_estimator=final_model,
            cv=5
        )

    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")

    return stack_model


# ============================================================
# FUNCTION 7: Build full pipeline
# ============================================================
def build_full_pipeline(preprocessor, stack_model):
    full_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", stack_model)
    ])
    return full_pipe


# ============================================================
# FUNCTION 8: Encode target if classification and target is text
# ============================================================
def encode_target_if_needed(y_data, problem_type):
    encoder = None

    if problem_type == "classification" and y_data.dtype == "object":
        encoder = LabelEncoder()
        y_data = encoder.fit_transform(y_data.astype(str))

    return y_data, encoder


# ============================================================
# FUNCTION 9: Train model
# ============================================================
def train_model(full_pipe, x_train, y_train):
    full_pipe.fit(x_train, y_train)
    return full_pipe


# ============================================================
# FUNCTION 10: Evaluate classification
# ============================================================
def evaluate_classification(trained_model, x_test, y_test, target_encoder=None):
    y_pred = trained_model.predict(x_test)

    if target_encoder is not None:
        class_labels = list(range(len(target_encoder.classes_)))
        class_names = [str(c) for c in target_encoder.classes_]
        report = classification_report(
            y_test, y_pred,
            labels=class_labels,
            target_names=class_names,
            zero_division=0
        )
        matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
    else:
        report = classification_report(y_test, y_pred, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)

    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": report,
        "matrix": matrix,
        "predictions": y_pred
    }

    return result


# ============================================================
# FUNCTION 11: Evaluate regression
# ============================================================
def evaluate_regression(trained_model, x_test, y_test):
    y_pred = trained_model.predict(x_test)

    result = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "predictions": y_pred
    }

    return result


# ============================================================
# FUNCTION 12: Get transformed train data for LIME
# ============================================================
def get_transformed_train_data(trained_model, x_train):
    x_train_tf = trained_model.named_steps["preprocessor"].transform(x_train)

    if hasattr(x_train_tf, "toarray"):
        x_train_tf = x_train_tf.toarray()

    return x_train_tf


# ============================================================
# FUNCTION 13: Get transformed test data for LIME
# ============================================================
def get_transformed_test_data(trained_model, x_test):
    x_test_tf = trained_model.named_steps["preprocessor"].transform(x_test)

    if hasattr(x_test_tf, "toarray"):
        x_test_tf = x_test_tf.toarray()

    return x_test_tf


# ============================================================
# FUNCTION 14: Get transformed feature names
# ============================================================
def get_feature_names(trained_model, x_train):
    num_cols, cat_cols = get_column_types(x_train)
    feature_names = []

    if len(num_cols) > 0:
        feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        encoder = trained_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(encoded_names)

    return feature_names


# ============================================================
# FUNCTION 15: LIME explanation
# ============================================================
def run_lime_explanation(trained_model, x_train, x_test, problem_type, class_names=None, sample_index=0):
    x_train_tf = get_transformed_train_data(trained_model, x_train)
    x_test_tf = get_transformed_test_data(trained_model, x_test)
    feature_names = get_feature_names(trained_model, x_train)

    if problem_type == "classification":
        explainer = LimeTabularExplainer(
            training_data=x_train_tf,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification"
        )

        predict_fn = trained_model.named_steps["model"].predict_proba

        exp = explainer.explain_instance(
            x_test_tf[sample_index],
            predict_fn
        )

    else:
        explainer = LimeTabularExplainer(
            training_data=x_train_tf,
            feature_names=feature_names,
            mode="regression"
        )

        predict_fn = trained_model.named_steps["model"].predict

        exp = explainer.explain_instance(
            x_test_tf[sample_index],
            predict_fn
        )

    return exp


# ============================================================
# MAIN SECTION
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # YOUR DATASET PATH
    # --------------------------------------------------------
    file_path = r"C:\Users\gkzee\OneDrive\Desktop\classData.xlsx"

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    df = load_dataset(file_path)
    df = clean_dataset(df)

    print("\n======================================================")
    print("DATASET LOADED SUCCESSFULLY")
    print("======================================================")
    print("\nDataset Shape:", df.shape)

    print("\nColumn Names in Dataset:")
    for col in df.columns:
        print(col)

    # --------------------------------------------------------
    # USER INPUTS
    # --------------------------------------------------------
    print("\n======================================================")
    print("ENTER TARGET COLUMN AND PROBLEM TYPE")
    print("======================================================")

    target_column = input("\nEnter the exact target column name from above list: ").strip()
    problem_type = input("Enter problem type ('classification' or 'regression'): ").strip().lower()

    # --------------------------------------------------------
    # SPLIT X AND Y
    # --------------------------------------------------------
    x_data, y_data = split_features_target(df, target_column)

    # --------------------------------------------------------
    # ENCODE TARGET IF NEEDED
    # --------------------------------------------------------
    y_data, target_encoder = encode_target_if_needed(y_data, problem_type)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------
    num_cols, cat_cols = get_column_types(x_data)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # --------------------------------------------------------
    # STACKING MODEL
    # --------------------------------------------------------
    stack_model = build_stacking_model(problem_type)

    # --------------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------------
    full_pipe = build_full_pipeline(preprocessor, stack_model)

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------
    if problem_type == "classification":
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data,
            test_size=0.2,
            random_state=42,
            stratify=y_data
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data,
            test_size=0.2,
            random_state=42
        )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------
    trained_model = train_model(full_pipe, x_train, y_train)

    # --------------------------------------------------------
    # A1 + A2 OUTPUT
    # --------------------------------------------------------
    print("\n======================================================")
    print("A1 + A2 OUTPUT")
    print("STACKING MODEL WITH PIPELINE")
    print("======================================================")

    if problem_type == "classification":
        result = evaluate_classification(trained_model, x_test, y_test, target_encoder)

        print("\nAccuracy:")
        print(result["accuracy"])

        print("\nClassification Report:")
        print(result["report"])

        print("\nConfusion Matrix:")
        print(result["matrix"])

        if target_encoder is not None:
            class_names = [str(c) for c in target_encoder.classes_]
        else:
            class_names = [str(c) for c in sorted(pd.Series(y_train).unique())]

    else:
        result = evaluate_regression(trained_model, x_test, y_test)

        print("\nMAE:")
        print(result["mae"])

        print("\nRMSE:")
        print(result["rmse"])

        print("\nR2 Score:")
        print(result["r2"])

        class_names = None

    # --------------------------------------------------------
    # A3 OUTPUT - LIME
    # --------------------------------------------------------
    print("\n======================================================")
    print("A3 OUTPUT")
    print("LIME EXPLANATION")
    print("======================================================")

    lime_exp = run_lime_explanation(
        trained_model=trained_model,
        x_train=x_train,
        x_test=x_test,
        problem_type=problem_type,
        class_names=class_names,
        sample_index=0
    )

    print("\nTop Feature Contributions:")
    for item in lime_exp.as_list():
        print(item)

    fig = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
