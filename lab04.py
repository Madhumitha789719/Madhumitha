import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)

# -------------------------------------------------
# A1 FUNCTIONS – CLASSIFICATION METRICS
# -------------------------------------------------

def load_project_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]
    y = df['G']
    return X, y


def train_knn(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return cm, precision, recall, f1


# -------------------------------------------------
# A2 FUNCTIONS – REGRESSION METRICS
# -------------------------------------------------

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


# -------------------------------------------------
# A3 FUNCTIONS – SYNTHETIC TRAINING DATA
# -------------------------------------------------

def generate_training_data():
    np.random.seed(42)
    X = np.random.uniform(1, 10, (20, 2))
    y = np.array([0 if (x[0] + x[1]) < 10 else 1 for x in X])
    return X, y


def plot_training_data(X, y):
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Training Data Scatter Plot')
    plt.show()


# -------------------------------------------------
# A4 FUNCTIONS – TEST DATA & CLASSIFICATION
# -------------------------------------------------

def generate_test_data():
    x_vals = np.arange(0, 10, 0.1)
    y_vals = np.arange(0, 10, 0.1)
    X_test = np.array([[x, y] for x in x_vals for y in y_vals])
    return X_test


def plot_test_classification(model, X_test):
    y_pred = model.predict(X_test)
    colors = ['blue' if label == 0 else 'red' for label in y_pred]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('kNN Test Data Classification (k=3)')
    plt.show()


# -------------------------------------------------
# A5 FUNCTIONS – DIFFERENT k VALUES
# -------------------------------------------------

def plot_multiple_k(X_train, y_train, X_test, k_values):
    for k in k_values:
        model = train_knn(X_train, y_train, k)
        y_pred = model.predict(X_test)
        colors = ['blue' if label == 0 else 'red' for label in y_pred]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, s=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'kNN Decision Boundary (k = {k})')
        plt.show()


# -------------------------------------------------
# A6 FUNCTIONS – PROJECT DATA (2 FEATURES)
# -------------------------------------------------

def plot_project_features(filepath):
    df = pd.read_csv(filepath)
    X = df[['Ia', 'Va']].values
    y = df['G'].values
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=5)
    plt.xlabel('Ia')
    plt.ylabel('Va')
    plt.title('Project Data Scatter Plot')
    plt.show()


# -------------------------------------------------
# A7 FUNCTIONS – HYPERPARAMETER TUNING
# -------------------------------------------------

def tune_k(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


# -------------------------------------------------
# MAIN PROGRAM (MANDATORY)
# -------------------------------------------------

if __name__ == "__main__":

    filepath = r"C:\Users\gkzee\Downloads\classData.csv"

    # ---------- A1 ----------
    X, y = load_project_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    knn_model = train_knn(X_train, y_train, 3)

    cm_train, p_tr, r_tr, f1_tr = evaluate_classification(knn_model, X_train, y_train)
    cm_test, p_te, r_te, f1_te = evaluate_classification(knn_model, X_test, y_test)

    print("\n--- A1 RESULTS ---")
    print("Training Confusion Matrix:\n", cm_train)
    print("Training Precision:", p_tr)
    print("Training Recall:", r_tr)
    print("Training F1 Score:", f1_tr)

    print("\nTest Confusion Matrix:\n", cm_test)
    print("Test Precision:", p_te)
    print("Test Recall:", r_te)
    print("Test F1 Score:", f1_te)

    # ---------- A3 ----------
    X_train_pts, y_train_pts = generate_training_data()
    plot_training_data(X_train_pts, y_train_pts)

    # ---------- A4 ----------
    X_test_pts = generate_test_data()
    knn_simple = train_knn(X_train_pts, y_train_pts, 3)
    plot_test_classification(knn_simple, X_test_pts)

    # ---------- A5 ----------
    plot_multiple_k(X_train_pts, y_train_pts, X_test_pts, [1, 3, 7, 15])

    # ---------- A6 ----------
    plot_project_features(filepath)

    # ---------- A7 ----------
    best_k, best_score = tune_k(X_train, y_train)
    print("\n--- A7 RESULTS ---")
    print("Best k value:", best_k)
    print("Best Cross-Validation Score:", best_score)
