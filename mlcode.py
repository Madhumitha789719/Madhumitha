# ============================================================
# MACHINE LEARNING ASSIGNMENT A1–A7 (WITH REQUIRED OUTPUTS)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    path = r"C:\Users\gkzee\Downloads\classData.csv"
    return pd.read_csv(path)

# ============================================================
# A1 – LINEAR REGRESSION USING ONE ATTRIBUTE
# ============================================================

def single_feature_split(data):
    X = data[['Ib']]
    y = data['Va']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    return model.predict(X)

# ============================================================
# A2 – CALCULATE MSE, RMSE, R2
# ============================================================

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

# ============================================================
# A3 – REGRESSION USING MULTIPLE ATTRIBUTES
# ============================================================

def multi_feature_split(data):
    X = data[['Ia','Ib','Ic','Vb','Vc']]
    y = data['Va']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# A4 – APPLY KMEANS CLUSTERING
# ============================================================

def clustering_features(data):
    return data[['Ia','Ib','Ic','Va','Vb','Vc']]

def perform_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(X)
    return model

# ============================================================
# A5 – CLUSTER VALIDATION METRICS
# ============================================================

def clustering_scores(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db

# ============================================================
# A6 – COMPUTE DISTORTION FOR DIFFERENT k
# ============================================================

def compute_distortions(X, k_range):
    distortions = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        model.fit(X)
        distortions.append(model.inertia_)
    return distortions

# ============================================================
# A7 – PLOT ELBOW GRAPH
# ============================================================

def plot_elbow(k_range, distortions):
    plt.figure()
    plt.plot(list(k_range), distortions, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion")
    plt.show()

# ============================================================
# ======================== MAIN ===============================
# ============================================================

data = load_dataset()

# ------------------- A1 -------------------
print("\nA1: Linear Regression using ONE attribute (Ib → Va)")

X_train, X_test, y_train, y_test = single_feature_split(data)
model = train_regression(X_train, y_train)

train_pred = predict(model, X_train)
test_pred = predict(model, X_test)

print("Model Coefficient:", model.coef_)
print("Model Intercept:", model.intercept_)

# ------------------- A2 -------------------
print("\nA2: Regression Error Metrics")

train_metrics = regression_metrics(y_train, train_pred)
test_metrics = regression_metrics(y_test, test_pred)

print("Training -> MSE:", train_metrics[0], "RMSE:", train_metrics[1], "R2:", train_metrics[2])
print("Testing  -> MSE:", test_metrics[0], "RMSE:", test_metrics[1], "R2:", test_metrics[2])

# ------------------- A3 -------------------
print("\nA3: Regression using MULTIPLE attributes")

X_train_m, X_test_m, y_train_m, y_test_m = multi_feature_split(data)
model_m = train_regression(X_train_m, y_train_m)

pred_m = predict(model_m, X_test_m)
metrics_m = regression_metrics(y_test_m, pred_m)

print("Multi-feature Metrics -> MSE:", metrics_m[0], "RMSE:", metrics_m[1], "R2:", metrics_m[2])

# ------------------- A4 -------------------
print("\nA4: K-Means Clustering (k=2)")

X_cluster = clustering_features(data)
kmeans_model = perform_kmeans(X_cluster, 2)

print("Cluster Centers:\n", kmeans_model.cluster_centers_)
print("Cluster Labels (first 20):\n", kmeans_model.labels_[:20])

# ------------------- A5 -------------------
print("\nA5: Clustering Evaluation Scores")

scores = clustering_scores(X_cluster, kmeans_model.labels_)
print("Silhouette Score:", scores[0])
print("Calinski-Harabasz Score:", scores[1])
print("Davies-Bouldin Index:", scores[2])

# ------------------- A6 -------------------
print("\nA6: Distortion Values for Different k")

k_range = range(2, 10)
distortions = compute_distortions(X_cluster, k_range)

for k, d in zip(k_range, distortions):
    print(f"k = {k}, Distortion = {d}")

# ------------------- A7 -------------------
print("\nA7: Elbow Method Graph")
plot_elbow(k_range, distortions)
