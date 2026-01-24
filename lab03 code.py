import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski
from sklearn.metrics import confusion_matrix

# =====================================================
# A1: Vector Operations
# =====================================================

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def euclidean_norm(v):
    return (sum(x ** 2 for x in v)) ** 0.5

# =====================================================
# A2: Mean, Std Deviation, Interclass Distance
# =====================================================

def mean_vector(data):
    return np.mean(data, axis=0)

def std_deviation(data):
    return np.std(data, axis=0)

def interclass_distance(m1, m2):
    return euclidean_norm(m1 - m2)

# =====================================================
# A3: Histogram (NON-BLOCKING – FIXED)
# =====================================================

def feature_statistics(feature):
    return np.mean(feature), np.var(feature)

def plot_histogram(feature):
    plt.figure()
    plt.hist(feature, bins=10)
    plt.title("A3: Feature Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show(block=False)
    plt.pause(1.5)
    plt.close()

# =====================================================
# A4: Minkowski Distance (Own)
# =====================================================

def minkowski_distance(v1, v2, p):
    return sum(abs(a - b) ** p for a, b in zip(v1, v2)) ** (1 / p)

# =====================================================
# A5: Compare Minkowski
# =====================================================

def compare_minkowski(v1, v2, p):
    return minkowski_distance(v1, v2, p), minkowski(v1, v2, p)

# =====================================================
# Dataset (2 Classes)
# =====================================================

def load_data():
    np.random.seed(1)
    c1 = np.random.normal(2, 0.5, (30, 2))
    c2 = np.random.normal(5, 0.5, (30, 2))
    X = np.vstack((c1, c2))
    y = np.array([0] * 30 + [1] * 30)
    return X, y

# =====================================================
# A6: Train-Test Split
# =====================================================

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

# =====================================================
# A7: Train kNN
# =====================================================

def train_knn(Xtr, ytr, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Xtr, ytr)
    return model

# =====================================================
# A8: Accuracy
# =====================================================

def get_accuracy(model, Xt, yt):
    return model.score(Xt, yt)

# =====================================================
# A9: Prediction
# =====================================================

def get_predictions(model, Xt):
    return model.predict(Xt)

# =====================================================
# A10: Own kNN
# =====================================================

def own_knn(Xtr, ytr, test, k):
    distances = []
    for i in range(len(Xtr)):
        d = euclidean_norm(Xtr[i] - test)
        distances.append((d, ytr[i]))
    distances.sort()
    labels = [l for _, l in distances[:k]]
    return max(set(labels), key=labels.count)

# =====================================================
# A11: Accuracy vs k
# =====================================================

def accuracy_vs_k(Xtr, Xt, ytr, yt):
    acc = []
    ks = range(1, 12)
    for k in ks:
        m = train_knn(Xtr, ytr, k)
        acc.append(get_accuracy(m, Xt, yt))
    plt.figure()
    plt.plot(ks, acc, marker='o')
    plt.title("A11: Accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show(block=False)
    plt.pause(1.5)
    plt.close()

# =====================================================
# A12: Confusion Matrix
# =====================================================

def get_confusion(yt, yp):
    return confusion_matrix(yt, yp)

# =====================================================
# A13: Metrics
# =====================================================

def metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1

# =====================================================
# A14: Matrix Inversion Classifier
# =====================================================

def matrix_inv_classifier(Xtr, ytr, Xt):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]
    w = np.linalg.pinv(Xb) @ ytr
    Xt_b = np.c_[np.ones(len(Xt)), Xt]
    return np.round(Xt_b @ w).astype(int)

# =====================================================
# MAIN PROGRAM (A1–A14 OUTPUTS)
# =====================================================

def main():
    print("\n========== LAB 03 OUTPUT ==========\n")

    X, y = load_data()

    print("A1:")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print("Dot Product:", dot_product(v1, v2))
    print("Norm:", euclidean_norm(v1), "\n")

    print("A2:")
    m1 = mean_vector(X[y == 0])
    m2 = mean_vector(X[y == 1])
    print("Interclass Distance:", interclass_distance(m1, m2), "\n")

    print("A3:")
    mean, var = feature_statistics(X[:, 0])
    print("Mean:", mean, "Variance:", var)
    plot_histogram(X[:, 0])

    print("A4:")
    for p in range(1, 6):
        print("p =", p, "Distance =", minkowski_distance(v1, v2, p))
    print()

    print("A5:")
    print(compare_minkowski(v1, v2, 3), "\n")

    Xtr, Xt, ytr, yt = split_data(X, y)

    print("A6: Train-Test Split Completed")

    model = train_knn(Xtr, ytr)

    print("A8 Accuracy:", get_accuracy(model, Xt, yt), "\n")

    print("A9 Predictions:", get_predictions(model, Xt)[:5], "\n")

    print("A10 Own kNN Prediction:", own_knn(Xtr, ytr, Xt[0], 3), "\n")

    print("A11:")
    accuracy_vs_k(Xtr, Xt, ytr, yt)

    print("A12:")
    cm = get_confusion(yt, get_predictions(model, Xt))
    print(cm, "\n")

    print("A13:")
    print(metrics(cm), "\n")

    print("A14:")
    print(matrix_inv_classifier(Xtr, ytr, Xt)[:5])

    print("\n========== END ==========")

main()
