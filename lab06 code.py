# ============================================================
# 22AIE213 - Lab Session 06
# Answers for A1 to A7 in ONE SINGLE PYTHON CODE
# Dataset path given by you:
# "C:\Users\gkzee\OneDrive\Desktop\classData.xlsx"
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

# ============================================================
# DATASET LOADING
# ============================================================
# Your given path
file_path = r"C:\Users\gkzee\OneDrive\Desktop\classData.xlsx"

# Fallback path if running in another system
if not os.path.exists(file_path):
    file_path = "/mnt/data/classData.xlsx"

df = pd.read_excel(file_path)

print("\n================ DATASET PREVIEW ================\n")
print(df.head())
print("\nColumns:", list(df.columns))

# ------------------------------------------------------------
# Target detection:
# If dataset has one-hot encoded class columns like G, C, B, A,
# combine them into a single target column named 'Target'.
# Otherwise, take last column as target.
# ------------------------------------------------------------
possible_target_cols = ["G", "C", "B", "A"]

if all(col in df.columns for col in possible_target_cols):
    # Create a single class label from one-hot encoded columns
    df["Target"] = df[possible_target_cols].idxmax(axis=1)
    feature_cols = [c for c in df.columns if c not in possible_target_cols + ["Target"]]
    target_col = "Target"
else:
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

X_original = df[feature_cols].copy()
y = df[target_col].copy()

print("\nTarget column used:", target_col)
print("Feature columns used:", feature_cols)

# ============================================================
# A1. WRITE A FUNCTION TO CALCULATE ENTROPY
#     If continuous numeric data is present, use equal-width binning
# ============================================================

# ---------- Equal Width Binning Function ----------
def equal_width_binning(series, bins=4):
    """
    Convert continuous numeric data into categorical bins using equal-width binning.
    """
    series = pd.Series(series).dropna()
    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series(["Bin_1"] * len(series), index=series.index)

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    labels = [f"Bin_{i+1}" for i in range(bins)]
    binned = pd.cut(series, bins=bin_edges, labels=labels, include_lowest=True, duplicates="drop")
    return binned

# ---------- Entropy Function ----------
def entropy(values):
    """
    Calculate entropy of a categorical target.
    """
    values = pd.Series(values)
    probs = values.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in probs if p > 0)

# ---------- Convert all continuous features to categorical ----------
def bin_features_equal_width(df_features, bins=4):
    """
    Apply equal-width binning to all numeric continuous features.
    Non-numeric columns are left unchanged.
    """
    out = pd.DataFrame(index=df_features.index)
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            out[col] = equal_width_binning(df_features[col], bins=bins).astype(str)
        else:
            out[col] = df_features[col].astype(str)
    return out

X_binned_equal_width = bin_features_equal_width(X_original, bins=4)

dataset_entropy = entropy(y)

print("\n================ A1. ENTROPY ================\n")
print("Entropy of target =", dataset_entropy)

# ============================================================
# A2. CALCULATE THE GINI INDEX VALUE FOR YOUR DATASET
# ============================================================

def gini_index(values):
    """
    Calculate Gini index of a categorical target.
    """
    values = pd.Series(values)
    probs = values.value_counts(normalize=True)
    return 1 - sum(p ** 2 for p in probs)

dataset_gini = gini_index(y)

print("\n================ A2. GINI INDEX ================\n")
print("Gini index of target =", dataset_gini)

# ============================================================
# A3. WRITE YOUR OWN MODULE FOR DETECTING THE ROOT NODE
#     Use Information Gain as impurity measure
# ============================================================

def information_gain(data, feature, target):
    """
    Calculate information gain of a feature with respect to target.
    Assumes feature is categorical / already binned.
    """
    total_entropy = entropy(data[target])
    feature_values = data[feature].unique()

    weighted_entropy = 0
    for val in feature_values:
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy

def find_root_node(data, features, target):
    """
    Return the feature with maximum information gain.
    """
    gains = {}
    for feature in features:
        gains[feature] = information_gain(data, feature, target)

    root_feature = max(gains, key=gains.get)
    return root_feature, gains

data_for_root = X_binned_equal_width.copy()
data_for_root[target_col] = y

root_feature, all_gains = find_root_node(data_for_root, feature_cols, target_col)

print("\n================ A3. ROOT NODE USING INFORMATION GAIN ================\n")
for f, g in all_gains.items():
    print(f"{f} --> Information Gain = {g}")
print("\nRoot node =", root_feature)

# ============================================================
# A4. GENERAL BINNING FUNCTION
#     equal-width or equal-frequency, parameters with defaults
# ============================================================

def binning(series, bins=4, method="equal_width"):
    """
    General binning function.
    method = 'equal_width' or 'equal_frequency'
    """
    s = pd.Series(series).dropna()

    if s.nunique() == 1:
        return pd.Series(["Bin_1"] * len(s), index=s.index)

    labels = [f"Bin_{i+1}" for i in range(bins)]

    if method == "equal_width":
        edges = np.linspace(s.min(), s.max(), bins + 1)
        return pd.cut(s, bins=edges, labels=labels, include_lowest=True, duplicates="drop")

    elif method == "equal_frequency":
        return pd.qcut(s, q=bins, labels=labels, duplicates="drop")

    else:
        raise ValueError("method must be 'equal_width' or 'equal_frequency'")

def bin_dataset(df_features, bins=4, method="equal_width"):
    """
    Apply chosen binning to all numeric columns in dataset.
    """
    out = pd.DataFrame(index=df_features.index)
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            out[col] = binning(df_features[col], bins=bins, method=method).astype(str)
        else:
            out[col] = df_features[col].astype(str)
    return out

# Example use of A4:
X_binned = bin_dataset(X_original, bins=4, method="equal_width")

print("\n================ A4. BINNING DONE ================\n")
print("Binning type used: equal_width")
print(X_binned.head())

# ============================================================
# A5. EXPAND ABOVE FUNCTIONS TO BUILD YOUR OWN DECISION TREE MODULE
# ============================================================

class MyDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        data = X.copy()
        data["Target"] = y.values
        self.tree = self._build_tree(data, X.columns.tolist(), "Target", depth=0)

    def _majority_class(self, target_values):
        return Counter(target_values).most_common(1)[0][0]

    def _build_tree(self, data, features, target, depth):
        # Stop conditions
        if len(data[target].unique()) == 1:
            return data[target].iloc[0]

        if len(features) == 0:
            return self._majority_class(data[target])

        if self.max_depth is not None and depth >= self.max_depth:
            return self._majority_class(data[target])

        if len(data) < self.min_samples_split:
            return self._majority_class(data[target])

        best_feature, gains = find_root_node(data, features, target)

        if gains[best_feature] <= 0:
            return self._majority_class(data[target])

        tree = {best_feature: {}}

        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]

            if subset.empty:
                tree[best_feature][value] = self._majority_class(data[target])
            else:
                remaining_features = [f for f in features if f != best_feature]
                subtree = self._build_tree(subset, remaining_features, target, depth + 1)
                tree[best_feature][value] = subtree

        return tree

    def _predict_one(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        root = next(iter(tree))
        value = row.get(root)

        if value in tree[root]:
            return self._predict_one(row, tree[root][value])
        else:
            # unseen value handling
            branches = tree[root]
            leaf_values = []

            def collect_leaves(subtree):
                if isinstance(subtree, dict):
                    for k in subtree:
                        collect_leaves(subtree[k])
                else:
                    leaf_values.append(subtree)

            for branch in branches.values():
                collect_leaves(branch)

            return Counter(leaf_values).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for _, row in X.iterrows()])

# Build our own tree on binned dataset
my_tree = MyDecisionTree(max_depth=4)
my_tree.fit(X_binned, y)

print("\n================ A5. MY OWN DECISION TREE ================\n")
print("Constructed Tree:")
print(my_tree.tree)

# ============================================================
# A6. DRAW AND VISUALIZE THE DECISION TREE CONSTRUCTED
# ============================================================
# For proper graphical visualization, using sklearn DecisionTreeClassifier

# Convert target labels into numeric if needed
y_encoded = pd.factorize(y)[0]

# Use binned categorical data encoded as category codes
X_encoded = X_binned.copy()
for col in X_encoded.columns:
    X_encoded[col] = pd.factorize(X_encoded[col])[0]

clf_full = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=4)
clf_full.fit(X_encoded, y_encoded)

plt.figure(figsize=(18, 10))
plot_tree(
    clf_full,
    feature_names=X_encoded.columns,
    class_names=[str(c) for c in pd.Series(y).unique()],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("A6. Decision Tree Visualization")
plt.show()

# ============================================================
# A7. USE 2 FEATURES FOR CLASSIFICATION AND VISUALIZE DECISION BOUNDARY
# ============================================================
# Use top 2 features based on information gain

sorted_features = sorted(all_gains.items(), key=lambda x: x[1], reverse=True)
top2_features = [sorted_features[0][0], sorted_features[1][0]]

print("\n================ A7. DECISION BOUNDARY ================\n")
print("Top 2 features used for decision boundary:", top2_features)

X2 = X_original[top2_features].copy()

# If target is string labels, encode target
y2 = pd.factorize(y)[0]

clf_2d = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=4)
clf_2d.fit(X2, y2)

plt.figure(figsize=(10, 7))
DecisionBoundaryDisplay.from_estimator(
    clf_2d,
    X2,
    response_method="predict",
    alpha=0.4
)

scatter = plt.scatter(
    X2.iloc[:, 0],
    X2.iloc[:, 1],
    c=y2,
    edgecolor="k"
)

plt.xlabel(top2_features[0])
plt.ylabel(top2_features[1])
plt.title("A7. Decision Boundary using Top 2 Features")
plt.show()

# ============================================================
# EXTRA: ACCURACY OF CUSTOM TREE (OPTIONAL CHECK)
# ============================================================

custom_pred = my_tree.predict(X_binned)
custom_acc = np.mean(custom_pred == y.values)

print("\n================ OPTIONAL RESULT ================\n")
print("Custom Decision Tree Accuracy on same dataset =", custom_acc)

# ============================================================
# END OF ALL A1 TO A7
# ============================================================
