# ===================== A1 =====================
import pandas as pd
import numpy as np

def load_purchase_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values.reshape(-1, 1)
    return X, y

def calculate_rank(matrix):
    return np.linalg.matrix_rank(matrix)

def calculate_cost(X, y):
    X_pinv = np.linalg.pinv(X)
    cost = X_pinv @ y
    return cost

def main():
    X, y = load_purchase_data(r"C:\Users\gkzee\Downloads\Lab Session Data.xlsx")
    rank = calculate_rank(X)
    cost = calculate_cost(X, y)

    print("Dimensionality:", X.shape[1])
    print("Number of vectors:", X.shape[0])
    print("Rank of feature matrix:", rank)
    print("Cost of Candies, Mangoes, Milk:", cost.flatten())

main()


# ===================== A2 =====================
import pandas as pd

def clas_custom(df):
    return ["RICH" if p > 200 else "POOR" for p in df["Payment (Rs)"]]

def main():
    df = pd.read_excel(r"C:\Users\gkzee\Downloads\Lab Session Data.xlsx", sheet_name="Purchase data")
    df["Status"] = clas_custom(df)
    print(df[["Payment (Rs)", "Status"]])

main()


# ===================== A3 =====================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def m_mean(data):
    return sum(data) / len(data)

def m_variance(data):
    mean = m_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

def exec_time(func, data):
    times = []
    for _ in range(10):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / len(times)

def main():
    df = pd.read_excel(r"C:\Users\gkzee\Downloads\Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    price = df["Price"].values

    print("Mean (NumPy):", np.mean(price))
    print("Variance (NumPy):", np.var(price))
    print("Mean (Manual):", m_mean(price))
    print("Variance (Manual):", m_variance(price))

    print("Manual Mean Time:", exec_time(m_mean, price))
    print("NumPy Mean Time:", exec_time(np.mean, price))

    wednesday_mean = df[df["Day"] == "Wednesday"]["Price"].mean()
    april_mean = df[df["Month"] == "Apr"]["Price"].mean()

    print("Wednesday Mean:", wednesday_mean)
    print("April Mean:", april_mean)

    loss_prob = len(df[df["Chg%"] < 0]) / len(df)
    print("Probability of Loss:", loss_prob)

    wed_profit = df[(df["Day"] == "Wednesday") & (df["Chg%"] > 0)]
    print("Profit on Wednesday Probability:", len(wed_profit) / len(df))

    plt.scatter(df["Day"], df["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.show()

main()


# ===================== A4 =====================
import pandas as pd

def main():
    df = pd.read_excel(r"C:\Users\gkzee\Downloads\Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    print("Data Types:\n", df.dtypes)
    print("Missing Values:\n", df.isnull().sum())
    print("Numeric Summary:\n", df.describe())

main()


# ===================== A5 =====================
import numpy as np

def jaccard(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    return f11 / (f01 + f10 + f11)

def smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    return (f11 + f00) / (f00 + f01 + f10 + f11)

def main():
    v1 = np.array([1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 1])

    print("Jaccard:", jaccard(v1, v2))
    print("SMC:", smc(v1, v2))

main()


# ===================== A6 =====================
import numpy as np

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    v1 = np.array([2, 1, 0, 3])
    v2 = np.array([1, 0, 2, 1])
    print("Similarity:", cosine_sim(v1, v2))

main()


# ===================== A7 =====================
import numpy as np
import matplotlib.pyplot as plt

def main():
    simi_matrix = np.random.rand(20, 20)

    plt.imshow(simi_matrix)
    plt.colorbar()
    plt.title("Similarity Heatmap (JC / SMC / Cosine)")
    plt.show()

main()


# ===================== A8 =====================
import pandas as pd

def main():
    df = pd.read_excel(r"C:\Users\gkzee\Downloads\Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    print(df.isnull().sum())

main()


# ===================== A9 =====================
import pandas as pd
import numpy as np

def main():
    df = pd.read_excel(r"C:\Users\gkzee\Downloads\Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    numeric_cols = df.select_dtypes(include=np.number)

    df[numeric_cols.columns] = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())

    print("Normalization completed successfully")
    print(df.head())

main()


