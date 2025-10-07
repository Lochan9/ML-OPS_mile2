import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pickle
import os

# ✅ Define directories that work inside the container
DATA_DIR = "/opt/airflow/data"
WORK_DIR = "/opt/airflow/working_data"

DATA_FILE = os.path.join(DATA_DIR, "Iris - all-numbers.csv")
MODEL_FILE = os.path.join(WORK_DIR, "iris_kmeans_model.pkl")
PLOT_FILE = os.path.join(WORK_DIR, "elbow_plot.png")


def load_data():
    """Load the Iris numeric dataset and serialize it."""
    print(f"[INFO] Loading dataset from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"[INFO] Dataset shape: {df.shape}")
    return pickle.dumps(df)


def data_preprocessing(data):
    """Normalize numeric data using MinMaxScaler."""
    df = pickle.loads(data)
    print(f"[INFO] Starting preprocessing... Shape: {df.shape}")

    df = df.dropna()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    print(f"[INFO] Data scaled successfully. Shape: {df_scaled.shape}")

    return pickle.dumps(df_scaled)


def build_save_model(data, filename=MODEL_FILE):
    """Build KMeans models, determine optimal clusters using Elbow method."""
    df_scaled = pickle.loads(data)
    print(f"[INFO] Building KMeans models... Data shape: {df_scaled.shape}")

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
        print(f"[DEBUG] k={k} -> SSE={kmeans.inertia_:.4f}")

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    best_k = kl.elbow or 3
    print(f"[INFO] Optimal number of clusters (elbow): {best_k}")

    best_model = KMeans(n_clusters=best_k, **kmeans_kwargs).fit(df_scaled)

    # ✅ Ensure working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)
    pickle.dump(best_model, open(filename, "wb"))
    print(f"[INFO] Final model saved at: {filename}")

    # ✅ Save elbow plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), sse, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"[INFO] Elbow plot saved at: {PLOT_FILE}")

    return sse


def load_model_elbow(filename=MODEL_FILE, sse=None):
    """Load saved model and show predictions."""
    model = pickle.load(open(filename, "rb"))
    df_test = pd.read_csv(DATA_FILE)
    predictions = model.predict(df_test)

    optimal_k = (
        KneeLocator(range(1, len(sse) + 1), sse, curve="convex", direction="decreasing").elbow
        if sse else model.n_clusters
    )
    print(f"[INFO] Predicted clusters for first 10 rows: {predictions[:10]}")

    return f"Predicted cluster for first row: {predictions[0]} | Optimal clusters: {optimal_k}"


if __name__ == "__main__":
    print("[INFO] Running KMeans pipeline locally...")
    serialized = load_data()
    scaled_data = data_preprocessing(serialized)
    sse_values = build_save_model(scaled_data)
    result = load_model_elbow(MODEL_FILE, sse_values)
    print("[RESULT]", result)
