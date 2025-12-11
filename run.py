import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from cluster import cluster
from sklearn.cluster import KMeans
from utils import find_optimal_k_elbow, plot_results_side_by_side
import time

## -- Data Loading -- 

df = pd.read_csv("response_serum_517.csv", index_col=0)
# Extract expression matrix 
x_original = df.values

## -- Preprocessing (Standard Scaler + Normalization) --
"""
To implement Pearson Correlation as the distance metric:
1. StandardScaler: Centers the data (Mean = 0, Std = 1).
   This is crucial because Pearson(x,y) == Cosine(x_centered, y_centered).
2. Normalize: Scales vectors to unit length (L2 norm = 1).
   This ensures that Euclidean distance on these vectors is proportional to Pearson.
"""
# 0. Scaler initialization
scaler = StandardScaler()

# 1. Standardization (Z-Score)
# We transpose because StandardScaler works on columns, but we want to normalize each row
x_scaled = scaler.fit_transform(x_original.T).T

# 2. Normalization
x_train = normalize(x_scaled)


## -- Model setup --

# Elbow method configuration
# Set use_elbow_method = True to automatically find optimal k
# Set use_elbow_method = False to use a specific k value
use_elbow_method = False
n_clusters = 5  # Used only if use_elbow_method = False

if use_elbow_method:
    n_clusters = find_optimal_k_elbow(x_train, k_range=(2, 7), size_population=100, 
                                      mutation_rate=0.001, crossover_rate=0.85, 
                                      iters=100, save_plot=True)
    print(f"Using optimal k = {n_clusters} from elbow method")
else:
    print(f"Using pre-defined k = {n_clusters}")
model = cluster(x_train, n_clusters=n_clusters, size_population=150, mutation_rate=0.01, crossover_rate=0.85, iters=200)

# Run the model
star_time = time.time()
model.fit()
end_time = time.time()

exe_time = end_time - star_time
print(f"Execution time: {exe_time:.2f} seconds")


## -- Plot Results --
plot_results_side_by_side(model, x_original)


