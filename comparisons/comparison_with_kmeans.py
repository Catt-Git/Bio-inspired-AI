import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize,  StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import os


from cluster import cluster
from utils import find_optimal_k_elbow

## -- Definition of gene signature --

PAM50_GROUPS_DICT = {
    "LUMINAL": ["ESR1", "PGR", "BCL2", "FOXA1", "GATA3", "BAG1", "MAPT", "SCUBE2", "MLPH", "CXXC5"],
    "PROLIFERATION": ["MKI67", "CCNB1", "AURKA", "BIRC5", "CCNE1", "CDC20", "CDC6", "CENPF", "CEP55", 
                       "EXO1", "KIF2C", "MELK", "MYBL2", "NDC80", "NUF2", "ORC6", "PTTG1", "RRM2", 
                       "TYMS", "UBE2C", "UBE2T", "ANLN"],
    "BASAL": ["KRT5", "KRT14", "KRT17", "EGFR", "MIA", "SFRP1", "PHGDH"],
    "HER2-ENRICHED": ["ERBB2", "GRB7"],
    "NORMAL-LIKE/OTHER": ["ACTR3B", "BLVRA", "LPL", "TMEM45B", "FGFR4", "MYC", "NAT1", "CDH3", "GPR160", "MMP11", "SLC39A6"]
}

## -- Helper Functions --

def calculate_twcv_external(labels, points, n_clusters):
    """Calculates TWCV using cosine distance on external labels"""
    twcv = 0.0
    for k in range(n_clusters):
        cluster_points = points[labels == k]
        if len(cluster_points) > 0:
            mean_vec = np.mean(cluster_points, axis=0)
            norm = np.linalg.norm(mean_vec)
            centroid = mean_vec / norm if norm > 1e-9 else mean_vec
            similarities = np.dot(cluster_points, centroid)
            distances = 1.0 - similarities
            twcv += np.sum(distances)
    return twcv

def dual_print(text, file_handle=None):
    """Prints on console and on file txt, if specified"""
    print(text)
    if file_handle:
        file_handle.write(text + "\n")

def write_gene_report(algo_name, df_data, labels, file_handle=None):
    """
    Generates gene report
    """
    dual_print("\n" + "="*70, file_handle)
    dual_print(f" GENE REPORT: {algo_name.upper()}", file_handle)
    dual_print("="*70, file_handle)

    for group_name, genes in PAM50_GROUPS_DICT.items():
        dual_print(f"\n🔹 GROUP: {group_name.upper()} (Expected number of genes: {len(genes)})", file_handle)
        
        cluster_counts = {} 
        missing = []
        
        for gene in genes:
            if gene in df_data.index:
                idx = df_data.index.get_loc(gene)
                c_id = labels[idx]
                if c_id not in cluster_counts:
                    cluster_counts[c_id] = []
                cluster_counts[c_id].append(gene)
            else:
                missing.append(gene)
            
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: len(x[1]), reverse=True)
        
        for c_id, found_genes in sorted_clusters:
            percent = (len(found_genes) / len(genes)) * 100
            dual_print(f"   --> CLUSTER {c_id}: contains {len(found_genes)} genes ({percent:.0f}%)", file_handle)
            dual_print(f"       [{', '.join(found_genes)}]", file_handle)
            
        if missing:
             dual_print(f"       (Missing in the dataset: {', '.join(missing)})", file_handle)
        
        dual_print("-" * 70, file_handle)

    dual_print("="*70 + "\n", file_handle)

## -- Data Loading --

filename_expr = "data_mrna_agilent_microarray.txt" 
df = pd.read_csv(filename_expr, sep="\t", index_col=0)

if 'Entrez_Gene_Id' in df.columns:
    df = df.drop(columns=['Entrez_Gene_Id'])
df = df.fillna(0)
df = df.groupby(df.index).mean()

# Gene selection based on variance
gene_variances = df.var(axis=1)
top_genes = gene_variances.sort_values(ascending=False).head(3000).index 
pam50_genes_list = [g for sublist in PAM50_GROUPS_DICT.values() for g in sublist]
available_pam50 = df.index.intersection(pam50_genes_list)
combined_genes = top_genes.union(available_pam50)

x_original_df = df.loc[combined_genes]
x_original = x_original_df.values
# 0. Scaler initialization
scaler = StandardScaler()

# 1. Standardization (Z-Score)
# We transpose because StandardScaler works on columns, but we want to normalize each row
x_scaled = scaler.fit_transform(x_original.T).T

# 2. Normalization
x_train = normalize(x_scaled)

## -- Parameters Setup --
n_clusters = 5  
print(f"Configuration: K={n_clusters}")

metrics_data = {
    "Algorithm": ["Genetic Algorithm", "K-Means"],
    "Time_Seconds": [],
    "Silhouette_Cosine": [],
    "TWCV_Single_Run": [],
    "Stability_Mean_TWCV": [],
    "Stability_Std_TWCV": [],
    "Stability_Best_TWCV": [],
    "Stability_Worst_TWCV": []
}

## -- Single Run --
print("\n--- SINGLE EXECUTION (Performance) ---")

# --- A. Genetic Algorithm ---
print("Running GA...", end=" ", flush=True)
start_ga = time.time()
model_ga = cluster(x_train, n_clusters=n_clusters, size_population=150, 
                   mutation_rate=0.01, crossover_rate=0.85, iters=200)
model_ga.fit()
time_ga = time.time() - start_ga
labels_ga = model_ga.population[0].chromosome
twcv_ga = model_ga.population[0].twcv
sil_ga = silhouette_score(x_train, labels_ga, metric='cosine')
print(f"Done ({time_ga:.2f}s)")

metrics_data["Time_Seconds"].append(time_ga)
metrics_data["Silhouette_Cosine"].append(sil_ga)
metrics_data["TWCV_Single_Run"].append(twcv_ga)

# --- B. K-Means ---
print("Running K-Means...", end=" ", flush=True)
start_km = time.time()
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels_km = kmeans.fit_predict(x_train)
time_km = time.time() - start_km
twcv_km = calculate_twcv_external(labels_km, x_train, n_clusters)
sil_km = silhouette_score(x_train, labels_km, metric='cosine')
print(f"Done ({time_km:.2f}s)")

metrics_data["Time_Seconds"].append(time_km)
metrics_data["Silhouette_Cosine"].append(sil_km)
metrics_data["TWCV_Single_Run"].append(twcv_km)

## -- Stability Test (multiple runs) -- 
n_runs = 5
results_ga_stab = []
results_km_stab = []

# Loop GA
print("GA Loop: ", end="")
for i in range(n_runs):
    print(f".", end="", flush=True)
    m_ga = cluster(x_train, n_clusters=n_clusters, size_population=100, 
                   mutation_rate=0.01, crossover_rate=0.85, iters=100)
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    m_ga.fit()
    sys.stdout = old_stdout
    results_ga_stab.append(m_ga.population[0].twcv)

# Loop K-Means
print(" | KMeans Loop: ", end="")
for i in range(n_runs):
    print(f".", end="", flush=True)
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=300, random_state=i)
    l_km = km.fit_predict(x_train)
    results_km_stab.append(calculate_twcv_external(l_km, x_train, n_clusters))

metrics_data["Stability_Mean_TWCV"].append(np.mean(results_ga_stab))
metrics_data["Stability_Std_TWCV"].append(np.std(results_ga_stab))
metrics_data["Stability_Best_TWCV"].append(np.min(results_ga_stab))
metrics_data["Stability_Worst_TWCV"].append(np.max(results_ga_stab))

metrics_data["Stability_Mean_TWCV"].append(np.mean(results_km_stab))
metrics_data["Stability_Std_TWCV"].append(np.std(results_km_stab))
metrics_data["Stability_Best_TWCV"].append(np.min(results_km_stab))
metrics_data["Stability_Worst_TWCV"].append(np.max(results_km_stab))


# FILE 1: Metric Comparison
df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv("metric_comparison.csv", sep=';', decimal=',', index=False)

# FILE 2: Gene Report
filename_report = "gene_report.txt"

with open(filename_report, "w", encoding="utf-8") as f_report:
    write_gene_report("Genetic Algorithm", x_original_df, labels_ga, file_handle=f_report)
    
    write_gene_report("K-Means Standard", x_original_df, labels_km, file_handle=f_report)

print("\n--- Metrics ---")
print(df_metrics.to_string(index=False))
