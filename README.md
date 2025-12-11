# Genetic Algorithm for Gene Expression Clustering

This repository contains a Python implementation of a **Genetic Algorithm (GA)** designed for the unsupervised clustering of high-dimensional transcriptomic data.

The algorithm is specifically optimized to minimize the **Total Within-Cluster Variance (TWCV)** and is benchmarked against the standard **K-Means** algorithm using TCGA (The Cancer Genome Atlas) datasets.

## Project Structure

To keep the repository clean, the core logic is located in the root, while the comparative analysis scripts are organized in a separate folder.

```text
.
├── cluster.py                 # Main class: orchestrates the clustering process
├── genetic.py                 # GA Core: Population, Individual, Crossover, Mutation logic
├── utils.py                   # Helper functions (Elbow method, distance metrics)
├── run.py                     # Script to run a single instance of the GA
├── comparisons/               # Folder for benchmarking and validation
│   └── comparison_with_kmeans.py  # Script comparing GA vs K-Means (Metrics + Reports)
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
