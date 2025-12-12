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
```

## Key Features
* **Global Optimization:** Unlike K-Means, which can get stuck in local minima, the GA explores the solution space globally using evolutionary operators.
* **Custom Fitness Function:** Optimized for TWCV (Total Within Cluster Variance) to maximize cluster compactness.
* **Bioinformatics Validated:** Tested on real cancer datasets (BRCA, CRC, GBM, LUAD) with ground-truth biological subtypes (PAM50, CMS, Verhaak, etc.).
* **Stability Analysis:** Includes tools to measure the stability of the clustering over multiple runs.

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Due to GitHub's file size limits, the gene expression datasets used in this project are hosted externally. All datasets are derived from **The Cancer Genome Atlas (TCGA)** and have been pre-processed to retain the **top 3000 genes** with the highest variance.

### Download Links
Please download the datasets from the following links and place them in the `data/` directory of this repository.

| Dataset | Cancer Type | Signatures | Size | Download |
| :--- | :--- | :--- | :--- | :--- |
| **BRCA** | Breast Invasive Carcinoma | PAM50 | ~X MB | [https://portal.gdc.cancer.gov/projects/TCGA-BRCA](#) |
| **CRC** | Colorectal Cancer | CMS | ~X MB | [https://portal.gdc.cancer.gov/projects/TCGA-COAD](#) |
| **GBM** | Glioblastoma Multiforme | Verhaak | ~X MB | [https://portal.gdc.cancer.gov/projects/TCGA-GBM](#) |
| **LUAD** | Lung Adenocarcinoma | Wilkerson | ~X MB | [https://www.cancerimagingarchive.net/collection/tcga-lusc/](#) |

> **Note:** Replace the `[Download Here](#)` links above with your actual hosting URLs (e.g., Google Drive, Dropbox, OneDrive).

### Directory Structure
To ensure the scripts run correctly, your project folder should look like this after downloading:

```text
project-root/
├── data/                  <-- Create this folder
│   ├── BRCA.csv      <-- Place downloaded files here
│   ├── CRC.csv
│   ├── GBM.csv
│   └── LUAD.csv
├── comparison_with_kmeans.py
├── main_clustering.py
├── README.md
└── ...
```

## Usage

### 1. Running the Genetic Algorithm (Single Run)
To run the GA on a dataset and obtain the cluster labels:

1.  Place your dataset file (e.g., `data_mrna_seq.txt`) in the root directory.
2.  Edit `run.py` to point to your dataset filename.
3.  Run the script:

```bash
python run.py
```

### 2. Comparative Analysis (GA vs K-Means)
The script in the `comparisons/` folder performs a full benchmark:
* Runs both GA and K-Means.
* Calculates metrics: **Silhouette Score**, **TWCV**, **Time**.
* Generates detailed **text reports** mapping clusters to biological groups.
* Produces **Heatmaps** and **Confusion Matrices**.

**How to run it:**
Since the script is in a subfolder, it automatically handles imports from the parent directory (provided the system path fix is applied). Run it as follows:

```bash
python comparisons/comparison_with_kmeans.py
```
**Outputs**
The comparison scripts generates the following files:
* metrics_comparison.csv: table comparing time, silhouette and TWCV.
* gene_report.txt: text file showing exactly which genes ended up in which clusters.
