import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from cluster import cluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import time


def find_optimal_k_elbow(data: np.array, k_range: tuple = (2, 8), size_population: int = 50, 
                         mutation_rate: float = 0.1, crossover_rate: float = 0.85, 
                         iters: int = 100, save_plot: bool = True) -> int:
    """
    Finds the optimal number of clusters using the elbow method.
    
    Parameters:
    -----------
    data : np.array
        The dataset to cluster
    k_range : tuple
        Range of k values to test (min_k, max_k), default (2, 8)
    size_population : int
        Population size for the genetic algorithm
    mutation_rate : float
        Initial mutation rate for the genetic algorithm
    crossover_rate : float
        Initial crossover rate for the genetic algorithm
    iters : int
        Number of iterations for each k test
    save_plot : bool
        Whether to save the elbow plot to results folder
    
    Returns:
    --------
    int : The optimal number of clusters based on the elbow point
    """
    twcv_scores = []
    k_values = list(range(k_range[0], k_range[1] + 1))
    
    print("\n=== ELBOW METHOD: Testing different k values ===")
    for k in k_values:
        print(f"\n--- Testing k = {k} ---")
        model_temp = cluster(data, n_clusters=k, size_population=size_population, 
                           mutation_rate=mutation_rate, crossover_rate=crossover_rate, iters=iters)
        model_temp.fit()
        best_twcv = model_temp.population[0].twcv
        twcv_scores.append(best_twcv)
        print(f"k = {k}: TWCV = {best_twcv:.2f}")
    
    # Calculate the elbow point using the "knee" method
    # We'll use a simple approach: find the point with maximum distance to the line
    # connecting the first and last points
    def distance_to_line(point_idx, x_vals, y_vals):
        # Line from first to last point
        x1, y1 = x_vals[0], y_vals[0]
        x2, y2 = x_vals[-1], y_vals[-1]
        xp, yp = x_vals[point_idx], y_vals[point_idx]
        
        # Distance from point to line formula
        numerator = abs((y2 - y1) * xp - (x2 - x1) * yp + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if denominator == 0:
            return 0
        return numerator / denominator
    
    # Normalize values for fair comparison
    k_vals_norm = np.array(k_values) / max(k_values)
    twcv_norm = np.array(twcv_scores) / max(twcv_scores)
    
    distances = [distance_to_line(i, k_vals_norm, twcv_norm) for i in range(len(k_values))]
    optimal_idx = np.argmax(distances)
    optimal_k = k_values[optimal_idx]
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, twcv_scores, 'bo-', linewidth=2, markersize=8)
    plt.plot(optimal_k, twcv_scores[optimal_idx], 'ro', markersize=12, 
             label=f'Elbow at k={optimal_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('TWCV', fontsize=12)
    plt.title('Elbow Method: TWCV vs Number of Clusters', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save elbow plot
    if save_plot:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/elbow_method.png", dpi=300, bbox_inches='tight')
        print(f"\nElbow plot saved to: results/elbow_method.png")
    
    plt.show()
    
    print(f"\n=== OPTIMAL k = {optimal_k} (TWCV = {twcv_scores[optimal_idx]:.2f}) ===\n")
    return optimal_k

# Plot of results and fitness convergence
def plot_results_side_by_side(ga_model, original_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

     # Plot 1: convergence
    ax1.plot(ga_model.best_twcv_history, color='blue', linewidth=2)
    ax1.set_title("Convergence")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("TWCV")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: clusters on the original data 
    best_chromosome = ga_model.population[0]
    labels = best_chromosome.chromosome

    points_to_plot = original_data

    if points_to_plot.shape[1] > 2:
        pca = PCA(n_components=2)
        points_plot = pca.fit_transform(points_to_plot)
    else:
        points_plot = points_to_plot

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, lbl in enumerate(unique_labels):
        mask = (labels == lbl)
        ax2.scatter(points_plot[mask, 0], points_plot[mask, 1], 
                    label=f'Cluster {lbl}', alpha=0.7, color=colors[i])

    ax2.set_title("Best Partition")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

