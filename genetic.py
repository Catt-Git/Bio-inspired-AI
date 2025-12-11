import numpy as np
import random

"""
This code defines the class GeneticClustering. Each instance represents a single individual of the population.

__init__(self, chromosome, points, n_clusters)
Function to initialize the single individual of the population. 
It calculates the centroids of the initial clusters, the twcv and checks if the solution is valid (i.e., has no empty clusters)

calculate_centroids(self)
Function to calculate the geometric mean of the points assigned to each cluster.

calculate_legality(self)
Function to measure how valid the solution is. In particular, a solution is considered valid if none of the k clusters are empty.
Returns the ratio non_empty_clusters/total_clusters, if is equal to 1 the individual is fully legal.
This value is used in the fitness function to heavily penalize solutions with empty clusters.

calculate_twcv(self)
Function to calculate the Total Within-Cluster Variation, the cost function we want to minimize.
TWCV = Σ_k Σ_{x ∈ C_k} ||x - c_k||²

mutation(self, mutation_probability)
Function to perform a distance based mutation on the chromosome. 
It uses a probabilistic approach that:
- Iterates through every point;
- Based on probability mutation_probability, a point is selected for mutation. The function doesn't pick a new cluster uniformly at random;
- Calculates the distance from the point to all existing centroids;
- Builds a probability distribution for the assignment of the point to a cluster where:
  1. If the distance to cluster k is small, probability is high.
  2. If the distance is large, probability is low.
- Mutation nudges points toward clusters that are geometrically closer, avoiding chaotic jumps to far-away clusters.

crossover(self, other, alpha, noise_level)
Function to perform a crossover operation that works on centroids rather than labels.
1. Matching: it pairs the centroids of parent A with the closest centroids of parent B (greedy matching);
2. Blending: it creates new centroids by averaging the paired parents: C_new=alpha*C1 + (1-alpha)*C2.
   The new centroid is a weighted average of the two, plus optional Gaussian noise;
3. Assignment: it reassigns all points to the nearest newly created centroid.

assign_to_nearest_centroids_list(self, centroid_list)
A helper function used to convert a list of coordinates (centroids) back into a list of labels (chromosome).
It calculates the distance from every point to every centroid in the list and assigns the point to the index of the nearest one.
"""

class GeneticClustering:
    def __init__(self, chromosome: np.array, points: np.array, n_clusters: int):

        """
        Initializes a clustering solution and immediately calculates quality metrics (phenotype) based on point assignments (genotype).

        Args:
        chromosome: 1D array where chromosome[i] is the cluster assignment for point i
        points: data matrix (n_samples, n_features)
        n_clusters: expected number of clusters (k)
        """

        self.chromosome = np.array(chromosome)
        self.points = points
        self.n_clusters = n_clusters

        # Compute initial properties
        self.centroids = self.calculate_centroids()     
        self.twcv = self.calculate_twcv()               
        self.legality_ratio = self.calculate_legality() 
        self.fitness_scores = 0.0
    
    def calculate_centroids(self):

        """
        Calculates the geometric coordinates of the centroids and normalizes them.
        Handles empty clusters by assignign them 'None'

        Returns:
        centroids: a List of numpy arrays (or None) representing cluster centers
        """

        centroids = []
        for k in range(self.n_clusters):
            cluster_points = self.points[self.chromosome == k]
            if len(cluster_points) > 0:
                mean_vec = np.mean(cluster_points, axis=0)
                # --- COSINE FIX: Normalize Centroid ---
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-9:
                    centroids.append(mean_vec / norm)
                else:
                    centroids.append(mean_vec)
            else:
                centroids.append(None)
        return centroids
    
    def calculate_legality(self):

        """
        Returns ratio of non-empty clusters.
        Used to penalize degenerate solutions that collapse into fewer than k clusters.
        """

        non_empty_clusters = len([c for c in self.centroids if c is not None])
        return non_empty_clusters / self.n_clusters
    
    def calculate_twcv(self):

        """
        Calculates Total Within-Cluster Variation using cosine distance.
        Dist = 1 - Similarity
        """
        
        twcv = 0.0
        for k in range(self.n_clusters):
            centroid = self.centroids[k]
            if centroid is not None:
                cluster_points = self.points[self.chromosome == k]
                # Similarity = dot product
                similarities = np.dot(cluster_points, centroid)
                # Distance = 1 - similarity
                distances = 1.0 - similarities
                twcv += np.sum(distances)
        return twcv
    
    def mutation(self, mutation_probability):

        """
        Applies a distance-based probabilistic mutation.
        Instead of reassigning a point randomly, it favors moving the point  toward clusters whose centroids are spatially closer.
        """

        new_chromosome = self.chromosome.copy()
        n_samples = len(self.points)

        # The mutation happens with probability mutation_probability, independently for each allele
        # Here the random check is vectorized
        random_vals = np.random.random(n_samples)
        mutate_indices = np.where(random_vals < mutation_probability)[0]

        for i in mutate_indices:
            point = self.points[i]
            d_vals = []
            
            # Calculates distance from all the current centroids
            for k in range(self.n_clusters):
                centroid = self.centroids[k]
                if centroid is None:
                    # If the cluster is empty, distance = max distance (1.0 for cosine similarity)
                    d_vals.append(1.0)
                else:
                    sim = np.dot(point, centroid)
                    dist = 1.0 - sim
                    d_vals.append(dist)
            
            d_vals = np.array(d_vals)
            d_max = np.max(d_vals)
            # Invert distance: closer clusters get higher probability
            numerator = 1.5 * d_max - d_vals + 1e-6 
            # Clip negative values to avoid crashes
            numerator = np.maximum(numerator, 0.0)
            
            total = np.sum(numerator)
            if total == 0:
                prob_distribution = np.ones(self.n_clusters) / self.n_clusters
            else:
                # Normalization to obtain the probability
                prob_distribution = numerator / total

            # Choice of new cluster for point i 
            new_cluster = np.random.choice(self.n_clusters, p=prob_distribution)
            new_chromosome[i] = new_cluster
            
        return GeneticClustering(new_chromosome, self.points, self.n_clusters)
    
    def crossover(self, other, crossover_probability, alpha, noise_level):
        """
        Performs a geometric crossover.
        1. Aligns centroids from both parents by minimizing distance (Greedy Matching).
        2. Interpolates aligned centroids to create new 'hyper-centroids'.
        3. Adds optional noise for exploration.
        4. Reassigns points to the newly generated centroids.
        """

        if random.random() < crossover_probability:
            # Case A: crossover occurs
            dim = self.points.shape[1]
            c1 = [c if c is not None else np.zeros(dim) for c in self.centroids]
            c2 = [c if c is not None else np.zeros(dim) for c in other.centroids]

            # Greedy matching
            c1_indices = list(range(self.n_clusters))
            c2_indices = list(range(self.n_clusters))
            matched_pairs = []

            # Calculate similarity matrix between all the centroids
            mat_c1 = np.array(c1)
            mat_c2 = np.array(c2)
    
            sim_matrix = np.dot(mat_c1, mat_c2.T)
            # Compute distance matrix 
            dist_matrix = 1.0 - sim_matrix
            
            while len(c1_indices) > 0 and len(c2_indices) > 0:
                min_dist = np.inf
                best_pair = (-1, -1)
                
                # Manual search on remaining indices
                for i in c1_indices:
                    for j in c2_indices:
                        if dist_matrix[i, j] < min_dist:
                            min_dist = dist_matrix[i, j]
                            best_pair = (i, j)
                            
                matched_pairs.append(best_pair)
                c1_indices.remove(best_pair[0])
                c2_indices.remove(best_pair[1])
            
            # Interpolation and noise
            new_centroids = [None] * self.n_clusters
            for idx1, idx2 in matched_pairs:
                base_c = alpha * c1[idx1] + (1 - alpha) * c2[idx2]
                
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, size=base_c.shape)
                    base_c += noise
                
                # Re-Normalize Child Centroid
                # Mixing two vectors changes length, must re-normalize for Cosine
                norm = np.linalg.norm(base_c)
                if norm > 1e-9:
                    base_c = base_c / norm
                    
                new_centroids[idx1] = base_c 
            
            return self.assign_to_nearest_centroids_list(new_centroids)
        else:
            # Case B: crossover doesn't occur
            # Returns an exact copy of current parent
            return GeneticClustering(self.chromosome.copy(), self.points, self.n_clusters)
        
    def assign_to_nearest_centroids_list(self, centroid_list):

        """
        Converts a list of geometric centroids into a new GeneticClustering object (chromosome).
        Assigns every point in the dataset to its nearest centroid based on max cosine similarity.
        """

        centroids_matrix = np.array(centroid_list)
        
        # Safety normalization for incoming centroids
        norms = np.linalg.norm(centroids_matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        centroids_matrix = centroids_matrix / norms

        # Similarity = dot product
        # points (N x F) . centroids.T (F x K) = (N x K)
        similarity = np.dot(self.points, centroids_matrix.T)

        # Assigns each point to the index of the nearest centroid
        # Argmax similarity = Argmin distance
        new_labels = np.argmax(similarity, axis=1)

        return GeneticClustering(new_labels, self.points, self.n_clusters)