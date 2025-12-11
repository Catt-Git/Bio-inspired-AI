import numpy as np
import random
import matplotlib.pyplot as plt
from genetic import GeneticClustering 

"""
This code defines the class cluster, which manages the population and the evolutionary process.

__init__(self, X, n_clusters, size_population, mutation_rate, crossover_rate, iters)
Function to st up the simulation parameters: population size, mutation rate, crossover rate, number of iterations and the dataset X.
It inizializes the empty population list and history tracker.

calculate_population_fitness(self)
Function to determine how good every individual in the population is.
The problem goal is to minimize the TWCV, but GAsusually work by maximizing fitness, this function inverts the problem.
1. Normalization: converts error to score using 1.5 * max_error - individual_error;
2. Penalizes illegal strings (solutions with empty clusters):
    - Finds the worst fitness among valid solutions (f_min)
    - Illegal solutions get a fraction of that worst fitness (Ratio * f_min)

fit(self)
Main execution loop of the Genetic Algorithm.
1. Initialization: creates random starting chromosomes;
2. Dynamic Decay of parameters:
    - Early generations: high mutation, high noise and high crossover rate for exploration;
    - Late generations: low mutation, low/zero noise and low crossover rate for exploitation and refinement.
3. Elitism: explicitily saves the absolute best individual;
4. Selection: uses Roulette Wheel selection to pick parents;
5. Breeding: creates the next generation by crossover and mutation.

create_random_chromosome(self)
Helper function to generate a random list of integers between 0 and n_clusters as a starting point for the very first generation.
Each integer represents the cluster assignment for a specific data point

show_plot(self)
Function to show the decreasing trend of the best TWCV at each generation.
"""

class cluster:
    def __init__(self, X: np.array, n_clusters: int, size_population: int, mutation_rate: float, crossover_rate: float, iters: int):

        """
        Initializes the Genetic Algorithm orchestrator.
        
        Args:
            X: dataset (n_samples, n_features)
            n_clusters: target number of clusters (k)
            size_population: number of individuals in each generation
            mutation_rate: initial probability of mutation (decays over time)
            crossover_rate: initial probability of crossover (decays over time)
            iters: total number of generations to run
        """

        self.X = X
        self.n_clusters = n_clusters
        self.size_population = size_population
        self.mutation_rate = mutation_rate 
        self.crossover_rate = crossover_rate
        self.iters = iters
        
        self.n_samples = self.X.shape[0]
        self.population = []
        self.best_twcv_history = []

    def calculate_population_fitness(self):

        """
        Evaluates the fitness of the entire population.
 
        1. Converts the minimization problem (minimize TWCV) into maximization
        2. Penalizes 'illegal' individuals by assigning them a fitness strictly lower than the worst 'legal' individual
        """
        
        # Find the maximum TWCV in the current population
        twcv_values = [ind.twcv for ind in self.population]
        twcv_max = max(twcv_values)

        # Find legal strings (individual with no empty clusters)
        legal_individuals = [ind for ind in self.population if ind.legality_ratio == 1.0]

        # Calculate F_min (minimum fitness of legal strings)
        # If there are no legal strings, default F_min = 1
        f_min = 1.0

        # Preliminary fitness calculation for legal strings to find F_min
        if legal_individuals:
            legal_fitnesses = [(1.5 * twcv_max - ind.twcv) for ind in legal_individuals]
            f_min = min(legal_fitnesses)
        
        # Assign final fitness
        for ind in self.population:
            if ind.legality_ratio == 1.0:
                # Legal string formula: reward low error
                ind.fitness_scores = 1.5 * twcv_max - ind.twcv
            else:
                # Illegal string formula: penalizes heavily based on emptiness
                #  Fitness =  e(Sz) * F_min
                ind.fitness_scores = ind.legality_ratio * f_min
        
    def fit(self):
        #Initialization
        self.population = []
        for _ in range(self.size_population):
            chrom = self.create_random_chromosome()
            self.population.append(GeneticClustering(chrom, self.X, self.n_clusters))

        self.counter = 0

        while self.counter  < self.iters:
            # -- Dynamic parameter calculation --
            # Progress ranges from 0.0 (start) to (1.0) finish
            progress = self.counter / self.iters

            # 1. Mutation rate
            # Goes to 0.001
            current_mutation = self.mutation_rate * (1-progress) + 0.001

            # 2. Crossover rate
            # Goes to 0.65
            current_crossover = self.crossover_rate - (progress * (self.crossover_rate - 0.65))

            # 3. Crossover noise
            current_noise = 0.5 * (1- progress)

            # -- Fitness evaluation --
            self.calculate_population_fitness()
            # Sort population by fitness (descending)
            self.population = sorted(self.population, key=lambda x:  x.fitness_scores, reverse=True)

            best_ind = self.population[0]
            self.best_twcv_history.append(best_ind.twcv)

            # Print parameters
            print(f"Gen {self.counter}: Best TWCV {best_ind.twcv:.2f} | Mut: {current_mutation:.4f} | Cross: {current_crossover:.2f}")

            # -- Elitism --
            next_generation = [self.population[0]] # Keep only the absolute best

            # -- Selection --
            fitness_sum = sum(ind.fitness_scores for ind in self.population)
            probs = [ind.fitness_scores / fitness_sum for ind in self.population] if fitness_sum > 0 else None

            # -- Generation --
            while len(next_generation) < self.size_population:
                p1, p2 = np.random.choice(self.population, size=2, p=probs)

                # A. Crossover with dynamic noise and adaptive probability
                alpha_dyn = random.uniform(0.2, 0.8)
                child = p1.crossover(p2, current_crossover, alpha=alpha_dyn, noise_level=current_noise)

                # B. Mutation
                child = child.mutation(current_mutation)

                next_generation.append(child)
            
            self.population =next_generation
            self.counter += 1
    
    def create_random_chromosome(self):

        """
        Generates a random starting solution (genotype).
        Returns an array of integers [0, n_clusters-1] of length n_samples.
        """

        return np.random.randint(0, self.n_clusters, size=self.n_samples)

    def show_plot(self):

        """
        Visualizes the convergence history.
        Plots the best TWCV (Total Within-Cluster Variation) value found at each generation.
        """
        
        plt.plot(self.best_twcv_history)
        plt.title("Convergence (TWCV minimization)")
        plt.xlabel("Generation")
        plt.ylabel("TWCV Error")
        plt.show()
        

