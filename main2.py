import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Parameters for Genetic Algorithm
population_size = 100  # Number of individuals in the population
generations = 500  # Number of generations to evolve
mutation_rate = 0.1  # Probability of mutation
elitism_count = 2  # Number of best individuals to carry forward
truck_capacity = 271 
demands = [0, 55, 90, 87, 120, 80, 100, 96, 78, 105, 135]
# Data jarak dari tabel (dalam km)
file_path = "data_route.csv"
data = pd.read_csv(file_path, header=None)  # header=None karena file hanya berisi angka

# Mengonversi DataFrame menjadi matriks numpy
distance_matrix = data.values
num_locations = distance_matrix.shape[0]

# Initialize population
def initialize_population(size, num_locations):
    population = []
    for _ in range(size):
        route = list(range(1, num_locations))  # Exclude depot (index 0)
        random.shuffle(route)
        route = [0] + route + [0]  # Start and end at the depot
        population.append(route)
    return population

# Calculate route distance
def calculate_route_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    return distance

# Evaluate population fitness
def evaluate_population(population, distance_matrix):
    fitness = []
    for route in population:
        distance = calculate_route_distance(route, distance_matrix)
        fitness.append(1 / distance)  # Fitness is inversely proportional to distance
    return fitness

# Select parents using tournament selection
def select_parents(population, fitness, tournament_size=5):
    parents = []
    for _ in range(2):  # Select two parents
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])  # Higher fitness wins
        parents.append(winner[0])
    return parents

# Order Crossover
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
    offspring = [-1] * len(parent1)
    offspring[start:end] = parent1[start:end]
    
    pos = end
    for gene in parent2[1:-1]:
        if gene not in offspring:
            if pos >= len(parent1) - 1:
                pos = 1
            offspring[pos] = gene
            pos += 1
    offspring[0], offspring[-1] = 0, 0  # Ensure depot at start and end
    return offspring

# Mutate a route
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route) - 1), 2)  # Exclude depot
        route[i], route[j] = route[j], route[i]
    return route
# Calculate route distance with demand constraints and truck capacity, returning to DC if capacity is exceeded
def calculate_route_distance_with_reloading(route, distance_matrix, demands, truck_capacity):
    total_distance = 0
    current_load = 0

    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]

        # Add demand for the next node (if not the depot)
        if current_node != 0:
            current_load += demands[current_node]

        # Check if truck capacity is exceeded
        if current_load > truck_capacity:
            # Return to DC to reload
            total_distance += distance_matrix[current_node][0]  # Back to DC
            total_distance += distance_matrix[0][next_node]  # From DC to next node
            current_load = demands[current_node]  # Reset load to current node's demand
        else:
            total_distance += distance_matrix[current_node][next_node]

    return total_distance

# Genetic Algorithm adjusted for reloading at DC
def genetic_algorithm_with_reloading(distance_matrix, demands, truck_capacity, population_size, generations, mutation_rate, elitism_count):
    population = initialize_population(population_size, num_locations)
    
    print(f"=== Populasi Awal ===")
    for i, ind in enumerate(population):
        print(f"Individu {i+1}: {ind}")

    for generation in range(generations):
        print(f"\n=== Generasi {generation + 1} ===")
        
        # Evaluate fitness
        fitness = [1 / calculate_route_distance_with_reloading(ind, distance_matrix, demands, truck_capacity) for ind in population]
        fitness_table = pd.DataFrame({"Individu": population, "Fitness": fitness})
        print("\nTabel Fitness:")
        print(fitness_table)
        
        # Elitism
        next_population = []
        elite_indices = sorted(range(len(fitness)), key=lambda x: fitness[x], reverse=True)[:elitism_count]
        for idx in elite_indices:
            next_population.append(population[idx])

        # Crossover
        print("\nProses Crossover:")
        while len(next_population) < population_size:
            parents = select_parents(population, fitness)
            offspring = order_crossover(parents[0], parents[1])
            print(f"Parents: {parents[0]}, {parents[1]} => Offspring: {offspring}")
            next_population.append(offspring)

        # Mutation
        print("\nProses Mutasi:")
        for i in range(len(next_population)):
            original = next_population[i]
            next_population[i] = mutate(next_population[i], mutation_rate)
            print(f"Original: {original} => Mutasi: {next_population[i]}")

        # Update population
        population = next_population
        print("\nPopulasi Baru:")
        for i, ind in enumerate(population):
            route_distance = calculate_route_distance_with_reloading(ind, distance_matrix, demands, truck_capacity)
            print(f"Individu {i+1}: {ind}, Jarak: {route_distance:.2f} km")
    
    # Return the best route and distance
    final_fitness = [1 / calculate_route_distance_with_reloading(ind, distance_matrix, demands, truck_capacity) for ind in population]
    best_index = final_fitness.index(max(final_fitness))
    best_route = population[best_index]
    best_distance = calculate_route_distance_with_reloading(best_route, distance_matrix, demands, truck_capacity)
    return best_route, best_distance

# Run the adjusted genetic algorithm for the new scenario
best_route, best_distance = genetic_algorithm_with_reloading(
    distance_matrix,
    demands,
    truck_capacity,
    population_size,
    generations,
    mutation_rate,
    elitism_count
)

# Print the final result
print("\n=== Hasil Akhir ===")
print(f"Rute Terbaik     : {best_route}")
print(f"Jarak Total (km) : {best_distance:.2f}")

