import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Genetic Algorithm
population_size = 100 
generations = 500  
mutation_rate = 0.1 
elitism_count = 2  
truck_capacity = 271  

# Load distance matrix and demands
file_path = "data_route.csv"
distance_matrix = pd.read_csv(file_path, header=None).values
num_locations = distance_matrix.shape[0]

# Example demands for each location
demands = [0, 55, 90, 87, 120, 80, 100, 96, 78, 105, 135]  

# Initialize population
def initialize_population(size, num_locations):
    population = []
    for _ in range(size):
        route = list(range(1, num_locations))  
        random.shuffle(route)
        route = [0] + route + [0]  
        population.append(route)
    return population

# Calculate route distance and update route, ensuring capacity constraints are respected
def calculate_route_distance(route, distance_matrix, demands, truck_capacity):
    total_distance = 0
    current_load = 0
    updated_route = [route[0]]  

    for i in range(1, len(route)):  # Iterate over the route
        current_node = route[i - 1]
        next_node = route[i]

        # Check the demand of the next node (if it's not the depot)
        if next_node != 0:
            current_load += demands[next_node]

        # If truck capacity is exceeded
        if current_load > truck_capacity:
            # Return to DC to reload
            total_distance += distance_matrix[current_node][0]  
            updated_route.append(0)  
            current_load = demands[next_node]  

            # Start fresh from DC to the next node
            total_distance += distance_matrix[0][next_node]
            updated_route.append(next_node)
        else:
            # Continue to the next node
            total_distance += distance_matrix[current_node][next_node]
            updated_route.append(next_node)

    return total_distance, updated_route


# Evaluate population fitness
def evaluate_population(population, distance_matrix, demands, truck_capacity):
    fitness = []
    updated_routes = []
    for route in population:
        distance, updated_route = calculate_route_distance(
            route, distance_matrix, demands, truck_capacity
        )
        fitness.append(1 / distance) 
        updated_routes.append(updated_route)
    return fitness, updated_routes

# Select parents using tournament selection
def select_parents(population, fitness, tournament_size=5):
    parents = []
    for _ in range(2):  
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        winner = max(tournament, key=lambda x: x[1]) 
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
    offspring[0], offspring[-1] = 0, 0  
    return offspring

# Mutate a route
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route) - 1), 2) 
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(distance_matrix, demands, truck_capacity, population_size, generations, mutation_rate, elitism_count):
    population = initialize_population(population_size, num_locations)
    print(f"=== Populasi Awal ===")
    for i, ind in enumerate(population):
        print(f"Individu {i+1}: {ind}")

    for generation in range(generations):
        print(f"\n=== Generasi {generation + 1} ===")
        
        # Evaluate fitness and updated routes
        fitness, updated_routes = evaluate_population(population, distance_matrix, demands, truck_capacity)
        fitness_table = pd.DataFrame({"Individu": updated_routes, "Fitness": fitness})
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

        # Mutasi
        print("\nProses Mutasi:")
        for i in range(len(next_population)):
            original = next_population[i]
            next_population[i] = mutate(next_population[i], mutation_rate)
            print(f"Original: {original} => Mutasi: {next_population[i]}")

        # Populasi Baru
        population = next_population
        print("\nPopulasi Baru:")
        for i, ind in enumerate(population):
            distance, updated_route = calculate_route_distance(
                ind, distance_matrix, demands, truck_capacity
            )
            print(f"Individu {i+1}: {updated_route}, Jarak: {distance:.2f} km" )
    
    # Return the best route
    final_fitness, final_routes = evaluate_population(population, distance_matrix, demands, truck_capacity)
    best_index = final_fitness.index(max(final_fitness))
    best_route = final_routes[best_index]
    best_distance = calculate_route_distance(
        population[best_index], distance_matrix, demands, truck_capacity
    )[0]
    return best_route, best_distance

    
coordinates = [
    (0, 0), (2, 3), (4, 1), (6, 5), (3, 4), (5, 2), (1, 5),
    (7, 6), (8, 3), (9, 5), (10, 2)
]

def visualize_route(route, coordinates):
    x = [coordinates[loc][0] for loc in route]
    y = [coordinates[loc][1] for loc in route]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, '-o', label='Route')
    plt.title('Optimal Logistics Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    for i, loc in enumerate(route):
        plt.text(x[i], y[i], str(loc), fontsize=12, ha='right')
    plt.legend()
    plt.grid()
    plt.show()

# Jalankan algoritma genetika
# Output hasil
if __name__ == "__main__":
    best_route, best_distance = genetic_algorithm(
        distance_matrix,
        demands,
        truck_capacity,
        population_size,
        generations,
        mutation_rate,
        elitism_count
    )

    # Hitung nilai fitness rute terbaik
    best_fitness = 1 / best_distance

    # Output hasil
    print("\n=== Hasil Akhir ===")
    print(f"Rute Terbaik     : {best_route}")
    print(f"Jarak Total (km) : {best_distance:.2f}")
    print(f"Fitness  : {best_fitness:.6f}")  
    

    visualize_route(best_route, coordinates)
