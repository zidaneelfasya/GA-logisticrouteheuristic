import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Genetic Algorithm
population_size = 100  # Number of individuals in the population
mutation_rate = 0.1  # Probability of mutation
elitism_count = 2  # Number of best individuals to carry forward
stagnation_limit = 50  # Generasi tanpa perbaikan sebelum berhenti

# Truck capacity
truck_capacity = 271  # Maksimal unit per truk

# Data jarak dan permintaan dari file Excel
file_path = "data_route.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet", index_col=0)

# Mengonversi DataFrame menjadi matriks numpy
distance_matrix = data.iloc[:, :-1].values  # Semua kolom kecuali 'Demand'
demands = data.iloc[:, -1].values  # Kolom terakhir sebagai permintaan
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

# Calculate route distance and check capacity
def calculate_route_distance_and_capacity(route, distance_matrix, demands):
    distance = 0
    load = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
        load += demands[route[i + 1]]  # Toko berikutnya
        if load > truck_capacity:
            return distance, False  # Melampaui kapasitas
    return distance, True

# Evaluate population fitness
def evaluate_population(population, distance_matrix, demands):
    fitness = []
    for route in population:
        distance, is_feasible = calculate_route_distance_and_capacity(route, distance_matrix, demands)
        if is_feasible:
            fitness.append(1 / distance)  # Fitness is inversely proportional to distance
        else:
            fitness.append(1 / (distance + 1e6))  # Penalize infeasible solutions
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

# Split route into multiple trips based on truck capacity
def split_route(route, demands, truck_capacity):
    trips = []
    current_trip = [0]
    current_load = 0
    for loc in route[1:-1]:
        if current_load + demands[loc] > truck_capacity:
            current_trip.append(0)  # Return to depot
            trips.append(current_trip)
            current_trip = [0, loc]
            current_load = demands[loc]
        else:
            current_trip.append(loc)
            current_load += demands[loc]
    current_trip.append(0)  # Return to depot
    trips.append(current_trip)
    return trips

# Main Genetic Algorithm with dynamic generations
def genetic_algorithm(distance_matrix, demands, population_size, mutation_rate, elitism_count, stagnation_limit=50):
    population = initialize_population(population_size, num_locations)
    print(f"=== Populasi Awal ===")
    for i, ind in enumerate(population[:5]):  # Tampilkan 5 individu pertama untuk kejelasan
        print(f"Individu {i+1}: {ind}")
    print("...")  # Indikasi ada lebih banyak individu
    
    best_route = None
    best_distance = float('inf')
    stagnation_counter = 0  # Menghitung jumlah generasi tanpa perbaikan
    generation = 0
    
    while stagnation_counter < stagnation_limit:
        generation += 1
        print(f"\n=== Generasi {generation} ===")
        
        # Evaluasi fitness
        fitness = evaluate_population(population, distance_matrix, demands)
        fitness_table = pd.DataFrame({"Individu": population, "Fitness": fitness})
        print("\nTabel Fitness:")
        print(fitness_table.head())  # Tampilkan 5 teratas
        
        # Cari rute terbaik di generasi ini
        current_best_index = fitness.index(max(fitness))
        current_best_route = population[current_best_index]
        current_best_distance, is_feasible = calculate_route_distance_and_capacity(current_best_route, distance_matrix, demands)
        
        # Periksa apakah rute terbaik diperbarui
        if is_feasible and current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route
            stagnation_counter = 0  # Reset stagnasi jika ada perbaikan
            print(f"Perbaikan ditemukan: {best_route} dengan jarak {best_distance:.2f} km")
        else:
            stagnation_counter += 1
            print(f"Tidak ada perbaikan. Stagnasi: {stagnation_counter}/{stagnation_limit}")
        
        # Elitisme
        next_population = []
        elite_indices = sorted(range(len(fitness)), key=lambda x: fitness[x], reverse=True)[:elitism_count]
        for idx in elite_indices:
            next_population.append(population[idx])
        print("\nElitisme: Menambahkan individu terbaik ke populasi berikutnya")
        
        # Crossover
        print("\nProses Crossover:")
        while len(next_population) < population_size:
            parents = select_parents(population, fitness)
            offspring = order_crossover(parents[0], parents[1])
            next_population.append(offspring)
            print(f"Parents: {parents[0]} & {parents[1]} => Offspring: {offspring}")
        
        # Mutasi
        print("\nProses Mutasi:")
        for i in range(len(next_population)):
            original = next_population[i]
            next_population[i] = mutate(next_population[i], mutation_rate)
            if next_population[i] != original:
                print(f"Mutasi pada Individu {i+1}: {original} => {next_population[i]}")
        
        # Populasi Baru
        population = next_population
        print("\nPopulasi Baru dengan Jarak:")
        for i, ind in enumerate(population[:5]):  # Tampilkan 5 individu pertama untuk kejelasan
            route_distance, is_feasible = calculate_route_distance_and_capacity(ind, distance_matrix, demands)
            status = "Feasible" if is_feasible else "Infeasible"
            print(f"Individu {i+1}: {ind}, Jarak: {route_distance:.2f} km, Status: {status}")
        print("...")
    
    print(f"\n=== Evolusi Berakhir Setelah {generation} Generasi ===")
    print(f"Rute Terbaik: {best_route}")
    print(f"Jarak Total: {best_distance:.2f} km")
    
    return best_route, best_distance

# Coordinates for visualization
coordinates = [
    (0, 0), (2, 3), (4, 1), (6, 5), (3, 4), (5, 2), (1, 5),
    (7, 6), (8, 3), (9, 5), (10, 2)
]

# Visualize route
# def visualize_route(route, coordinates):
#     trips = split_route(route, demands, truck_capacity)
#     plt.figure(figsize=(12, 8))
#     colors = plt.cm.get_cmap('tab10', len(trips))
    
#     for i, trip in enumerate(trips):
#         x = [coordinates[loc][0] for loc in trip]
#         y = [coordinates[loc][1] for loc in trip]
#         plt.plot(x, y, '-o', color=colors(i), label=f'Trip {i+1}')
    
#     plt.title('Optimal Logistics Route with Multiple Trips')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     for i, loc in enumerate(route):
#         plt.text(coordinates[loc][0], coordinates[loc][1], str(loc), fontsize=12, ha='right')
#     plt.legend()
#     plt.grid()
#     plt.show()

# Split route into multiple trips based on truck capacity
def split_route(route, demands, truck_capacity):
    trips = []
    current_trip = [0]
    current_load = 0
    for loc in route[1:-1]:
        if current_load + demands[loc] > truck_capacity:
            current_trip.append(0)  # Return to depot
            trips.append(current_trip)
            current_trip = [0, loc]
            current_load = demands[loc]
        else:
            current_trip.append(loc)
            current_load += demands[loc]
    current_trip.append(0)  # Return to depot
    trips.append(current_trip)
    return trips

# Jalankan algoritma dan buat kesimpulan
if __name__ == "__main__":
    best_route, best_distance = genetic_algorithm(
        distance_matrix,
        demands,
        population_size,
        mutation_rate,
        elitism_count,
        stagnation_limit
    )
    
    # Output kesimpulan
    print("\n=== Hasil Akhir ===")
    print(f"Rute Terbaik     : {best_route}")
    print(f"Jarak Total (km) : {best_distance:.2f}")
    
    # Visualize the result
    # visualize_route(best_route, coordinates)
