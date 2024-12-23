import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from scipy.ndimage import label


# Define custom colors for each terrain type
terrain_colors = ["green", "blue", "darkgreen", "gray"]
custom_cmap = ListedColormap(terrain_colors)

# Terrain
TERRAIN = {
    0: "Grass",
    1: "River",
    2: "Tree",
    3: "Mountain",
}

MAP_SIZE = 20
POP = 200
GEN = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
ELITISM = 20

# Maps
def generate_random_map(size):
    return np.random.randint(0, len(TERRAIN), (size, size))

# Use custom_cmap when visualizing the map
def visualize_map(map_data, generation, index, output_folder="output"):
    plt.figure(figsize=(6, 6))
    plt.imshow(map_data, cmap=custom_cmap, interpolation="nearest")
    plt.title(f"Gen {generation}, Map {index}")
    plt.axis("off")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f"{output_folder}/gen_{generation}_map_{index}.png")
    plt.close()

def mutate(map_data):
    new_map = map_data.copy()
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if random.random() < MUTATION_RATE:
                new_map[i, j] = random.randint(0, len(TERRAIN) - 1)
    return new_map

def uniform_crossover(parent1, parent2):
    if random.random() <= CROSSOVER_RATE:
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if random.random() >= 0.5:
                    offspring1[i][j] = parent2[i][j]
                    offspring2[i][j] = parent1[i][j]
        return offspring1, offspring2
    return parent1.copy(), parent2.copy()

def fitness(pop):
    fitness = 0

    # River fit
    river_tiles = (pop == 1)
    labeled_array, num_segments = label(river_tiles)
    if num_segments == 1:
        fitness += MAP_SIZE
    else:
        fitness -= num_segments * (MAP_SIZE ** 0.5) # Penalize for multiple river segments

    # Grass fit
    grass_tiles = np.sum(pop == 0)
    total_tiles = MAP_SIZE ** 2
    grass_percentage = grass_tiles / total_tiles
    if grass_percentage >= 0.5:
        fitness += 10  # Reward for having more grass tiles
    else:
        fitness -= (0.5 - grass_percentage) * 100  # Penalty for less than 50% grass tiles

    # Tree fit
    forest_tiles = (pop == 2)
    labeled_forests, num_forest_clusters = label(forest_tiles)
    cluster_sizes = np.array([np.sum(labeled_forests == i) for i in range(1, num_forest_clusters + 1)])
    fitness += np.sum(cluster_sizes ** 2) / MAP_SIZE  # Reward larger forest clusters

    # Mountain fit
    # Reward larger mountain clusters and punish if don't
    mountain_tiles = (pop == 3)
    labeled_mountains, num_mountain_clusters = label(mountain_tiles)
    cluster_sizes = []
    for cluster_id in range(1, num_mountain_clusters + 1):  # Skip label 0 (background)
        cluster_size = np.sum(labeled_mountains == cluster_id)  # Count tiles in this cluster
        cluster_sizes.append(cluster_size)
    for i in cluster_sizes:
        if i < MAP_SIZE:
            fitness -= (MAP_SIZE - i) * (MAP_SIZE ** 0.5)
        else:
            fitness += (i - MAP_SIZE) * (MAP_SIZE ** 0.5)

    return fitness

def tournament_selection(pop):
    players = np.random.choice(len(pop), 2, replace = False)
    if fitness(pop[players[0]]) > fitness(pop[players[1]]):
        return pop[players[0]]
    return pop[players[1]]

def evolutionary_algorithm():
    population = [generate_random_map(MAP_SIZE) for _ in range(POP)]

    for generation in range(GEN):
        print(f"Gen {generation + 1}")
        allfit = [fitness(i) for i in population]
        sortPop = [[population[i], allfit[i]] for i in range(POP)]
        sortPop = sorted(sortPop, key=lambda x: x[1], reverse = True)
        # print(sortPop)
        keep = sortPop[:ELITISM]
        newGen = [ind[0] for ind in keep]

        # crossover
        new_population = []
        times = (POP-ELITISM) / 2
        for i in range(int(times)):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = uniform_crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)
        # mutation
        new_population = [mutate(ind) for ind in new_population]

        for i in range(POP-ELITISM):
            newGen.append(new_population[i])

        population = newGen

        maxFit = np.argmax(allfit)
        print(maxFit, allfit[maxFit])

    visualize_map(population[maxFit], generation, maxFit)

if __name__ == "__main__":
    evolutionary_algorithm()