import numpy as np
import time
import math
import concurrent.futures
import random

# Example node array with 5 unique node identifiers
nodes = list(range(6223910, 6223910 + 10))  # Example unique node identifiers

# Generate a random distance matrix for 5 nodes
num_nodes = len(nodes)
np.random.seed(42)  # For reproducibility

# Random distances, making sure they are symmetric
random_distances = np.random.rand(num_nodes, num_nodes) * 10  # Random distances scaled
edges = random_distances + random_distances.T  # Making it symmetric
np.fill_diagonal(edges, 0)  # Distance to itself is 0

# Function to generate a random path (permutation of nodes)
def generate_random_path(nodes):
    return random.sample(nodes, len(nodes))


# Number of nodes in the TSP
num_nodes = 5000
nodes = list(range(num_nodes))

# Start the timer
start_time = time.time()
max_time = 5  # maximum time to generate paths (25 seconds)

# List to store generated paths
paths = []

# Generate paths until 25 seconds have passed
while time.time() - start_time < max_time:
    path = generate_random_path(nodes)
    path.append(path[0])
    print(path)
    paths.append(path)

best_cost = 0
best_tour = np.inf
for path in paths:
    tour_cost = np.sum(edges[path[:-1], path[1:]]) + edges[path[-1], path[0]]
    if best_tour < best_tour:
        best_cost = tour_cost
        best_tour = path

# Output the number of unique paths generated
print(f"Number of unique paths generated in {max_time} seconds: {len(paths)}")

# Output the best tour
print(f"Best tour: {best_tour}")

# Output the best tour cost
print(f"Best tour cost: {best_cost}")
