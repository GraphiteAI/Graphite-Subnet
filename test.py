import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from itertools import permutations
import time
import random

# Example node array with 5000 unique node identifiers
nodes = list(range(6223910, 6223910 + 3897))

# Number of nodes
num_nodes = len(nodes)

# Seed for reproducibility
np.random.seed(42)

# Generate a random symmetric distance matrix (edges)
random_distances = np.random.rand(num_nodes, num_nodes) * 10
edges = random_distances + random_distances.T
np.fill_diagonal(edges, 0)  # Distance from a node to itself is 0

print(edges)

# Convert the distance matrix into a NetworkX graph
G = nx.Graph()

# Add nodes
for i in range(num_nodes):
    G.add_node(i)


# Add edges with corresponding weights (distances)
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        G.add_edge(i, j, weight=edges[i, j])

# TSP objective function (tour length calculation)
def calculate_tour_length(G, tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += G[tour[i]][tour[i + 1]]["weight"]
    total_distance += G[tour[-1]][tour[0]]["weight"]  # Return to start
    return total_distance


# Heuristic solver for TSP using Simulated Annealing
def simulated_annealing(
    G, initial_temp=1000, cooling_rate=0.995, stopping_temp=0.00001
):
    # Initial tour (random permutation)
    current_tour = list(np.random.permutation(list(G.nodes)))
    best_tour = list(current_tour)
    best_cost = calculate_tour_length(G, best_tour)
    current_temp = initial_temp

    while current_temp > stopping_temp:
        # Select two nodes to swap
        i, j = np.random.choice(len(current_tour), 2, replace=False)
        new_tour = list(current_tour)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        # Calculate new cost
        new_cost = calculate_tour_length(G, new_tour)

        # Accept new tour based on probability
        if new_cost < best_cost or np.random.rand() < np.exp(
            (best_cost - new_cost) / current_temp
        ):
            current_tour = list(new_tour)
            if new_cost < best_cost:
                best_tour = list(new_tour)
                best_cost = new_cost

        # Cool down
        current_temp *= cooling_rate
    best_tour.append(best_tour[0])  # Return to start
    
    return best_tour, best_cost


def nearest_neighbor(distance_matrix):
    n = len(distance_matrix[0])
    visited = [False] * n
    route = []
    total_distance = 0

    current_node = 0
    route.append(current_node)
    visited[current_node] = True

    for node in range(n - 1):
        # Find the nearest unvisited neighbour
        nearest_distance = np.inf
        nearest_node = random.choice(
            [i for i, is_visited in enumerate(visited) if not is_visited]
        )  # pre-set as random unvisited node
        for j in range(n):
            if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                nearest_distance = distance_matrix[current_node][j]
                nearest_node = j

        # Move to the nearest unvisited node
        route.append(nearest_node)
        visited[nearest_node] = True
        total_distance += nearest_distance
        current_node = nearest_node

    # Return to the starting node
    total_distance += distance_matrix[current_node][route[0]]
    route.append(route[0])
    return route, total_distance


time_start = time.time()
# Solve the TSP
best_tour_nearest, best_cost_nearest = nearest_neighbor(edges)

# Output the result
print(f"Best tour nearest: {best_tour_nearest}and {len(best_tour_nearest)}")
print("Best tour cost nearest:", best_cost_nearest)
print(f"Time taken: {time.time() - time_start}:")

time_start = time.time()
# Solve the TSP
best_tour, best_cost = simulated_annealing(G)

# Output the result
print(f"Best tour annealing: {best_tour}and {len(best_tour)}")
print("Best tour cost annealing:", best_cost)
print(f"Time taken: {time.time() - time_start}:")
