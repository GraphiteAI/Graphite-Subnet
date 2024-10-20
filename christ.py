import numpy as np
import networkx as nx
import time

# Example node array with 100 unique node identifiers
nodes = list(range(6223910, 6223910 + 100))  # Example unique node identifiers

# Generate a random distance matrix for 100 nodes
num_nodes = len(nodes)
np.random.seed(42)  # For reproducibility

# Random distances, making sure they are symmetric
random_distances = np.random.rand(num_nodes, num_nodes) * 10  # Random distances scaled
edges = random_distances + random_distances.T  # Making it symmetric
np.fill_diagonal(edges, 0)  # Distance to itself is 0

print(f"Edges: {edges}")
# 1. Create a graph using NetworkX
G = nx.Graph()

# Add edges to the graph with distances as weights
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        G.add_edge(nodes[i], nodes[j], weight=edges[i, j])

# Step 1: Create a Minimum Spanning Tree (MST)
mst = nx.minimum_spanning_tree(G)

# Step 2: Find vertices with odd degree in the MST
odd_degree_nodes = [v for v, degree in mst.degree() if degree % 2 == 1]

# Step 3: Find a Minimum Weight Perfect Matching on the odd-degree vertices
subgraph = G.subgraph(odd_degree_nodes)  # Subgraph of odd degree vertices
matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True)

# Step 4: Combine MST and matching to form a multigraph (union of edges)
multigraph = nx.MultiGraph(mst)  # Start with the MST
multigraph.add_edges_from(matching)  # Add the matching edges

# Step 5: Find an Eulerian circuit in the multigraph
eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=nodes[0]))

# Step 6: Convert Eulerian circuit to Hamiltonian by removing repeated nodes
visited = set()
hamiltonian_path = []
for u, v in eulerian_circuit:
    if u not in visited:
        hamiltonian_path.append(v)
        visited.add(v)
# Add the starting node at the end to complete the circuit
hamiltonian_path.append(hamiltonian_path[0])

# Calculate the cost of the Hamiltonian circuit
total_cost = 0
for i in range(len(hamiltonian_path) - 1):
    total_cost += edges[nodes.index(hamiltonian_path[i])][
        nodes.index(hamiltonian_path[i + 1])
    ]

# Output the result
print("Hamiltonian Circuit (Approximate TSP Solution):", hamiltonian_path)
print("Total Cost of the Circuit:", total_cost)


def solve_tsp_nearest_neighbor(dist_matrix, num_attempts=8):
    num_points = len(dist_matrix)
    best_tour = None
    best_cost = float("inf")

    # Adjust the number of attempts based on the number of points
    if num_points <= 2000:
        num_attempts = 12
    elif num_points <= 3000:
        num_attempts = 6
    elif num_points <= 4000:
        num_attempts = 5
    else:
        num_attempts = 4

    print(f"num_attempts: {num_attempts}")

    def nearest_neighbor(start_node):
        visited = [False] * num_points
        tour = [start_node]
        visited[start_node] = True

        for _ in range(1, num_points):
            last = tour[-1]
            next_node = np.argmin(
                [
                    dist_matrix[last][i] if not visited[i] else float("inf")
                    for i in range(num_points)
                ]
            )
            tour.append(next_node)
            visited[next_node] = True

        return tour

    # Attempt starting from node 0
    tour_from_zero = nearest_neighbor(0)
    cost_from_zero = sum(
        dist_matrix[tour_from_zero[i]][tour_from_zero[i + 1]]
        for i in range(-1, num_points - 1)
    )

    # Add cost to return to the start node
    cost_from_zero += dist_matrix[tour_from_zero[-1]][tour_from_zero[0]]

    print("Path from 0 node:", tour_from_zero + [0])
    print("Cost from 0 node:", cost_from_zero)

    if cost_from_zero < best_cost:
        best_cost = cost_from_zero
        best_tour = tour_from_zero

    # Attempt starting from random nodes
    for _ in range(num_attempts):
        start_node = np.random.randint(num_points)  # Randomly select a starting point
        current_tour = nearest_neighbor(start_node)
        current_cost = sum(
            dist_matrix[current_tour[i]][current_tour[i + 1]]
            for i in range(-1, num_points - 1)
        )

        # Add cost to return to the start node
        current_cost += dist_matrix[current_tour[-1]][current_tour[0]]

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour

    # Create the final tour that includes the return to the start node
    best_tour_with_return = best_tour + [best_tour[0]]

    return best_tour_with_return, best_cost


time_start = time.time()
# Solve the TSP for the given distance matrix
best_tour, best_cost = solve_tsp_nearest_neighbor(edges)

# Print the TSP path and cost
print("Best path NN:", best_tour)
print("Best cost NN:", best_cost)
print(f"Time taken: {time.time() - time_start}:")
