import numpy as np
import time
import math
import concurrent.futures
import random

LIMIT_TIME_OUT = 20
# Example node array with 5 unique node identifiers
nodes = list(range(6223910, 6223910 + 4898))  # Example unique node identifiers

# Generate a random distance matrix for 5 nodes
num_nodes = len(nodes)
np.random.seed(42)  # For reproducibility

# Random distances, making sure they are symmetric
random_distances = np.random.rand(num_nodes, num_nodes) * 10  # Random distances scaled
edges = random_distances + random_distances.T  # Making it symmetric
np.fill_diagonal(edges, 0)  # Distance to itself is 0

print(f"Edges: {edges}")

def solve_tsp_nearest_neighbor(dist_matrix, num_attempts=8):
    num_points = len(dist_matrix)
    best_tour = None
    best_cost = float("inf")

    # Adjust the number of attempts based on the number of points
    if num_points <= 2000:
        num_attempts = 400
    elif num_points <= 3000:
        num_attempts = 200
    elif num_points <= 4000:
        num_attempts = 150
    else:
        num_attempts = 230

    print(f"num_attempts: {num_attempts}")

    def nearest_neighbor(start_node):
        visited = np.zeros(num_points, dtype=bool)
        tour = [start_node]
        visited[start_node] = True

        for _ in range(1, num_points):
            last = tour[-1]
            # Vectorized operation to find the next node
            next_node = np.argmin(np.where(visited, float("inf"), dist_matrix[last]))
            tour.append(int(next_node))
            visited[next_node] = True

        tour.append(start_node)
        tour_cost = np.sum(dist_matrix[tour[:-1], tour[1:]])

        return tour, tour_cost, start_node

    # Attempt starting from node 0
    tour_from_zero, cost_from_zero, start_node = nearest_neighbor(0)

    print("Path from 0 node:", tour_from_zero)
    print("Cost from 0 node:", cost_from_zero)

    if cost_from_zero < best_cost:
        best_cost = cost_from_zero
        best_tour = tour_from_zero

    # Use multi-processing for random node attempts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(num_attempts):
            start_node = np.random.randint(num_points)
            print(f"Process start node: {start_node}")
            futures.append(executor.submit(nearest_neighbor, start_node))
            # print(f"Process start node: {_}")
            # futures.append(executor.submit(nearest_neighbor, _))

        for future in concurrent.futures.as_completed(
            futures, timeout=LIMIT_TIME_OUT
        ):
            try:
                current_tour, current_cost, start_node = future.result()
                print(f"Process result of start node: {start_node}")
                # Add cost to return to the start node
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour
                # Check for timeout
                if time.time() - time_start > LIMIT_TIME_OUT:
                    print(
                        f"Timeout reached at {time.time() - time_start} seconds. Returning the best solution so far."
                    )
                    break
            except concurrent.futures.TimeoutError:
                print(
                    f"Timeout reached at {time.time() - time_start} seconds. Returning the best solution so far."
                )
                break
    print(
        f"Diff between best cost: {best_cost} and returned and zero cost: {cost_from_zero} is: {best_cost - cost_from_zero}"
    )

    return best_tour, best_cost


time_start = time.time()
# Solve the TSP for the given distance matrix
best_tour, best_cost = solve_tsp_nearest_neighbor(edges)

# Print the TSP path and cost
# print("Best path:", best_tour)
print("Best cost:", best_cost)
print(f"Time taken: {time.time() - time_start}:")
