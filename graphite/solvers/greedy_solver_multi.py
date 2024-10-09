# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Union
import matplotlib.pyplot as plt
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.utils.graph_utils import timeout, sdmtsp_2_tsp, decompose_sdmtsp, get_multi_minmax_tour_distance, get_tour_distance
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class NearestNeighbourMultiSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int)->List[int]:
        # naively apply greedy solution and compute total tour length
        m = formatted_problem.n_salesmen
        distance_matrix = formatted_problem.edges
        n = len(distance_matrix[0])
        visited = [False] * n
        route = []
        total_distance = 0

        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for node in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
            # Find the nearest unvisited neighbour
            nearest_distance = np.inf
            nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])# pre-set as random unvisited node
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
        print(f"Distance of single TSP path: {total_distance}")
        route += [0]
        # naive threshold for how long a subroute should be
        threshold_distance = total_distance / m

        # walkthrough the path and iteratively record the index by which a multiple of threshold_distance is crossed
        breakpoint_indices = []

        cumulative_dist = 0
        for i, city in enumerate(route[:-1]):
            cumulative_dist += distance_matrix[city][route[i+1]]
            if cumulative_dist > threshold_distance:
                breakpoint_indices.append(i+1)
                cumulative_dist = 0
        
        # reassign routes via the breaks
        paths = []
        prev_stop = None
        for index in breakpoint_indices:
            try:
                if prev_stop:
                    paths.append([0] + route[prev_stop:index+1] + [0])
                    prev_stop = index + 1
                else:
                    # indicates that this is subpath for the first salesman
                    paths.append(route[:index+1] + [0])
                    prev_stop = index + 1
            except IndexError as e:
                print(e)
        
        paths.append([0]+route[prev_stop:]) # append last route

        distances = []
        for path in paths:
            distance = 0
            for i, city in enumerate(path[:-1]):
                distance += distance_matrix[city][path[i+1]]
            distances.append(distance)

        print(f"multiple salesmen distances: {distances}")

        return paths
    
    def problem_transformations(self, problem: Union[GraphV2ProblemMulti]):
        # this means that it is a single depot mTSP fomulation
        # Transform the mTSP formulation of single depot into TSP by duplicating source depot
        # For m salesmen and n original cities, add 1 more column and row for each additional salesman that has identical cost structure to the source node.
        return problem

if __name__=="__main__":
    # # runs the solver on a test MetricTSP
    # n_nodes = 100
    # test_problem = GraphV1Problem(n_nodes=n_nodes)
    # solver = NearestNeighbourSolver(problem_types=[test_problem])
    # start_time = time.time()
    # route = asyncio.run(solver.solve_problem(test_problem))
    # print(f"{solver.__class__.__name__} Solution: {route}")
    # print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")


    ## Test case for GraphV2Problem
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    loaded_datasets = {}
    with np.load('dataset/Asia_MSB.npz') as f:
        loaded_datasets["Asia_MSB"] = f['data']
    with np.load('dataset/World_TSP.npz') as f:
        loaded_datasets['World_TSP'] = f['data']
    def recreate_edges(problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
    
    n_nodes = random.randint(2000, 5000)
    dataset_ref = "World_TSP"
    selected_ids = random.sample(range(len(loaded_datasets[dataset_ref])), n_nodes)
    print(len(selected_ids))
    n_salesmen = 4
    depots = [0]*4
    # Show the plot
    solver = NearestNeighbourMultiSolver()
    test_problem_multi = GraphV2ProblemMulti(n_nodes=n_nodes, dataset_ref=dataset_ref, selected_ids=selected_ids, n_salesmen=n_salesmen, depots=depots)
    test_problem_multi.edges = recreate_edges(test_problem_multi)
    print(len(test_problem_multi.edges))
    paths = asyncio.run(solver.solve_problem(test_problem_multi))
    visited_nodes = []
    for path in paths:
        visited_nodes.extend(path)
    visited_nodes.sort()
    for i, node in enumerate(visited_nodes[:-1]):
        if visited_nodes[i+1] - node > 1:
            raise ValueError("invalid routes")
    assert max(visited_nodes) == n_nodes - 1, ValueError("not the right number of cities")
    # print(len(paths))
    # for path in paths:
    #     print(", ".join([str(x) for x in path]) + "\n\n")
    test_synapse_multi = GraphV2Synapse(problem = test_problem_multi, solution=paths)
    path_cost = get_multi_minmax_tour_distance(test_synapse_multi)
    print(path_cost)
    
    # Plot each path
    for path in paths:
        # Extract coordinates for the current path
        node_coords_np = loaded_datasets[test_problem_multi.dataset_ref]
        x_values = [np.array([node_coords_np[i][1:] for i in test_problem_multi.selected_ids])[i][0] for i in path]
        y_values = [np.array([node_coords_np[i][1:] for i in test_problem_multi.selected_ids])[i][1] for i in path]
    
        # Plot the path
        plt.plot(x_values, y_values, marker='o')  # 'o' adds markers at the cities
        plt.text(x_values[0], y_values[0], f'Start {path[0]}', fontsize=9, verticalalignment='bottom')
        plt.text(x_values[-1], y_values[-1], f'End {path[-1]}', fontsize=9, verticalalignment='top')

    # Add labels and title
    plt.title('Paths between Cities')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.legend([f'Path {i+1}' for i in range(len(paths))], loc='best')

    # Show the plot
    plt.show()

    # for m in range(2, 10):
    #     test_problem_multi = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB", n_salesmen=m, depots = [0]*m)
    #     if isinstance(test_problem_multi, GraphV2ProblemMulti):
    #         test_problem_multi.edges = recreate_edges(test_problem_multi)
    #         route_multi = asyncio.run(solver.solve_problem(test_problem_multi))
    #         solver = NearestNeighbourSolver(problem_types=[test_problem])
    #         paths = decompose_sdmtsp(route_multi, test_problem_multi.n_salesmen)
    #         print(len(paths))

    #         test_synapse_multi = GraphV2Synapse(problem = test_problem_multi, solution=paths)
    #         total_path_cost = get_multi_tour_distance(test_synapse_multi)
    #         print(total_path_cost)
                
    #             # Plot each path
    #         for path in paths:
    #             # Extract coordinates for the current path
    #             node_coords_np = loaded_datasets[test_problem_multi.dataset_ref]
    #             x_values = [np.array([node_coords_np[i][1:] for i in test_problem_multi.selected_ids])[i][0] for i in path]
    #             y_values = [np.array([node_coords_np[i][1:] for i in test_problem_multi.selected_ids])[i][1] for i in path]
                
    #             # Plot the path
    #             plt.plot(x_values, y_values, marker='o')  # 'o' adds markers at the cities
    #             plt.text(x_values[0], y_values[0], f'Start {path[0]}', fontsize=9, verticalalignment='bottom')
    #             plt.text(x_values[-1], y_values[-1], f'End {path[-1]}', fontsize=9, verticalalignment='top')

    #         # Add labels and title
    #         plt.title('Paths between Cities')
    #         plt.xlabel('X Coordinate')
    #         plt.ylabel('Y Coordinate')
    #         plt.grid()
    #         plt.legend([f'Path {i+1}' for i in range(len(paths))], loc='best')

    #         # Show the plot
    #         plt.show()
    # print(f"{solver.__class__.__name__} Solution: {paths}")
    # print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")

    # solver = NearestNeighbourSolver(problem_types=[test_problem])
    # start_time = time.time()
    # route = asyncio.run(solver.solve_problem(test_problem))
    # print(f"{solver.__class__.__name__} Solution: {route}")
    # print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
