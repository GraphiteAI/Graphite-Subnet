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
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.utils.graph_utils import timeout
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem:List[List[Union[int, float]]], future_id:int)->List[int]:
        distance_matrix = formatted_problem
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
        route.append(route[0])
        return route

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges
        
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
    from graphite.data.distance import euc_2d, geom, man_2d
    loaded_datasets = {}
    with np.load('dataset/Asia_MSB.npz') as f:
        loaded_datasets["Asia_MSB"] = np.array(f['data'])
    def recreate_edges(problem: GraphV2Problem):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = [node_coords_np[i][1:] for i in problem.selected_ids]
        num_nodes = len(node_coords)
        edge_matrix = np.zeros((num_nodes, num_nodes))
        if problem.cost_function == "Geom":
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i != j:
                        distance = geom(node_coords[i], node_coords[j])
                        edge_matrix[i][j] = distance
                        edge_matrix[j][i] = distance  # Since it's symmetric
        if problem.cost_function == "Euclidean2D":
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i != j:
                        distance = euc_2d(node_coords[i], node_coords[j])
                        edge_matrix[i][j] = distance
                        edge_matrix[j][i] = distance  # Since it's symmetric
        if problem.cost_function == "Manhatten2D":
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i != j:
                        distance = man_2d(node_coords[i], node_coords[j])
                        edge_matrix[i][j] = distance
                        edge_matrix[j][i] = distance  # Since it's symmetric
        problem.edges = edge_matrix
    n_nodes = random.randint(2000, 5000)
    # randomly select n_nodes indexes from the selected graph
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Euclidean2D", dataset_ref="Asia_MSB")
    print("Problem", test_problem)
    if isinstance(test_problem, GraphV2Problem):
            problem = recreate_edges(test_problem)
    solver = NearestNeighbourSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
