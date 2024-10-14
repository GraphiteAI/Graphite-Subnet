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
from graphite.utils.graph_utils import timeout, get_multi_minmax_tour_distance
from graphite.data.dataset_utils import load_default_dataset
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
    # runs the solver on a test MetricTSP
    class Mock:
        def __init__(self) -> None:
            pass        
    
    mock = Mock()

    load_default_dataset(mock)

    n_nodes = 500
    m = 4

    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=[0]*m)
    solver = NearestNeighbourMultiSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")