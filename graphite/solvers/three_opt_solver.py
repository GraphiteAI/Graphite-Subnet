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


class ThreeOptSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.number_trials = 10

    def calculate_total_distance(self, route, distance_matrix):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        return total_distance

    # 2-opt optimization
    def two_opt(self, route, distance_matrix):
        n = len(route)
        best_route = route[:]
        best_distance = self.calculate_total_distance(route, distance_matrix)

        improvement = True
        while improvement:
            improvement = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    new_distance = self.calculate_total_distance(
                        new_route, distance_matrix
                    )
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improvement = True
            route = best_route
        return best_route, best_distance

    # 3-opt optimization
    def three_opt(self, route, distance_matrix):
        n = len(route)
        best_route = route[:]
        best_distance = self.calculate_total_distance(route, distance_matrix)

        improvement = True
        while improvement:
            improvement = False
            for i in range(1, n - 3):
                for j in range(i + 1, n - 2):
                    for k in range(j + 1, n - 1):
                        # Create the four possible reconnection strategies for 3-opt
                        new_routes = [
                            route[:i] + route[i:j + 1][::-1] + route[j + 1:k + 1][::-1] + route[k + 1:],  # 2-opt + reverse
                            route[:i] + route[j + 1:k + 1] + route[i:j + 1] + route[k + 1:],  # 3-opt exchange
                            route[:i] + route[i:j + 1] + route[j + 1:k + 1][::-1] + route[k + 1:],  # reverse middle segment
                            route[:i] + route[j + 1:k + 1][::-1] + route[i:j + 1][::-1] + route[k + 1:],  # 2-opt both segments
                        ]

                        for new_route in new_routes:
                            new_distance = self.calculate_total_distance(
                                new_route, distance_matrix
                            )
                            if new_distance < best_distance:
                                best_route = new_route
                                best_distance = new_distance
                                improvement = True
            route = best_route
        return best_route, best_distance

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        visited = [False] * n
        route = []
        total_distance = 0

        # Nearest Neighbor heuristic
        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for node in range(n - 1):
            if self.future_tracker.get(future_id):
                return None

            # Find the nearest unvisited neighbor
            nearest_distance = np.inf
            nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])  # random start
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

        print(f"Total Distance (Nearest Neighbor): {total_distance}")

        # Apply 2-opt optimization
        optimized_route_2opt, optimized_distance_2opt = self.two_opt(
            route, distance_matrix
        )
        print(f"Total Distance After 2-opt: {optimized_distance_2opt}")

        # Apply 3-opt optimization
        optimized_route_3opt, optimized_distance_3opt = self.three_opt(
            optimized_route_2opt, distance_matrix
        )
        print(f"Total Distance After 3-opt: {optimized_distance_3opt}")

        return optimized_route_3opt
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
    print("Hello")
