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
import concurrent.futures
from math import sqrt

import bittensor as bt

LIMIT_TIME_OUT = 20

class MultiGreedySolver(BaseSolver):
    def __init__(
        self,
        problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = [
            GraphV1Problem(n_nodes=2),
            GraphV1Problem(n_nodes=2, directed=True, problem_type="General TSP"),
        ],
    ):
        super().__init__(problem_types=problem_types)
        self.number_trials = 10

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        time_start = time.time()
        print(f"Time start solve: {time_start}")

        # Calculate the Euclidean distance between every pair of points
        # dist_matrix = np.zeros((num_points, num_points))
        # for i in range(num_points):
        #     for j in range(i + 1, num_points):
        #         dist = sqrt(
        #             (formatted_problem[i][0] - formatted_problem[j][0]) ** 2
        #             + (formatted_problem[i][1] - formatted_problem[j][1]) ** 2
        #         )
        #         dist_matrix[i][j] = dist_matrix[j][i] = dist

        dist_matrix = formatted_problem
        num_points = len(dist_matrix[0])
        best_tour = None
        best_cost = float("inf")

        # Adjust the number of attempts based on the number of points
        if num_points <= 2000:
            num_attempts = 500
        elif num_points <= 3000:
            num_attempts = 200
        elif num_points <= 4000:
            num_attempts = 150
        else:
            num_attempts = 300

        print(f"num_attempts: {num_attempts}")

        def nearest_neighbor(start_node):
            visited = np.zeros(num_points, dtype=bool)
            tour = [start_node]
            visited[start_node] = True

            for _ in range(1, num_points):
                last = tour[-1]
                # Vectorized operation to find the next node
                next_node = np.argmin(
                    np.where(visited, float("inf"), dist_matrix[last])
                )
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
                print(f"Process start node: {start_node}")
                start_node = np.random.randint(num_points)
                futures.append(executor.submit(nearest_neighbor, start_node))

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
                        print(f"Timeout reached at {time.time() - time_start} seconds. Returning the best solution so far.")
                        break
                except concurrent.futures.TimeoutError:
                    print(f"Timeout reached at {time.time() - time_start} seconds. Returning the best solution so far.")
                    break

        print(f"Diff between best cost: {best_cost} and returned and zero cost: {cost_from_zero} is: {best_cost - cost_from_zero}")

        return best_tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges


if __name__ == "__main__":
    # # runs the solver on a test MetricTSP
    # n_nodes = 100
    # test_problem = GraphV1Problem(n_nodes=n_nodes)
    # solver = NearestNeighbourSolver(problem_types=[test_problem])
    # start_time = time.time()
    # route = asyncio.run(solver.solve_problem(test_problem))
    # print(f"{solver.__class__.__name__} Solution: {route}")
    # print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
    print("Hello")
