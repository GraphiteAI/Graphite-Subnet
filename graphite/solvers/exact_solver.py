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

from typing import Union, List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from graphite.utils.graph_utils import timeout
import numpy as np
import asyncio
import time

DEFAULT_SOLVER_TIMEOUT = 20

class DPSolver(BaseSolver):
    def __init__(self, problem_types:List[str]=['Metric TSP', 'General TSP']):
        self.problem_types = problem_types

    @timeout(DEFAULT_SOLVER_TIMEOUT)
    async def solve(self, formatted_problem:List[List[Union[int, float]]])->List[int]:
        distance_matrix = formatted_problem
        if not self.is_solvable(distance_matrix):
            return False
        
        n = len(distance_matrix)
        memo = {}

        async def visit(visited, current):
            if (visited, current) in memo:
                return memo[(visited, current)]

            if visited == (1 << n) - 1:
                return distance_matrix[current][0], [current]  # Return to the origin city

            min_cost = float('inf')
            min_path = []
            for city in range(n):
                if visited & (1 << city) == 0:
                    await asyncio.sleep(0) # Add a small sleep to yield control to the event loop through the for loop
                    cost, path = await visit(visited | (1 << city), city)
                    cost += distance_matrix[current][city]
                    if cost < min_cost:
                        min_cost = cost
                        min_path = [current] + path

            memo[(visited, current)] = min_cost, min_path
            return min_cost, min_path

        min_cost, min_path = await visit(1, 0)
        min_path.append(0)  # Add the origin city at the end
        return min_path

    def problem_transformations(self, problem: GraphProblem)->List[List[Union[int,float]]]:
        return problem.edges
    
    def is_solvable(self, distance_matrix):
        # checks if any row or any col has only inf values
        distance_arr = np.array(distance_matrix).astype(np.float32)
        np.fill_diagonal(distance_arr, np.inf)
        rows_with_only_inf = np.all(np.isinf(distance_arr) | np.isnan(distance_arr), axis=1)
        has_row_with_only_inf = np.any(rows_with_only_inf)

        cols_with_only_inf = np.all(np.isinf(distance_arr) | np.isnan(distance_arr), axis=0)
        has_col_with_only_inf = np.any(cols_with_only_inf)
        return not has_col_with_only_inf and not has_row_with_only_inf

if __name__=="__main__":
    # runs the solver on a test General TSP with 13 nodes
    test_problem = GraphProblem(problem_type='General TSP', objective_function='min', visit_all=True, to_origin=True, n_nodes=13, nodes=[], edges=[[32, 85, 95, 90, 33, 20, 18, 52, 48, 32, 58, 35, 62], [54, 38, 94, 40, 4, 46, 24, 36, 6, 25, 45, 56, 53], [60, 2, 68, 7, 96, 29, 59, 86, 56, 61, 65, 10, 69], [74, 39, 27, 40, 21, 88, 73, 60, 14, 19, 48, 8, 7], [46, 38, 84, 96, 92, 16, 51, 83, 11, 58, 40, 19, 25], [14, 43, 16, 13, 95, 63, 61, 59, 15, 81, 20, 2, 14], [41, 22, 11, 20, 74, 65, 8, 38, 63, 1, 11, 24, 87], [54, 12, 3, 91, 97, 76, 99, 76, 20, 9, 33, 41, 41], [8, 6, 86, 8, 79, 70, 6, 93, 89, 10, 39, 89, 3], [99, 85, 5, 36, 74, 19, 81, 13, 22, 24, 10, 49, 93], [58, 27, 29, 77, 65, 74, 93, 58, 16, 56, 88, 35, 86], [29, 50, 81, 46, 8, 1, 84, 13, 6, 66, 84, 91, 21], [39, 88, 61, 39, 4, 22, 92, 3, 53, 60, 31, 39, 22]], directed=True, simple=True, weighted=False, repeating=False)
    solver = DPSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {13} Nodes: {time.time()-start_time}")
