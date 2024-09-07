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
import asyncio
import time
import concurrent.futures

class DPSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int)->List[int]:
        distance_matrix = formatted_problem
        if not self.is_solvable(distance_matrix):
            return False
        
        n = len(distance_matrix)
        memo = {}

        async def visit(visited, current):
            if self.future_tracker.get(future_id):
                return None, None
            if (visited, current) in memo:
                return memo[(visited, current)]

            if visited == (1 << n) - 1:
                return distance_matrix[current][0], [current]  # Return to the origin city

            min_cost = float('inf')
            min_path = []
            for city in range(n):
                if visited & (1 << city) == 0:
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
 
    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem])->List[List[Union[int,float]]]:
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
    
