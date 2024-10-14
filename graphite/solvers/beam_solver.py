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
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.utils.graph_utils import timeout, get_tour_distance, get_multi_minmax_tour_distance
import numpy as np
import random
import asyncio
import time

class BeamSearchSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int, beam_width:int=3)->List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])

        # Initialize the beam with the starting point (0) and a total distance of 0
        beam = [(0, [0], 0)]
        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
            candidates = []

            # Expand each path in the beam
            for current_node, path, current_distance in beam:
                for next_node in range(n):
                    if next_node not in path:
                        new_path = path + [next_node]
                        new_distance = current_distance + distance_matrix[current_node][next_node]
                        candidates.append((next_node, new_path, new_distance))

            # Sort candidates by their current distance and select the top-k candidates (beam_width)
            candidates.sort(key=lambda x: x[2])
            beam = candidates[:min(beam_width, len(candidates))]

        # Complete the tour by returning to the starting point (0)
        final_candidates = []
        for current_node, path, current_distance in beam:
            final_distance = current_distance + distance_matrix[current_node][0]
            final_candidates.append((path + [0], final_distance))

        # Select the best tour from the final candidates
        best_path, best_distance = min(final_candidates, key=lambda x: x[1])

        return best_path
    
    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti]):
        return problem.edges
        
if __name__=='__main__':
    # runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphV1Problem(n_nodes=n_nodes)
    solver = BeamSearchSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")