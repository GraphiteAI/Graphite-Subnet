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
import random
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti
from graphite.utils.graph_utils import timeout, sdmtsp_2_tsp
import numpy as np
import time
import asyncio
import random

import bittensor as bt

def split_into_sublists(lst, m):
    # Shuffle the list to ensure randomness
    random.shuffle(lst)
    
    # Calculate the size of each sublist
    k, r = divmod(len(lst), m)
    
    # Create sublists
    sublists = [lst[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(m)]
    
    return sublists

class NaiveSolver(BaseSolver):
    '''
    Mock solver for comparison. Returns the route as per the random selection.
    '''
    def __init__(self, problem_types:List[Union[GraphV2Problem, GraphV2ProblemMulti]]=[GraphV2Problem(), GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int)->List[int]:
        # naively split 
        m = formatted_problem.n_salesmen
        n = formatted_problem.n_nodes
        city_paths = split_into_sublists(list(range(1,n)), m)
        completed_tours = []
        for path in city_paths:
            completed_tours.append([0] + path + [0])
        return completed_tours

    def problem_transformations(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
        if isinstance(problem, GraphV2Problem):
            return problem.edges
        elif isinstance(problem, GraphV2ProblemMulti):
            if problem.single_depot == True:
                # this means that it is a single depot mTSP fomulation
                # Transform the mTSP formulation of single depot into TSP by duplicating source depot
                # For m salesmen and n original cities, add 1 more column and row for each additional salesman that has identical cost structure to the source node.
                problem.edges = np.array(problem.edges)
                print(problem.edges.shape)
                problem.edges = sdmtsp_2_tsp(problem.edges, problem.n_salesmen, )
                print(problem.edges.shape)
                return problem
            else:
                pass