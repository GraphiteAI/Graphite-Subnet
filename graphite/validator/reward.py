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

import torch
import numpy as np
from typing import List, Union
from graphite.utils.constants import BENCHMARK_SOLUTIONS, COST_FUNCTIONS
from graphite.utils.graph_utils import is_valid_solution
from graphite.protocol import GraphV1Problem, GraphV1Synapse, GraphV2Problem, GraphV2Synapse
from graphite.solvers import NearestNeighbourSolver, BeamSearchSolver, DPSolver, HPNSolver
from graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
import bittensor as bt

import asyncio
import time

def is_approximately_equal(value1, value2, tolerance_percentage=0.00001):
    # Handle infinite values explicitly
    if np.isinf(value1) or np.isinf(value2):
        return value1 == value2
    
    # Calculate the absolute tolerance from the percentage
    tolerance = tolerance_percentage / 100 * value1
    return np.isclose(value1, value2, atol=tolerance)

def score_worse_than_reference(score, reference, objective_function):
    if objective_function == "min":
        if score > reference:
            return True
        else:
            return False
    else:
        if score < reference:
            return True
        else:
            return False


class ScoreResponse:
    def __init__(self, mock_synapse: Union[GraphV1Synapse, GraphV2Synapse]):
        self.synapse = mock_synapse
        self.problem = self.synapse.problem
        # internally, validators apply a 30s timeout as the benchmark solution
        # self.solver = NearestNeighbourSolverVali() # create instance of benchmark solver
        self.solver = BENCHMARK_SOLUTIONS[self.problem.problem_type]() # create instance of benchmark solver
        self.cost_function = COST_FUNCTIONS[self.problem.problem_type] # default cost function is "get_tour_distance" or "get_multi_minmax_tour_distance"
        # asyncio.create_task(self.get_benchmark())
        self._current_num_concurrent_forwards = 1

    @property
    def current_num_concurrent_forwards(self):
        return self._current_num_concurrent_forwards
    
    @current_num_concurrent_forwards.setter
    def current_num_concurrent_forwards(self, value):
        self._current_num_concurrent_forwards = value
    
    async def get_benchmark(self):
        # self.benchmark_path = await self.solver.solve_problem(self.problem) # this is False if the problem is unsolvable
        if self.solver.is_valid_problem(self.problem):
            self.benchmark_path =  await self.solver.solve(self.solver.problem_transformations(self.problem), self._current_num_concurrent_forwards)
        else:
            # solver is unable to solve the given problem
            bt.logging.error(f"current solver: {self.__class__.__name__} cannot handle received problem: {self.problem.problem_type}")
            return False
        self.synapse.solution = self.benchmark_path
        bt.logging.info(f"Validator found solution: {self.benchmark_path}")
        self.benchmark_score = self.score_response(self.synapse)
        bt.logging.info(f"Validator score: {self.benchmark_score}")

    def get_score(self, response: Union[GraphV1Synapse, GraphV2Synapse]):
        # all cost_functions should handle False as an indication that the problem was unsolvable and assign it a value of np.inf
        synapse_copy = self.synapse
        synapse_copy.solution = response.solution
        path_cost = self.cost_function(synapse_copy)
        return path_cost

    def score_response(self, response: Union[GraphV1Synapse, GraphV2Synapse]):
        if is_valid_solution(self.problem, response.solution):
            response_score = self.get_score(response)
            # check if the response beats greedy algorithm: return 0 if it performs poorer than greedy
            return response_score
        elif response.solution == False:
            return np.inf
        else:
            # the miner's response is invalid, return 0
            return np.inf

# we let scores range from 0.2 to 1 based on min_max_scaling w.r.t benchmark and best scores
# if no score is better than benchmark, scores that fail the benchmark (already set to None) are given a reward of 0 and the rest that match the benchmark get 1.0
def scaled_rewards(scores, benchmark: float, objective_function:str = 'min'):
    def score_gap(score, best_score, reference):
        if is_approximately_equal(score, reference):
            return 0.2 # matched the benchmark so assign a floor score
        elif score_worse_than_reference(score, reference, objective_function):
            return 0 # scored worse than the required benchmark
        else:
            # proportionally scale rewards based on the relative normalized scores
            assert (not is_approximately_equal(best_score, reference) and not score_worse_than_reference(best_score, reference, objective_function)), ValueError(f"Best score is worse than reference: best-{best_score}, ref-{reference}")
            return (1 - abs(best_score-score)/abs(best_score-reference))*0.8 + 0.2
    # we find scores that correspond to finite path costs
    # bt.logging.info(f"Miners were scored: {scores}")
    # print(f"Miners were scored: {scores}")
    filtered_scores = [score for score in scores if ((score is not None) and (score != np.inf))]
    # bt.logging.info(f"With the valid scores of: {filtered_scores}")
    # print(f"With the valid scores of: {filtered_scores}")
    if filtered_scores:
        best_score = min(filtered_scores) if objective_function=='min' else max(filtered_scores)
        worst_score = max(filtered_scores) if objective_function=='min' else min(filtered_scores)
    else:
        # this means that no valid score was found
        return [0 for score in scores]

    # if best_score == benchmark:
    #     return [int(score!=None) for score in scores]
    if benchmark == np.inf:
        return [score_gap(score, best_score, worst_score) for score in scores]
    elif not score_worse_than_reference(worst_score, benchmark, 'min'):
        return [score_gap(score, best_score, worst_score) for score in scores]
    else:
        return [score_gap(score, best_score, benchmark) for score in scores]

def get_rewards(
    self,
    score_handler: ScoreResponse,
    responses: List[Union[GraphV1Synapse, GraphV2Synapse]],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.
    - score_handl (ScoreResponse): An instance of the ScoreResponse class that corresponds to the given problem

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    miner_scores = [score_handler.score_response(response) for response in responses]

    # Compute rewards
    rewards = scaled_rewards(miner_scores, score_handler.benchmark_score)

    return torch.FloatTensor(
        rewards
    ).to(self.device)

if __name__=='__main__':
    # simulate solvers and responses
    test_problem = GraphV1Problem(n_nodes=10, directed=True)
    # print(test_problem.get_info(verbosity=3))

    # test miner synapse
    miner_1_synapse = GraphV1Synapse(problem=test_problem)
    miner_1_solver = NearestNeighbourSolver()
    route_1 = asyncio.run(miner_1_solver.solve_problem(miner_1_synapse.problem))
    miner_1_synapse.solution = route_1
    
    miner_2_synapse = GraphV1Synapse(problem=test_problem)
    miner_2_solver = DPSolver()
    route_2 = asyncio.run(miner_2_solver.solve_problem(miner_2_synapse.problem))
    miner_2_synapse.solution = route_2

    miner_3_synapse = GraphV1Synapse(problem=test_problem)
    miner_3_solver = HPNSolver()
    route_3 = asyncio.run(miner_3_solver.solve_problem(miner_3_synapse.problem))
    miner_3_synapse.solution = route_3

    miner_4_synapse = GraphV1Synapse(problem=test_problem)
    miner_4_solver = BeamSearchSolver()
    route_4 = asyncio.run(miner_4_solver.solve_problem(miner_4_synapse.problem))
    miner_4_synapse.solution = route_4
    # print(f"reconstructing graph synapse: {GraphV1Synapse.from_headers(miner_1_synapse.to_headers())}")
    responses = [miner_1_synapse, miner_2_synapse, miner_3_synapse, miner_4_synapse]

    score_handler = ScoreResponse(mock_synapse=GraphV1Synapse(problem=test_problem))
    asyncio.run(score_handler.get_benchmark())
    # Get all the reward results by iteratively calling your reward() function.
    miner_scores = [score_handler.score_response(response) for response in responses]
    # print(miner_scores)

    # Compute rewards
    # rewards = scaled_rewards(masked_scores,score_handler.benchmark_score)
    rewards = scaled_rewards(miner_scores,score_handler.benchmark_score)
    # print(f"Rewards: {rewards}")
    
