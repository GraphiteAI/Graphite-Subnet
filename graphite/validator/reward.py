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
from graphite.utils.graph_utils import is_valid_solution, get_portfolio_distribution_similarity
from graphite.protocol import GraphV1Problem, GraphV1Synapse, GraphV2Problem, GraphV2Synapse, GraphV1PortfolioSynapse
from graphite.solvers import NearestNeighbourSolver, BeamSearchSolver, DPSolver, HPNSolver
from graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
import bittensor as bt

import asyncio
import time
from copy import deepcopy
from typing import Optional

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
            bt.logging.error(f"reward _ current solver: {self.__class__.__name__} cannot handle received problem: {self.problem.problem_type}")
            return False
        self.synapse.solution = self.benchmark_path
        bt.logging.info(f"Validator found solution: {self.benchmark_path}")
        self.benchmark_score = self.score_response(self.synapse)
        bt.logging.info(f"Validator score: {self.benchmark_score}")

    def get_score(self, response: Union[GraphV1Synapse, GraphV2Synapse]):
        # all cost_functions should handle False as an indication that the problem was unsolvable and assign it a value of np.inf
        synapse_copy = self.synapse
        synapse_copy.solution = response.solution
        try:
            path_cost = self.cost_function(synapse_copy)
        except Exception as e:
            # AssertionError or ValueError indicating an invalid solution
            print(e)
            path_cost = np.inf
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

class ScorePortfolioResponse:
    def __init__(self, mock_synapse: Union[GraphV1PortfolioSynapse], solution):
        self.synapse = mock_synapse
        self.problem = self.synapse.problem
        # internally, validators apply a 30s timeout as the benchmark solution
        # self.solver = NearestNeighbourSolverVali() # create instance of benchmark solver
        self.solver = BENCHMARK_SOLUTIONS[self.problem.problem_type]() # create instance of benchmark solver
        self.cost_function = COST_FUNCTIONS[self.problem.problem_type] # default cost function is "get_tour_distance" or "get_multi_minmax_tour_distance"
        # asyncio.create_task(self.get_benchmark())
        self._current_num_concurrent_forwards = 1
        self.benchmark = solution

    @property
    def current_num_concurrent_forwards(self):
        return self._current_num_concurrent_forwards
    
    @current_num_concurrent_forwards.setter
    def current_num_concurrent_forwards(self, value):
        self._current_num_concurrent_forwards = value

    def get_score(self, response: Union[GraphV1PortfolioSynapse]):
        # all cost_functions should handle False as an indication that the problem was unsolvable and assign it a value of np.inf
        synapse_copy = self.synapse
        synapse_copy.solution = response.solution
        try:
            swaps, objective_score = self.cost_function(synapse_copy)
            return swaps, objective_score
        except Exception as e:
            # AssertionError or ValueError indicating an invalid solution
            print(e)
            return 1000000, 0

    def score_response(self, response: Union[GraphV1PortfolioSynapse]):
        if is_valid_solution(self.problem, response.solution):
            swaps, objective_score = self.get_score(response)
            # check if the response beats greedy algorithm: return 0 if it performs poorer than greedy
            return swaps, objective_score
        elif response.solution == False:
            return 1000000, 0
        else:
            # the miner's response is invalid, return 0
            return 1000000, 0
    
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
            return ((1 - abs(best_score-score)/abs(best_score-reference))**2)*0.8 + 0.2
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

def scaled_portfolio_rewards(scores, benchmark: any, objective_function:str = 'max'):
    def score_gap(score, best_score, reference):
        if is_approximately_equal(score, reference):
            return 0.2 # matched the benchmark so assign a floor score
        elif score_worse_than_reference(score, reference, objective_function):
            return 0 # scored worse than the required benchmark
        else:
            # proportionally scale rewards based on the relative normalized scores
            assert (not is_approximately_equal(best_score, reference) and not score_worse_than_reference(best_score, reference, objective_function)), ValueError(f"Best score is worse than reference: best-{best_score}, ref-{reference}")
            return ((1 - abs(best_score-score)/abs(best_score-reference))**2)*0.8 + 0.2
    

    swap_values = [1/score[0] for score in scores if score[0] != 1000000 and score[1] != 0] + [1/benchmark[0]]
    objective_score_values = [score[1] for score in scores if score[0] != 1000000 and score[1] != 0] + [benchmark[1]]

    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0.0 for x in values]

    norm_swap_values = normalize(swap_values)
    norm_objective_score_values = normalize(objective_score_values)
    normalized_data = [[f, s] for f, s in zip(norm_swap_values, norm_objective_score_values)]

    weighted_scores = []
    for score in normalized_data:
        weighted_scores.append(score[0]*0.6 + score[1]*0.4)

    benchmark_score = weighted_scores[len(weighted_scores)-1]
    weighted_scores = weighted_scores[:-1]

    final_scores = []
    i = 0  
    for score in scores:
        if score[0] == 1000000 or score[1] == 0:
            final_scores.append(0)
        else:
            final_scores.append(weighted_scores[i])
            i += 1

    if final_scores:
        best_score = max(final_scores)
        worst_score = min(final_scores) 
    else:
        # this means that no valid score was found
        return [0 for score in scores]

    if len(weighted_scores) == 1:
        # NOTE: this means that none of the miners returned a valid solution even though the benchmark is valid
        return [0 for score in scores]
    elif not score_worse_than_reference(worst_score, benchmark_score, 'max'):
        return [score_gap(score, best_score, worst_score) for score in final_scores]
    else:
        return [score_gap(score, best_score, benchmark_score) for score in final_scores]

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

def get_portfolio_rewards(
    self,
    score_handler: ScorePortfolioResponse,
    responses: List[Union[GraphV1PortfolioSynapse]],
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
    for response in responses:
        # cast the indices to int
        try:
            if isinstance(response.solution, list) and isinstance(response.solution[0], list):
                response.solution = [[int(swap[0]), int(swap[1]), int(swap[2]), swap[3]] for swap in response.solution]
        except IndexError:
            pass
        except TypeError:
            # indicates the solution is not a list of list of Union[int, float]
            pass
    miner_scores = [score_handler.score_response(response) for response in responses]

    benchmark_response = deepcopy(responses[0])
    benchmark_response.solution = score_handler.benchmark
    rewards = scaled_portfolio_rewards(miner_scores, get_portfolio_distribution_similarity(benchmark_response))

    ## final checks
    rewards = [reward if response.solution is not None else 0 for reward, response in zip(rewards, responses)]

    return torch.FloatTensor(
        rewards
    ).to(self.device)


from graphite.yield_protocol import YieldDataRequestSynapse, MinerYield, LeaderPerformanceData

class ScoreYieldResponse:
    def __init__(self, mock_synapse: YieldDataRequestSynapse):
        self.synapse = mock_synapse
        self.weights = {
            "historical_daily_pnl": 0.1, # We reward leaders for good recent performance
            "sharpe_ratio": 0.55, # We want leaders to optimize for good stable returns
            "max_drawdown": 0.1, # We also want to avoid large drawdowns
            "num_copy_traders": 0.1, # We also want to reward leaders who have large impact on the copy trading community
            "notional_value_of_copy_traders": 0.05,
            "volume": 0.1 # We want to reward leaders who actively manage their portfolios which drives attributed volume
        }

    def get_last_week_average(self, data: Optional[list[float]]) -> float:
        if data is None:
            return - np.inf
        window_size = 7
        if len(data) < window_size:
            return - np.inf
        else:
            data = data[-window_size:]
            # Create exponential weights where more recent data has higher weight
            weights = np.exp(np.linspace(-1, 0, window_size))
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            # Calculate weighted average
            return np.sum(data * weights)
    
    def transpose_data(self, performance_data_list: List[Union[LeaderPerformanceData, None]]) -> dict[str, np.ndarray]:
        '''
        returns a dictionary of data based on the data type
        '''
        data = {}
        for field_name in LeaderPerformanceData.model_fields.keys():
            data[field_name] = [getattr(performance_data, field_name) if isinstance(performance_data, LeaderPerformanceData) else None for performance_data in performance_data_list]
        return data


    def compute_score(self, transposed_data: dict[str, List[Union[float, int]]]) -> dict[str, np.ndarray]:
        '''
        Takes the transposed data and computes a score for each of the fields.
        Uses ordinal ranking where equal values get the same rank.
        '''
        def map_null_to_inf(data: List[Union[float, int]], negative: bool = False) -> List[Union[float, int]]:
            '''
            Args:
                data: List[Union[float, int]]
                negative: bool - Set True if the field semantically ranks higher values as better
            Returns:
                List[Union[float, int]]
            '''
            if negative:
                return [-np.inf if value is None else value for value in data]
            else:
                return [np.inf if value is None else value for value in data]
        
        scores = {}
        for field_name, field_data in transposed_data.items():
            if field_name == "historical_daily_pnl":
                values = [self.get_last_week_average(data) for data in field_data]
            elif field_name == "max_drawdown":
                # we want to minimize the max drawdown so we negate the values
                field_data = map_null_to_inf(field_data)
                values = [-value for value in field_data]
            else:
                field_data = map_null_to_inf(field_data, True)
                values = field_data
            
            # Convert to numpy array and handle None values
            values = np.array([v if v is not None else -np.inf for v in values])
            
            # Get the indices that would sort the array
            sort_indices = np.argsort(values)
            
            # Initialize ranks array
            ranks = np.zeros_like(sort_indices)
            
            # Assign ranks, handling ties --> higher rank is better score
            current_rank = 0
            current_value = None
            for i, idx in enumerate(sort_indices):
                if values[idx] != current_value:
                    current_rank = i
                    current_value = values[idx]
                ranks[idx] = current_rank
            
            # Normalize ranks to be between 0 and 1
            if len(ranks) > 1:
                scores[field_name] = ranks / (len(ranks) - 1)
            else:
                scores[field_name] = np.array([0.5])  # If only one value, give it middle rank
                
        return scores
    
    def get_composite_score(self, scores: dict[str, float]) -> float:
        '''
        returns a composite score for the leader
        '''
        return sum(score * self.weights[field_name] for field_name, score in scores.items())
    
    def get_rewards(self, synapse: YieldDataRequestSynapse):
        miner_uids = [miner_yield.uid for miner_yield in synapse.yields]
        # for each of the fields, we rank the miner_uids and generate a score
        # a composite score is generate by taking the weighted sum of the scores
        transposed_data = self.transpose_data([miner_yield.yield_data for miner_yield in synapse.yields])
        scores = self.compute_score(transposed_data)
        rewards = self.get_composite_score(scores)
        return rewards
    
if __name__=='__main__':
    # # simulate solvers and responses
    # test_problem = GraphV1Problem(n_nodes=10, directed=True)
    # # print(test_problem.get_info(verbosity=3))

    # # test miner synapse
    # miner_1_synapse = GraphV1Synapse(problem=test_problem)
    # miner_1_solver = NearestNeighbourSolver()
    # route_1 = asyncio.run(miner_1_solver.solve_problem(miner_1_synapse.problem))
    # miner_1_synapse.solution = route_1
    
    # miner_2_synapse = GraphV1Synapse(problem=test_problem)
    # miner_2_solver = DPSolver()
    # route_2 = asyncio.run(miner_2_solver.solve_problem(miner_2_synapse.problem))
    # miner_2_synapse.solution = route_2

    # miner_3_synapse = GraphV1Synapse(problem=test_problem)
    # miner_3_solver = HPNSolver()
    # route_3 = asyncio.run(miner_3_solver.solve_problem(miner_3_synapse.problem))
    # miner_3_synapse.solution = route_3

    # miner_4_synapse = GraphV1Synapse(problem=test_problem)
    # miner_4_solver = BeamSearchSolver()
    # route_4 = asyncio.run(miner_4_solver.solve_problem(miner_4_synapse.problem))
    # miner_4_synapse.solution = route_4
    # # print(f"reconstructing graph synapse: {GraphV1Synapse.from_headers(miner_1_synapse.to_headers())}")
    # responses = [miner_1_synapse, miner_2_synapse, miner_3_synapse, miner_4_synapse]

    # score_handler = ScoreResponse(mock_synapse=GraphV1Synapse(problem=test_problem))
    # asyncio.run(score_handler.get_benchmark())
    # # Get all the reward results by iteratively calling your reward() function.
    # miner_scores = [score_handler.score_response(response) for response in responses]
    # # print(miner_scores)

    # # Compute rewards
    # # rewards = scaled_rewards(masked_scores,score_handler.benchmark_score)
    # rewards = scaled_rewards(miner_scores,score_handler.benchmark_score)
    # # print(f"Rewards: {rewards}")
    




    # Portfolio V1 
    import random
    from graphite.solvers.greedy_portfolio_solver import GreedyPortfolioSolver
    from graphite.protocol import GraphV1PortfolioProblem

    num_portfolio = random.randint(50, 200)
    subtensor = bt.Subtensor("finney")
    subnets_info = subtensor.all_subnets()
    pools = [[subnet_info.tao_in.tao, subnet_info.alpha_in.tao] for subnet_info in subnets_info]
    num_subnets = len(pools)
    avail_alphas = [subnet_info.alpha_out.tao for subnet_info in subnets_info]

    # Create initialPortfolios: random non-negative token allocations
    initialPortfolios: List[List[Union[float, int]]] = []
    for _ in range(num_portfolio):
        portfolio = [random.uniform(0, avail_alpha//(2*num_portfolio)) if netuid != 0 else random.uniform(0, 10000/num_portfolio) for netuid, avail_alpha in enumerate(avail_alphas)]  # up to 100k tao and random amounts of alpha_out tokens
        initialPortfolios.append(portfolio)

    # Create constraintTypes: mix of 'eq', 'ge', 'le'
    constraintTypes: List[str] = []
    for _ in range(num_subnets):
        constraintTypes.append(random.choice(["eq", "ge", "le", "ge", "le"])) # eq : ge : le = 1 : 2 : 2

    # Create constraintValues: match the types
    constraintValues: List[Union[float, int]] = []
    for ctype in constraintTypes:
        # ge 0 / le 100 = unconstrained subnet
        if ctype == "eq":
            constraintValues.append(random.uniform(0.5, 3.0))  # small fixed value
        elif ctype == "ge":
            constraintValues.append(random.uniform(0.0, 5.0))   # lower bound
        elif ctype == "le":
            constraintValues.append(random.uniform(10.0, 100.0))  # upper bound

    ### Adjust constraintValues in-place to make sure feasibility is satisfied.
    eq_total = sum(val for typ, val in zip(constraintTypes, constraintValues) if typ == "eq")
    min_total = sum(val for typ, val in zip(constraintTypes, constraintValues) if typ in ("eq", "ge"))
    max_total = sum(val if typ in ("eq", "le") else 100 for typ, val in zip(constraintTypes, constraintValues))

    # If eq_total > 100, need to scale down eq constraints
    if eq_total > 100:
        scale = 100 / eq_total
        for i, typ in enumerate(constraintTypes):
            if typ == "eq":
                constraintValues[i] *= scale

    # After fixing eq, recompute min and max
    min_total = sum(val for typ, val in zip(constraintTypes, constraintValues) if typ in ("eq", "ge"))
    max_total = sum(val if typ in ("eq", "le") else 100 for typ, val in zip(constraintTypes, constraintValues))

    # If min_total > 100, reduce some "ge" constraints
    if min_total > 100:
        ge_indices = [i for i, typ in enumerate(constraintTypes) if typ == "ge"]
        excess = min_total - 100
        if ge_indices:
            for idx in ge_indices[::-1]:
                if constraintValues[idx] > 0:
                    reduction = min(constraintValues[idx], excess)
                    constraintValues[idx] -= reduction
                    excess -= reduction
                    if excess <= 0:
                        break

    # If max_total < 100, increase some "le" constraints
    if max_total < 100:
        le_indices = [i for i, typ in enumerate(constraintTypes) if typ == "le"]
        shortage = 100 - max_total
        if le_indices:
            for idx in le_indices:
                if constraintValues[idx] < 100:
                    increment = min(100-constraintValues[idx], shortage)
                    constraintValues[idx] += increment
                    shortage -= increment
                    if shortage <= 0:
                        break
    
    # Final clip: make sure no negatives
    for i in range(len(constraintValues)):
        constraintValues[i] = max(0, constraintValues[i])

    test_problem = GraphV1PortfolioProblem(problem_type="PortfolioReallocation", 
                                    n_portfolio=num_portfolio, 
                                    initialPortfolios=initialPortfolios, 
                                    constraintValues=constraintValues,
                                    constraintTypes=constraintTypes,
                                    pools=pools)

    solver1 = GreedyPortfolioSolver(problem_types=[test_problem])

    async def main(timeout):
        try:
            route1 = await asyncio.wait_for(solver1.solve_problem(test_problem), timeout=timeout)
            return route1
        except asyncio.TimeoutError:
            print(f"Solver1 timed out after {timeout} seconds")
            return None  # Handle timeout case as needed

    start_time = time.time()
    swaps = asyncio.run(main(10)) 
    print("Swaps:", len(swaps), "\nNum portfolios:", test_problem.n_portfolio, "\nSubnets:", len(test_problem.constraintTypes), "\nTime Taken:", time.time()-start_time)

    # swaps = swaps[:-1]
    # print("initialPortfolios", test_problem.initialPortfolios)
    test_synapse = GraphV1PortfolioSynapse(problem = test_problem, solution = swaps)
    test_synapse2 = GraphV1PortfolioSynapse(problem = test_problem, solution = swaps[:-150])

    responses = [test_synapse, test_synapse2]

    score_handler = ScorePortfolioResponse(mock_synapse=GraphV1PortfolioSynapse(problem=test_problem), solution=swaps[:-150])

    miner_scores = [score_handler.score_response(response) for response in responses]

    benchmark_response = deepcopy(responses[0])
    benchmark_response.solution = score_handler.benchmark
    # print(miner_scores, get_portfolio_distribution_similarity(benchmark_response))
    rewards = scaled_portfolio_rewards(miner_scores, get_portfolio_distribution_similarity(benchmark_response))
    print(rewards)
