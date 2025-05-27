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
from graphite.solvers.greedy_solver_multi import NearestNeighbourMultiSolver
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.utils.graph_utils import timeout, get_portfolio_distribution_similarity
from graphite.data.dataset_utils import load_default_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import numpy as np
import time
import asyncio
import random
import math
import bittensor as bt
from graphite.base.subnetPool import SubnetPool
from copy import deepcopy

class GreedyPortfolioSolver(BaseSolver):
    '''
    This solver is a constructive nearest_neighbour algorithm that assigns cities to subtours based on the min increase in objective function value.
    '''
    def __init__(self, problem_types:List[GraphV1PortfolioProblem]=[GraphV1PortfolioProblem()]):
        super().__init__(problem_types=problem_types)
    
    def get_valid_start(self, depot_id, distance_matrix, taken_nodes:list[int]=[], selection_range:int=5) -> int:
        distances = [(city_id, distance) for city_id, distance in enumerate(distance_matrix[depot_id].copy())]
        # reverse sort the copied list and pop from it
        assert (selection_range + len(taken_nodes)) < len(distances)
        distances.sort(reverse=True, key=lambda x: x[1])
        closest_cities = []
        while len(closest_cities) < selection_range:
            selected_city = None
            while not selected_city:
                city_id, distance = distances.pop()
                if city_id not in taken_nodes:
                    selected_city = city_id
            closest_cities.append(selected_city)
        return closest_cities[0]
        
    def get_starting_tours(self, depots, distance_matrix):
        taken_nodes = depots.copy()
        initial_incomplete_tours = []
        for depot in depots:
            first_visit = self.get_valid_start(depot, distance_matrix, taken_nodes)
            initial_incomplete_tours.append([depot, first_visit])
            taken_nodes.append(first_visit)
        return initial_incomplete_tours
    
    async def solve(self, formatted_problem:GraphV1PortfolioProblem, future_id:int):
        """
        formatted_problem:
            problem_type: Literal['PortfolioReallocation'] = Field('PortfolioReallocation', description="Problem Type")
            n_portfolio: int = Field(3, description="Number of Portfolios")
            initialPortfolios: List[List[Union[float, int]]] = Field([[0]*100]*3, description="Number of tokens in each subnet for each of the n_portfolio eg. 3 portfolios with 0 tokens in any of the 100 subnets")
            constraintValues: List[Union[float, int]] = Field([1.0]*100, description="Overall Percentage for each subnet in equivalent TAO after taking the sum of all portfolios; they do not need to add up to 100%")
            constraintTypes: List[str] = Field(["ge"]*100, description="eq = equal to, ge = greater or equals to, le = lesser or equals to - the value provided in constraintValues")
            pools: List[Union[float, int]] = Field([[1.0, 1.0]]*100, description="Snapshot of current pool states of all subnets when problem is issued, list idx = netuid, [num_tao_tokens, num_alpha_tokens]")
        
        output:
            bool or solution
            solution = [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_tokens], ... ]
        """

        ### Individual portfolio level swaps required
        def instantiate_pools(pools):
            current_pools: List[SubnetPool] = []
            for netuid, pool in enumerate(pools):
                current_pools.append(SubnetPool(pool[0], pool[1], netuid))
            return current_pools

        start_pools = instantiate_pools(formatted_problem.pools)

        initialPortfolios = deepcopy(formatted_problem.initialPortfolios)
        total_tao = 0
        portfolio_tao = [0] * formatted_problem.n_portfolio
        portfolio_swaps = [] # [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_tokens], ... ]
        for idx, portfolio in enumerate(initialPortfolios):
            for netuid, alpha_token in enumerate(portfolio):
                if alpha_token > 0:
                    emitted_tao = start_pools[netuid].swap_alpha_to_tao(alpha_token)
                    portfolio_swaps.append([idx, netuid, 0, int(alpha_token)])
                    total_tao += emitted_tao
                    portfolio_tao[idx] += emitted_tao

        for netuid, constraint_type in enumerate(formatted_problem.constraintTypes):
            if netuid != 0:
                constraint_value = formatted_problem.constraintValues[netuid]
                tao_required = constraint_value/100 * total_tao
                if constraint_type == "eq" or constraint_type == "ge":
                    for idx in range(len(portfolio_tao)):
                        tao_to_swap = min(portfolio_tao[idx], tao_required)
                        if tao_to_swap > 0:
                            alpha_emitted = start_pools[netuid].swap_tao_to_alpha(tao_to_swap)
                            portfolio_swaps.append([idx, 0, netuid, int(tao_to_swap)])
                            tao_required -= tao_to_swap
                            portfolio_tao[idx] -= tao_to_swap

        return portfolio_swaps
    
    def problem_transformations(self, problem: GraphV1PortfolioProblem):
        return problem



if __name__=="__main__":
    subtensor = bt.Subtensor("finney")
    subnets_info = subtensor.all_subnets()
    
    num_portfolio = random.randint(50, 200)
    pools = [[subnet_info.tao_in.rao, subnet_info.alpha_in.rao] for subnet_info in subnets_info]
    num_subnets = len(pools)
    avail_alphas = [subnet_info.alpha_out.rao for subnet_info in subnets_info]
    
    # Create initialPortfolios: random non-negative token allocations
    initialPortfolios: List[List[int]] = []
    for _ in range(num_portfolio):
        portfolio = [int(random.uniform(0, avail_alpha//(2*num_portfolio))) if netuid != 0 else int(random.uniform(0, 10000*1e9/num_portfolio)) for netuid, avail_alpha in enumerate(avail_alphas)]  # up to 100k tao and random amounts of alpha_out tokens
        # On average, we assume users will invest in about 50% of the subnets
        portfolio = [portfolio[i] if random.random() < 0.5 else 0 for i in range(num_subnets)]
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
    
    for idx, constraintValue in enumerate(constraintValues):
        if random.random() < 0.5:
            constraintTypes[idx] = "eq"
            constraintValues[idx] = 0

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
    print("Swaps:", swaps, "\nNum portfolios:", test_problem.n_portfolio, "\nSubnets:", len(test_problem.constraintTypes), "\nTime Taken:", time.time()-start_time)

    # swaps = swaps[:-1]
    # print("initialPortfolios", test_problem.initialPortfolios)
    test_synapse = GraphV1PortfolioSynapse(problem = test_problem, solution = swaps)
    swaps, objective_score = get_portfolio_distribution_similarity(test_synapse)
    print("Swaps:", swaps, "\nScore:", objective_score)
    