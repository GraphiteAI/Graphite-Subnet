import bittensor as bt
from bittensor import axon, dendrite

from graphite.validator.reward import ScorePortfolioResponse, get_portfolio_rewards
from graphite.utils.uids import get_available_uids

import time
from datetime import datetime

from graphite.base.validator import ScoreType, BaseValidatorNeuron
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.organic_protocol import OrganicPortfolioRequestSynapse, OrganicPortfolioResponseSynapse
from graphite.solvers.greedy_portfolio_solver import GreedyPortfolioSolver


import copy
from graphite.utils.graph_utils import get_portfolio_distribution_similarity
from typing import List, Union, Optional
from pydantic import ValidationError
import asyncio

async def fetch_organic_problem(self: BaseValidatorNeuron):
    # Send a request to the organic provider
    request = OrganicPortfolioRequestSynapse()
    axons = [self.organic_axon]
    bt.logging.info(f"Axons: {axons}")
    responses = await self.dendrite(
        axons=axons,
        synapse=request,
        deserialize=True,
        timeout=12
    )
    response = responses[0]
    if not isinstance(response, OrganicPortfolioRequestSynapse) or response.problem is None:
        return None
    return response

async def forward_organic_solution(self: BaseValidatorNeuron, organic_solution: OrganicPortfolioResponseSynapse)->OrganicPortfolioResponseSynapse:
    # Send the solution to the dendrite and do nothing
    axons = [self.organic_axon]
    bt.logging.info(f"Forwarding Organic Solution for Portfolio rebalancing with job_id: {organic_solution.job_id}")
    responses = await self.dendrite(
        axons=axons,
        synapse=organic_solution,
        deserialize=True,
        timeout=12
    )
    return responses[0]

async def organic_forward(self: BaseValidatorNeuron):
    # Get the organic portfolio problem
    organic_synapse: Optional[OrganicPortfolioRequestSynapse] = await fetch_organic_problem(self)    
    if organic_synapse is None:
        return
    bt.logging.info(f"Organic synapse: {organic_synapse.problem.n_portfolio} portfolios with {len(organic_synapse.problem.pools)} subnets")
    # map the new problem to the GraphV1PortfolioSynapse
    synapse = GraphV1PortfolioSynapse(problem=copy.deepcopy(organic_synapse.problem))
    solver1 = GreedyPortfolioSolver(problem_types=[organic_synapse.problem])
    async def main(timeout, test_problem_obj):
        try:
            swaps = await asyncio.wait_for(solver1.solve_problem(test_problem_obj), timeout=timeout)
            return swaps
        except asyncio.TimeoutError:
            print(f"Solver1 timed out after {timeout} seconds")
            return None  # Handle timeout case as needed
    swaps = await main(10, organic_synapse.problem)

    test_synapse = GraphV1PortfolioSynapse(problem = organic_synapse.problem, solution = swaps)
    swap_count, objective_score = get_portfolio_distribution_similarity(test_synapse)
    
    if swaps != None:
        if swaps != False:
            if len(swaps) > 0 and swap_count != 1000000 and objective_score != 0:
                solution_found = True
    selected_uids = await self.get_k_uids()
    miner_uids = list(selected_uids.keys())
    bt.logging.info(f"Posting V1 portfolio_allocation ports: {organic_synapse.problem.n_portfolio}, subnets: {len(organic_synapse.problem.pools)}")
    # Send the synapse to the dendrite
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=12
    )
    bt.logging.info(f"Received Organic Responses: {len(responses)} responses")
    # Get the rewards from the responses
    graphsynapse_req_updated = GraphV1PortfolioSynapse(problem=organic_synapse.problem)
    score_response_obj = ScorePortfolioResponse(graphsynapse_req_updated, swaps)

    score_response_obj.current_num_concurrent_forwards = 1

    rewards = get_portfolio_rewards(self, score_handler=score_response_obj, responses=responses)
    rewards = rewards.numpy(force=True)

    # Update the scores
    self.update_scores(rewards, miner_uids, ScoreType.ORGANIC)

    # get the best solution and return it as a ResponseSynapse
    if responses[0].solution is not None:
        best_solution_response = responses[0].solution
    else:
        best_solution_response = swaps
    organic_solution = OrganicPortfolioResponseSynapse(problem=organic_synapse.problem, solution=best_solution_response, job_id=organic_synapse.job_id)
    await forward_organic_solution(self, organic_solution)
