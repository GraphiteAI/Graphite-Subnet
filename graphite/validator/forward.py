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

import bittensor as bt
from bittensor import axon, dendrite

from graphite.validator.reward import get_rewards, ScoreResponse, ScorePortfolioResponse, get_portfolio_rewards
from graphite.utils.uids import get_available_uids

import time
from datetime import datetime

from graphite.base.validator import ScoreType
from graphite.protocol import GraphV2Problem, GraphV2ProblemMulti, GraphV2ProblemMultiConstrained, GraphV2Synapse, MAX_SALESMEN, GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.solvers.greedy_solver_multi_4 import NearestNeighbourMultiSolver4
from graphite.solvers.greedy_portfolio_solver import GreedyPortfolioSolver
import numpy as np
import json
import wandb
import os
import random
import requests
import math

from graphite.utils.graph_utils import get_portfolio_distribution_similarity
from typing import List, Union
from pydantic import ValidationError
import asyncio

async def forward(self):

    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """

    bt.logging.info(f"CONCURRENCY IDX: {self.concurrencyIdx}")
    curr_idx = self.concurrencyIdx
    self.concurrencyIdx += 1

    did_organic_task = False
    organic_task_id = ""
    try:
        if self.bearer_token_is_valid:

            url = f"{self.organic_endpoint}/tasks/oldest/{curr_idx}"
            headers = {"Authorization": "Bearer %s"%self.db_bearer_token}
            api_response = requests.get(url, headers=headers)
            api_response_output = api_response.json()
            
            organic_task_id = api_response_output["_id"]
            del api_response_output["_id"]

            did_organic_task = True

            # bt.logging.info(f"ORGANIC TRAFFIC {api_response.text}")
        else:
            api_response_output = []
    except:
        api_response_output = []

    # problem weights
    ref_tsp_value = 0.1
    ref_mtsp_value = 0.1
    ref_mdmtsp_value = 0.2 
    ref_cmdmtsp_value = 0.4
    ref_portfolioV1_value = 0.2

    # randomly select n_nodes indexes from the selected graph
    prob_select = random.randint(0, len(list(self.loaded_datasets.keys()))-1)
    dataset_ref = list(self.loaded_datasets.keys())[prob_select]
    selected_problem_type_prob = random.random()
    if selected_problem_type_prob < ref_tsp_value:
        n_nodes = random.randint(2000, 5000)
        bt.logging.info(f"n_nodes V2 TSP {n_nodes}")
        bt.logging.info(f"dataset ref {dataset_ref} selected from {list(self.loaded_datasets.keys())}" )
        selected_node_idxs = random.sample(range(len(self.loaded_datasets[dataset_ref]['data'])), n_nodes)
        test_problem_obj = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)
    elif selected_problem_type_prob < ref_tsp_value + ref_mtsp_value:
        # single depot mTSP
        n_nodes = random.randint(500, 2000)
        bt.logging.info(f"n_nodes V2 mTSP {n_nodes}")
        bt.logging.info(f"dataset ref {dataset_ref} selected from {list(self.loaded_datasets.keys())}" )
        selected_node_idxs = random.sample(range(len(self.loaded_datasets[dataset_ref]['data'])), n_nodes)
        m = random.randint(2, 10)
        test_problem_obj = GraphV2ProblemMulti(problem_type="Metric mTSP", 
                                                n_nodes=n_nodes, 
                                                selected_ids=selected_node_idxs, 
                                                cost_function="Geom", 
                                                dataset_ref=dataset_ref, 
                                                n_salesmen=m, 
                                                depots=[0 for _ in range(m)])
    elif selected_problem_type_prob < ref_tsp_value + ref_mtsp_value + ref_mdmtsp_value:
        # multi depot mTSP
        n_nodes = random.randint(500, 2000)
        bt.logging.info(f"n_nodes V2 mTSP {n_nodes}")
        bt.logging.info(f"dataset ref {dataset_ref} selected from {list(self.loaded_datasets.keys())}" )
        selected_node_idxs = random.sample(range(len(self.loaded_datasets[dataset_ref]['data'])), n_nodes)
        m = random.randint(2, 10)
        test_problem_obj = GraphV2ProblemMulti(problem_type="Metric mTSP", 
                                                n_nodes=n_nodes, 
                                                selected_ids=selected_node_idxs, 
                                                cost_function="Geom", 
                                                dataset_ref=dataset_ref, 
                                                n_salesmen=m, 
                                                depots=sorted(random.sample(list(range(n_nodes)), k=m)), 
                                                single_depot=False)
    elif selected_problem_type_prob < ref_tsp_value + ref_mtsp_value + ref_mdmtsp_value + ref_cmdmtsp_value:
        non_uniform_demand_prob = random.random()
        # constrained multi depot mTSP
        if non_uniform_demand_prob < 0.5:
            n_nodes = random.randint(500, 2000)
            bt.logging.info(f"n_nodes V2 cmTSP {n_nodes}")
            bt.logging.info(f"dataset ref {dataset_ref} selected from {list(self.loaded_datasets.keys())}" )
            selected_node_idxs = random.sample(range(len(self.loaded_datasets[dataset_ref]['data'])), n_nodes)
            m = random.randint(2, 10)
            constraint = []
            depots = sorted(random.sample(list(range(n_nodes)), k=m))
            demand = [1]*n_nodes
            for depot in depots:
                demand[depot] = 0
            while sum(demand) > sum(constraint):
                constraint = [(math.ceil(n_nodes/m) + random.randint(0, int(n_nodes/m * 0.3)) - random.randint(0, int(n_nodes/m * 0.2))) for _ in range(m-1)]
                constraint += [(math.ceil(n_nodes/m) + random.randint(0, int(n_nodes/m * 0.3)) - random.randint(0, int(n_nodes/m * 0.2)))] if sum(constraint) > n_nodes - (math.ceil(n_nodes/m) - int(n_nodes/m * 0.2)) else [(n_nodes - sum(constraint) + random.randint(int(n_nodes/m * 0.2), int(n_nodes/m * 0.3)))]
            test_problem_obj = GraphV2ProblemMultiConstrained(problem_type="Metric cmTSP", 
                                                    n_nodes=n_nodes, 
                                                    selected_ids=selected_node_idxs, 
                                                    cost_function="Geom", 
                                                    dataset_ref=dataset_ref, 
                                                    n_salesmen=m, 
                                                    depots=depots, 
                                                    single_depot=False,
                                                    demand=demand,
                                                    constraint=constraint)
        else:
            solution_found = False
            while not solution_found:
                n_nodes = random.randint(500, 2000)
                bt.logging.info(f"n_nodes V2 randomized-demand cmTSP {n_nodes}")
                bt.logging.info(f"dataset ref {dataset_ref} selected from {list(self.loaded_datasets.keys())}" )
                selected_node_idxs = random.sample(range(len(self.loaded_datasets[dataset_ref]['data'])), n_nodes)
                m = random.randint(2, 10)
                constraint = []
                depots = sorted(random.sample(list(range(n_nodes)), k=m))
                demand = [random.randint(1, 9) for _ in range(n_nodes)]
                for depot in depots:
                    demand[depot] = 0
                while sum(demand) > sum(constraint):
                    total_demand_padded = sum(demand) + 9*m # padded to prevent invalid knap-sack problem conditions
                    constraint = [(math.ceil(total_demand_padded/m) + random.randint(0, int(total_demand_padded/m * 0.3)) - random.randint(0, int(total_demand_padded/m * 0.2))) for _ in range(m-1)]
                    constraint += [(math.ceil(total_demand_padded/m) + random.randint(0, int(total_demand_padded/m * 0.3)) - random.randint(0, int(total_demand_padded/m * 0.2)))] if sum(constraint) > total_demand_padded - (math.ceil(total_demand_padded/m) - int(total_demand_padded/m * 0.2)) else [(total_demand_padded - sum(constraint) + random.randint(int(total_demand_padded/m * 0.2), int(total_demand_padded/m * 0.3)))]
                test_problem_obj = GraphV2ProblemMultiConstrained(problem_type="Metric cmTSP", 
                                                        n_nodes=n_nodes, 
                                                        selected_ids=selected_node_idxs, 
                                                        cost_function="Geom", 
                                                        dataset_ref=dataset_ref, 
                                                        n_salesmen=m, 
                                                        depots=depots, 
                                                        single_depot=False,
                                                        demand=demand,
                                                        constraint=constraint)
                
                ## Run greedy to make sure there is a valid solution before we send out the problem
                test_problem_obj.edges = self.recreate_edges(test_problem_obj)
                solver1 = NearestNeighbourMultiSolver4(problem_types=[test_problem_obj])
                async def main(timeout, test_problem_obj):
                    try:
                        route1 = await asyncio.wait_for(solver1.solve_problem(test_problem_obj), timeout=timeout)
                        return route1
                    except asyncio.TimeoutError:
                        print(f"Solver1 timed out after {timeout} seconds")
                        return None  # Handle timeout case as needed
                route1 = await main(10, test_problem_obj)
                if route1 != None:
                    solution_found = True
                    test_problem_obj.edges = None
            bt.logging.info(f"Posted: n_nodes V2 randomized-demand cmTSP {n_nodes}")
    else:
        solution_found = False
        connection = False
        while not connection:
            try:
                subtensor = bt.Subtensor("finney")
                connection = True
            except:
                pass
        subnets_info = subtensor.all_subnets()
        while not solution_found:

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

            test_problem_obj = GraphV1PortfolioProblem(problem_type="PortfolioReallocation", 
                                            n_portfolio=num_portfolio, 
                                            initialPortfolios=initialPortfolios, # [[50, 200, 0], [50, 0, 300]], 
                                            constraintValues=constraintValues, #[50, 25, 25], 
                                            constraintTypes=constraintTypes, #["eq", "eq", "eq"], 
                                            pools=pools)
            
            solver1 = GreedyPortfolioSolver(problem_types=[test_problem_obj])
            async def main(timeout, test_problem_obj):
                try:
                    swaps = await asyncio.wait_for(solver1.solve_problem(test_problem_obj), timeout=timeout)
                    return swaps
                except asyncio.TimeoutError:
                    print(f"Solver1 timed out after {timeout} seconds")
                    return None  # Handle timeout case as needed
            swaps = await main(10, test_problem_obj)

            test_synapse = GraphV1PortfolioSynapse(problem = test_problem_obj, solution = swaps)
            swap_count, objective_score = get_portfolio_distribution_similarity(test_synapse)
            
            if swaps != None:
                if swaps != False:
                    if len(swaps) > 0 and swap_count != 1000000 and objective_score != 0:
                        solution_found = True
        bt.logging.info(f"Posted: V1 portfolio_allocation ports: {num_portfolio}, subnets: {num_subnets}")
            
    try:
        if isinstance(test_problem_obj, GraphV1PortfolioProblem):
            graphsynapse_req = GraphV1PortfolioSynapse(problem=test_problem_obj)
            bt.logging.info(f"GraphV1PortfolioSynapse {graphsynapse_req.problem.problem_type}, num_portfolio: {graphsynapse_req.problem.n_portfolio}\n")
        else:
            graphsynapse_req = GraphV2Synapse(problem=test_problem_obj)
            if "mTSP" in graphsynapse_req.problem.problem_type:
                bt.logging.info(f"GraphV2Synapse {graphsynapse_req.problem.problem_type}, n_nodes: {graphsynapse_req.problem.n_nodes}, depots: {graphsynapse_req.problem.depots}\n")
            else:
                bt.logging.info(f"GraphV2Synapse {graphsynapse_req.problem.problem_type}, n_nodes: {graphsynapse_req.problem.n_nodes}\n")
    except ValidationError as e:
        bt.logging.debug(f"GraphV2Synapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)


    # available_uids = await self.get_available_uids()
    
    if len(api_response_output) > 0:
        # if this is an organic request, we select the top k miners by incentive (with a mix of some outside the top k to increase solution diversity)
        selected_uids = await self.get_top_k_uids()
    else:
        # select random 30 miners that are available (i.e. responded to the isAlive synapse)
        selected_uids = await self.get_k_uids()
    # selected_uids = await self.get_available_uids()

    miner_uids = list(selected_uids.keys())
    bt.logging.info(f"Selected UIDS: {miner_uids}")


    if isinstance(test_problem_obj, GraphV2Problem):
        reconstruct_edge_start_time = time.time()
        edges = self.recreate_edges(test_problem_obj)
        reconstruct_edge_time = time.time() - reconstruct_edge_start_time

        bt.logging.info(f"synapse type {type(graphsynapse_req)}")
        # The dendrite client queries the network.
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids], #miner_uids
            synapse=graphsynapse_req,
            deserialize=True,
            timeout = 30 + reconstruct_edge_time, # 30s + time to reconstruct, can scale with problem types in the future
        )

        test_problem_obj.edges = edges
    else:
        # portfolio problem
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids], #miner_uids
            synapse=graphsynapse_req,
            deserialize=True,
            timeout = 0.0007 * test_problem_obj.n_portfolio * len(test_problem_obj.constraintTypes),
        )

    for idx, res in enumerate(responses):
        # trace log the process times
        bt.logging.trace(f"Miner {miner_uids[idx]} status code: {res.dendrite.status_code}, process_time: {res.dendrite.process_time}")

    bt.logging.info(f"NUMBER OF RESPONSES: {len(responses)}")

    if isinstance(test_problem_obj, GraphV2Problem):
        graphsynapse_req_updated = GraphV2Synapse(problem=test_problem_obj) # reconstruct with edges
        score_response_obj = ScoreResponse(graphsynapse_req_updated)

        score_response_obj.current_num_concurrent_forwards = self.current_num_concurrent_forwards

        try:
            await score_response_obj.get_benchmark()
        except:
            try:
                await score_response_obj.get_benchmark()
            except:
                await score_response_obj.get_benchmark()

        rewards = get_rewards(self, score_handler=score_response_obj, responses=responses)
        rewards = rewards.numpy(force=True)

        wandb_miner_distance = [np.inf for _ in range(self.metagraph.n.item())]
        wandb_miner_solution = [[] for _ in range(self.metagraph.n.item())]
        wandb_axon_elapsed = [np.inf for _ in range(self.metagraph.n.item())]
        wandb_rewards = [0 for _ in range(self.metagraph.n.item())]
        best_solution_uid = 0
        for id, uid in enumerate(miner_uids):
            wandb_rewards[uid] = rewards[id]
            if wandb_rewards[uid] == rewards.max():
                best_solution_uid = uid
            wandb_miner_distance[uid] = score_response_obj.score_response(responses[id]) if score_response_obj.score_response(responses[id])!=None else 0
            wandb_miner_solution[uid] = responses[id].solution
            wandb_axon_elapsed[uid] = responses[id].dendrite.process_time
    else:
        graphsynapse_req_updated = GraphV1PortfolioSynapse(problem=test_problem_obj)
        score_response_obj = ScorePortfolioResponse(graphsynapse_req_updated, swaps)

        score_response_obj.current_num_concurrent_forwards = self.current_num_concurrent_forwards

        rewards = get_portfolio_rewards(self, score_handler=score_response_obj, responses=responses)
        rewards = rewards.numpy(force=True)

        wandb_miner_swaps = [np.inf for _ in range(self.metagraph.n.item())]
        wandb_miner_objective = [np.inf for _ in range(self.metagraph.n.item())]
        wandb_miner_solution = [[] for _ in range(self.metagraph.n.item())]
        wandb_axon_elapsed = [np.inf for _ in range(self.metagraph.n.item())]
        wandb_rewards = [0 for _ in range(self.metagraph.n.item())]
        best_solution_uid = 0
        for id, uid in enumerate(miner_uids):
            wandb_rewards[uid] = rewards[id]
            if wandb_rewards[uid] == rewards.max():
                best_solution_uid = uid
            wandb_miner_swaps[uid] = score_response_obj.score_response(responses[id])[0] if score_response_obj.score_response(responses[id])!=None else 0
            wandb_miner_objective[uid] = score_response_obj.score_response(responses[id])[1] if score_response_obj.score_response(responses[id])!=None else 0
            wandb_miner_solution[uid] = responses[id].solution
            wandb_axon_elapsed[uid] = responses[id].dendrite.process_time


    # if len(responses) > 0 and did_organic_task == True:
    #     try:
    #         # url = f"{organic_endpoint}/pop_organic_task"
    #         # headers = {"Authorization": "Bearer db_bearer_token"}
    #         # api_response = requests.get(url, headers=headers)

    #         best_reward_idx = np.argmax(wandb_rewards)

    #         data = {
    #             "solution": wandb_miner_solution[best_reward_idx], 
    #             "distance": wandb_miner_distance[best_reward_idx]
    #         }
    #         url = f"{organic_endpoint}/tasks/{organic_task_id}"
    #         headers = {"Authorization": "Bearer db_bearer_token"}
    #         api_response = requests.put(url, json=data, headers=headers)

    #         did_organic_task = False
    #     except:
    #         pass

    # # clear database of old request > 10mins, both solved and unsolved
    # try:
    #     url = f"{organic_endpoint}/tasks/oldest"
    #     headers = {"Authorization": "Bearer db_bearer_token"}
    #     api_response = requests.delete(url, headers=headers)
    # except:
    #     pass

    configDict = {
                    "save_code": False,
                    "log_code": False,
                    "save_model": False,
                    "log_model": False,
                    "sync_tensorboard": False,
                }
    
    
    if isinstance(test_problem_obj, GraphV2Problem):
        try:
            configDict["problem_type"] = graphsynapse_req.problem.problem_type
        except:
            pass
        try:
            configDict["objective_function"] = graphsynapse_req.problem.objective_function
        except:
            pass
        try:
            configDict["visit_all"] = graphsynapse_req.problem.visit_all
        except:
            pass
        try:
            configDict["to_origin"] = graphsynapse_req.problem.to_origin
        except:
            pass
        try:
            configDict["n_nodes"] = graphsynapse_req.problem.n_nodes
        except:
            pass
        try:
            configDict["nodes"] = graphsynapse_req.problem.nodes
        except:
            pass
        try:
            configDict["edges"] = []
        except:
            pass
        try:
            configDict["directed"] = graphsynapse_req.problem.directed
        except:
            pass
        try:
            configDict["simple"] = graphsynapse_req.problem.simple
        except:
            pass
        try:
            configDict["weighted"] = graphsynapse_req.problem.weighted
        except:
            pass
        try:
            configDict["repeating"] = graphsynapse_req.problem.repeating
        except:
            pass
        try:
            configDict["selected_ids"] = graphsynapse_req.problem.selected_ids
        except:
            pass
        try:
            configDict["cost_function"] = graphsynapse_req.problem.cost_function
        except:
            pass
        try:
            configDict["dataset_ref"] = graphsynapse_req.problem.dataset_ref
        except:
            pass
        try:
            configDict["selected_uids"] = miner_uids
        except:
            pass
        try:
            configDict["n_salesmen"] = graphsynapse_req.problem.n_salesmen
            configDict["depots"] = graphsynapse_req.problem.depots
        except:
            pass
        try:
            configDict["demand"] = graphsynapse_req.problem.demand
            configDict["constraint"] = graphsynapse_req.problem.constraint
        except:
            pass

        try:
            configDict["time_elapsed"] = wandb_axon_elapsed
        except:
            pass

        try:
            configDict["best_solution"] = wandb_miner_solution[best_solution_uid]
        except:
            pass
        
        try:
            if self.subtensor.network == "test":
                wandb.init(
                    entity='graphite-subnet',
                    project="graphite-testnet",
                    config=configDict,
                    name=json.dumps({
                        "n_nodes": graphsynapse_req.problem.n_nodes,
                        "time": time.time(),
                        "validator": self.wallet.hotkey.ss58_address,
                        }),
                )
            else:
                wandb.init(
                    entity='graphite-ai',
                    project="Graphite-Subnet-V2",
                    config=configDict,
                    name=json.dumps({
                        "n_nodes": graphsynapse_req.problem.n_nodes,
                        "time": time.time(),
                        "validator": self.wallet.hotkey.ss58_address,
                        }),
                )
            for rewIdx in range(self.metagraph.n.item()):
                wandb.log({f"rewards-{self.wallet.hotkey.ss58_address}": wandb_rewards[rewIdx], f"distance-{self.wallet.hotkey.ss58_address}": wandb_miner_distance[rewIdx]}, step=int(rewIdx))

            self.cleanup_wandb(wandb)
        except Exception as e:
            print(f"Error initializing W&B: {e}")
    else:
        try:
            configDict["problem_type"] = graphsynapse_req.problem.problem_type
        except:
            pass
        try:
            configDict["initialPortfolios"] = graphsynapse_req.problem.initialPortfolios
        except:
            pass
        try:
            configDict["constraintValues"] = graphsynapse_req.problem.constraintValues
        except:
            pass
        try:
            configDict["constraintTypes"] = graphsynapse_req.problem.constraintTypes
        except:
            pass
       
        try:
            if self.subtensor.network == "test":
                wandb.init(
                    entity='graphite-subnet',
                    project="graphite-testnet",
                    config=configDict,
                    name=json.dumps({
                        "n_portfolio": graphsynapse_req.problem.n_portfolio,
                        "time": time.time(),
                        "validator": self.wallet.hotkey.ss58_address,
                        }),
                )
            else:
                wandb.init(
                    entity='graphite-ai',
                    project="Graphite-Subnet-V2",
                    config=configDict,
                    name=json.dumps({
                        "n_portfolio": graphsynapse_req.problem.n_portfolio,
                        "time": time.time(),
                        "validator": self.wallet.hotkey.ss58_address,
                        }),
                )
            for rewIdx in range(self.metagraph.n.item()):
                wandb.log({f"rewards-{self.wallet.hotkey.ss58_address}": wandb_rewards[rewIdx], f"swaps-{self.wallet.hotkey.ss58_address}": wandb_miner_swaps[rewIdx], f"objective-{self.wallet.hotkey.ss58_address}": wandb_miner_objective[rewIdx]}, step=int(rewIdx))

            self.cleanup_wandb(wandb)
        except Exception as e:
            print(f"Error initializing W&B: {e}")

    
    bt.logging.info(f"Scored responses: {rewards}")
    
    
    if len(rewards) > 0 and max(rewards) == 1:
        self.update_scores(rewards, miner_uids, ScoreType.SYNTHETIC)
        time.sleep(16) # for each block, limit 1 request per block
    elif max(rewards) == 0.2:
        new_rewards = []
        new_miner_uids = []
        for i in range(len(rewards)):
            if rewards[i] != 0.2:
                new_rewards.append(0)
                new_miner_uids.append(miner_uids[i])
        new_rewards = np.array(new_rewards)  # Creates (N,)
        if len(new_miner_uids) > 0:
            self.update_scores(new_rewards, new_miner_uids, ScoreType.SYNTHETIC)
            time.sleep(16) # for each block, limit 1 request per block
