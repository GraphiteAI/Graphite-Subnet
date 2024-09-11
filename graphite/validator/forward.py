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

from graphite.validator.reward import get_rewards, ScoreResponse
from graphite.utils.uids import get_available_uids

import time

from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV1Synapse, GraphV2Synapse
from graphite.data.dataset_generator import MetricTSPGenerator, GeneralTSPGenerator
        
import numpy as np
import json
import base64

import wandb
import os

import random
import requests
import asyncio

from pydantic import ValidationError

REFERENCE_BLOCK = 3793451

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

    start_block = REFERENCE_BLOCK # current test block
    end_block = start_block + 60 * 60 * 24 * 3 / 12 # block 3 days later
    
    bt.logging.info(f"S: {start_block}, E: {end_block}, C: {self.block} @ {(self.block - start_block)/(end_block - start_block)}")
    if random.random() > (self.block - start_block)/(end_block - start_block): # linear shift in distribution
        if len(api_response_output) > 0:
            try:
                test_problem_obj = GraphV1Problem(**api_response_output)
            except Exception as e:
                bt.logging.error(e)
                
                prob_select = random.randint(1, 2)
        
                if prob_select == 1:
                    problems, sizes = MetricTSPGenerator.generate_n_samples(1)
                    test_problem_obj = problems[0]
                else:
                    problems, sizes = GeneralTSPGenerator.generate_n_samples(1)
                    test_problem_obj = problems[0]
                    
                try:
                    url = f"{self.organic_endpoint}/tasks/{organic_task_id}"
                    headers = {"Authorization": "Bearer %s"%self.db_bearer_token}
                    api_response = requests.delete(url, headers=headers)

                    did_organic_task = False
                except:
                    pass
        else:
            prob_select = random.randint(1, 2)
            try:
                if prob_select == 1:
                    problems, sizes = MetricTSPGenerator.generate_n_samples(1)
                    test_problem_obj = problems[0]
                else:
                    problems, sizes = GeneralTSPGenerator.generate_n_samples(1)
                    test_problem_obj = problems[0]
            except ValidationError as e:
                bt.logging.debug(f"{'Metric TSP' if prob_select==1 else 'General TSP'}")
                bt.logging.debug(f"GraphV1Problem Validation Error: {e.json()}")
                bt.logging.debug(e.errors())
                bt.logging.debug(e)

        try:
            graphsynapse_req = GraphV1Synapse(problem=test_problem_obj)
            bt.logging.info(f"GraphV1Synapse Problem, n_nodes: {graphsynapse_req.problem.n_nodes}")
        except ValidationError as e:
            bt.logging.debug(f"GraphV1Synapse Validation Error: {e.json()}")
            bt.logging.debug(e.errors())
            bt.logging.debug(e)
    else:
        # determine the number of nodes to select
        n_nodes = random.randint(2000, 5000)
        # randomly select n_nodes indexes from the selected graph
        prob_select = random.randint(0, len(list(self.loaded_datasets.keys()))-1)
        dataset_ref = list(self.loaded_datasets.keys())[prob_select]
        bt.logging.info(f"n_nodes V2 {n_nodes}")
        bt.logging.info(f"dataset ref {dataset_ref} selected from {list(self.loaded_datasets.keys())}" )
        bt.logging.info(f"dataset length {len(self.loaded_datasets[dataset_ref]['data'])} from {self.loaded_datasets[dataset_ref]['data'].shape} " )
        selected_node_idxs = random.sample(range(len(self.loaded_datasets[dataset_ref]['data'])), n_nodes)
        test_problem_obj = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)

        try:
            graphsynapse_req = GraphV2Synapse(problem=test_problem_obj)
            bt.logging.info(f"GraphV2Synapse Problem, n_nodes: {graphsynapse_req.problem.n_nodes}")
        except ValidationError as e:
            bt.logging.debug(f"GraphV2Synapse Validation Error: {e.json()}")
            bt.logging.debug(e.errors())
            bt.logging.debug(e)


    # prob_select = random.randint(1, 2)
    
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

    reconstruct_edge_start_time = time.time()
    if isinstance(test_problem_obj, GraphV2Problem):
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


    if isinstance(test_problem_obj, GraphV2Problem):
        test_problem_obj.edges = edges
        # with open("gs_logs.txt", "a") as f:
        #     for hotkey in [self.metagraph.hotkeys[uid] for uid in miner_uids]:
        #         f.write(f"{hotkey}_{self.wallet.hotkey.ss58_address}_{edges.shape}_{time.time()}\n")
    for res in responses:
        try:
            if res.axon.status_code != None:
                res.axon.process_time = res.dendrite.process_time
                # bt.logging.info(f"Received responses axon: {res.axon} {res.solution}")
        except:
            pass
    bt.logging.info(f"NUMBER OF RESPONSES: {len(responses)}")

    if isinstance(test_problem_obj, GraphV2Problem):
        graphsynapse_req_updated = GraphV2Synapse(problem=test_problem_obj) # reconstruct with edges
        score_response_obj = ScoreResponse(graphsynapse_req_updated)
    elif isinstance(test_problem_obj, GraphV1Problem):
        score_response_obj = ScoreResponse(graphsynapse_req)

    score_response_obj.current_num_concurrent_forwards = self.current_num_concurrent_forwards
    await score_response_obj.get_benchmark()

    rewards = get_rewards(self, score_handler=score_response_obj, responses=responses)
    rewards = rewards.numpy(force=True)

    wandb_miner_distance = [np.inf for _ in range(self.metagraph.n.item())]
    wandb_miner_solution = [[] for _ in range(self.metagraph.n.item())]
    wandb_axon_elapsed = [np.inf for _ in range(self.metagraph.n.item())]
    wandb_rewards = [0 for _ in range(self.metagraph.n.item())]
    for id, uid in enumerate(miner_uids):
        wandb_rewards[uid] = rewards[id]
        wandb_miner_distance[uid] = score_response_obj.score_response(responses[id]) if score_response_obj.score_response(responses[id])!=None else 0
        wandb_miner_solution[uid] = responses[id].solution
        wandb_axon_elapsed[uid] = responses[id].axon.process_time


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
        configDict["time_elapsed"] = wandb_axon_elapsed
    except:
        pass
    
    if isinstance(test_problem_obj, GraphV1Problem):
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
                    project="Graphite-Subnet",
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
    elif isinstance(test_problem_obj, GraphV2Problem):
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

    
    bt.logging.info(f"Scored responses: {rewards}")
    
    
    if len(rewards) > 0:
        self.update_scores(rewards, miner_uids)
        time.sleep(16) # for each block, limit 1 request per block
