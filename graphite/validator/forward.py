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

from graphite.protocol import GraphProblem, GraphSynapse
from graphite.dataset.dataset_generator import MetricTSPGenerator, GeneralTSPGenerator
        
import numpy as np
import json
import base64

import wandb
import os

import random
import requests
import asyncio

from pydantic import ValidationError

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
    # try:
    #     if organic_endpoint is not None and organic_endpoint != "":

    #         url = f"{organic_endpoint}/tasks/oldest/{curr_idx}"
    #         headers = {"Authorization": "Bearer %s"%db_bearer_token}
    #         api_response = requests.get(url, headers=headers)
    #         api_response_output = api_response.json()
            
    #         organic_task_id = api_response_output["_id"]
    #         del api_response_output["_id"]

    #         did_organic_task = True

    #         # bt.logging.info(f"ORGANIC TRAFFIC {api_response.text}")
    #     else:
    #         api_response_output = []
    # except:
    #     api_response_output = []

    # if len(api_response_output) > 0:
    #     try:
    #         test_problem_obj = GraphProblem(**api_response_output)
    #     except Exception as e:
    #         bt.logging.error(e)
            
    #         prob_select = random.randint(1, 2)
    
    #         if prob_select == 1:
    #             problems, sizes = MetricTSPGenerator.generate_and_save_dataset(1)
    #             test_problem_obj = problems[0]
    #         else:
    #             problems, sizes = GeneralTSPGenerator.generate_and_save_dataset(1)
    #             test_problem_obj = problems[0]
                
    #         try:
    #             url = f"{organic_endpoint}/tasks/{organic_task_id}"
    #             headers = {"Authorization": "Bearer db_bearer_token"}
    #             api_response = requests.delete(url, headers=headers)

    #             did_organic_task = False
    #         except:
    #             pass
    # else:

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
        bt.logging.debug(f"GraphProblem Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)

    try:
        graphsynapse_req = GraphSynapse(problem=test_problem_obj)
    except ValidationError as e:
        bt.logging.debug(f"GraphSynapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)

    # prob_select = random.randint(1, 2)
    
    # if prob_select == 1:
    #     problems, sizes = MetricTSPGenerator.generate_and_save_dataset(1)
    #     test_problem_obj = problems[0]
    # else:
    #     problems, sizes = GeneralTSPGenerator.generate_and_save_dataset(1)
    #     test_problem_obj = problems[0]

    # graphsynapse_req = GraphSynapse(problem=test_problem_obj)
    bt.logging.info(f"Graph Synapse Problem, n_nodes: {graphsynapse_req.problem.n_nodes}")

    available_uids = await self.get_available_uids()

    miner_uids = list(available_uids.keys())

    # The dendrite client queries the network.
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids], #miner_uids
        synapse=graphsynapse_req,
        deserialize=True,
        timeout = 30, # can scale with problem types in the future
    )

    for res in responses:
        try:
            if res.axon.status_code != None:
                res.axon.process_time = res.dendrite.process_time
                # bt.logging.info(f"Received responses axon: {res.axon} {res.solution}")
        except:
            pass
    bt.logging.info(f"NUMBER OF RESPONSES: {len(responses)}")

    score_response_obj = ScoreResponse(graphsynapse_req)

    score_response_obj.current_num_concurrent_forwards = self.current_num_concurrent_forwards
    await score_response_obj.get_benchmark()

    rewards = get_rewards(self, score_handler=score_response_obj, responses=responses)
    rewards = rewards.numpy(force=True)

    wandb_miner_distance = [np.inf for _ in range(self.metagraph.n.item())]
    wandb_miner_solution = [[] for _ in range(self.metagraph.n.item())]
    wandb_rewards = [0 for _ in range(self.metagraph.n.item())]
    for id, uid in enumerate(miner_uids):
        wandb_rewards[uid] = rewards[id]
        wandb_miner_distance[uid] = score_response_obj.score_response(responses[id]) if score_response_obj.score_response(responses[id])!=None else 0
        wandb_miner_solution[uid] = responses[id].solution

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
        configDict["edges"] = graphsynapse_req.problem.edges
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
            wandb.log({f"rewards-{self.wallet.hotkey.ss58_address}": wandb_rewards[rewIdx], f"distance-{self.wallet.hotkey.ss58_address}": wandb_miner_distance[rewIdx]})

        self.cleanup_wandb(wandb)
    except Exception as e:
        print(f"Error initializing W&B: {e}")
    
    bt.logging.info(f"Scored responses: {rewards}")
    
    
    if len(rewards) > 0:
        self.update_scores(rewards, miner_uids)
        time.sleep(16) # for each block, limit 1 request per block
