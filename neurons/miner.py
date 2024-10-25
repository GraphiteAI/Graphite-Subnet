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

import time
from typing import Tuple, Union
import bittensor as bt

# Bittensor Miner Template:
import graphite

# import base miner class which takes care of most of the boilerplate
from graphite.base.miner import BaseMinerNeuron
from graphite.protocol import IsAlive

from graphite.solvers import NearestNeighbourSolver, DPSolver, NearestNeighbourMultiSolver, NearestNeighbourMultiSolver2, InsertionMultiSolver
from graphite.protocol import GraphV2Problem, GraphV1Synapse, GraphV2Synapse, GraphV2ProblemMulti
from graphite.utils.graph_utils import get_multi_minmax_tour_distance

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.axon.attach(
            forward_fn=self.is_alive,
            blacklist_fn=self.blacklist_is_alive,
        ).attach(
            forward_fn=self.forwardV2,
            blacklist_fn=self.blacklistV2,
            priority_fn=self.priorityV2,
        ).attach(
            forward_fn=self.forwardV1,
            blacklist_fn=self.blacklistV1,
            priority_fn=self.priorityV1,
        )

        self.solvers = {
            'small': DPSolver(),
            'large': NearestNeighbourSolver(),
            'multi_large_1': NearestNeighbourMultiSolver(),
            'multi_large_2': NearestNeighbourMultiSolver2(),
            'multi_large_3': InsertionMultiSolver()
        }
    
    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        # hotkey = self.wallet.hotkey.ss58_address
        # dend_hotkey = synapse.dendrite.hotkey
        # log_line = f"{hotkey[:5]}_{dend_hotkey[:5]}_{time.time()}\n"
        # with open("is_alive_logs.txt","a") as f:
        #     f.write(log_line)
        bt.logging.debug("Answered to be alive")
        synapse.completion = "True"
        return synapse
    
    def blacklist_is_alive( self, synapse: IsAlive ) -> Tuple[bool, str]:
        # TODO: implement proper blacklist logic for is_alive
        return False, "NaN"

    async def forward(
        self, synapse: Union[GraphV1Synapse, GraphV2Synapse]
    ) ->  Union[GraphV1Synapse, GraphV2Synapse]:
        """
        Processes the incoming 'Dummy' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.Dummy): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.Dummy: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        bt.logging.info(f"received synapse with problem: {synapse.problem.get_info(verbosity=2)}")
        
        bt.logging.info(
            f"Miner received input to solve {synapse.problem.n_nodes}"
        )
        
        if isinstance(synapse.problem, GraphV2Problem):
            # recreate problem edges for both the GraphV2Problem and GraphV2ProblemMulti which inherits from GraphV2Problem
            synapse.problem.edges = self.recreate_edges(synapse.problem)
        
        bt.logging.info(f"synapse dendrite timeout {synapse.timeout}")

        # Conditional assignment of problems to each solver
        if not isinstance(synapse.problem, GraphV2ProblemMulti):
            if synapse.problem.n_nodes < 15:
                # Solves the problem to optimality but is very computationally intensive
                route = await self.solvers['small'].solve_problem(synapse.problem)
            else:
                # Simple heuristic that does not guarantee optimality. 
                route = await self.solvers['large'].solve_problem(synapse.problem)
            synapse.solution = route
        else:
            routes = await self.solvers['multi_large'].solve_problem(synapse.problem)
            synapse.solution = routes
        
        bt.logging.info(
            f"Miner returned value {synapse.solution} {len(synapse.solution) if isinstance(synapse.solution, list) else synapse.solution}"
        )
        return synapse

    async def forwardV1(
        self, synapse: GraphV1Synapse
    ) ->  GraphV1Synapse:
        """
        Processes the incoming 'Dummy' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.Dummy): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.Dummy: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        bt.logging.info(f"received synapse with problem: {synapse.problem.get_info(verbosity=2)}")

        # hotkey = self.wallet.hotkey.ss58_address
        # dend_hotkey = synapse.dendrite.hotkey
        # log_line = f"{hotkey[:5]}_{dend_hotkey[:5]}_{synapse.problem.n_nodes}_{time.time()}\n"
        # with open("gs_logs.txt","a") as f:
        #     f.write(log_line)

        bt.logging.info(
            f"Miner received input to solve {synapse.problem.n_nodes}"
        )
        
        if isinstance(synapse.problem, GraphV2Problem):
            synapse.problem.edges = self.recreate_edges(synapse.problem)
        
        bt.logging.info(f"synapse dendrite timeout {synapse.timeout}")

        # Conditional assignment of problems to each solver
        if synapse.problem.n_nodes < 15:
            # Solves the problem to optimality but is very computationally intensive
            route = await self.solvers['small'].solve_problem(synapse.problem)
        else:
            # Simple heuristic that does not guarantee optimality. 
            route = await self.solvers['large'].solve_problem(synapse.problem)
        synapse.solution = route
        
        bt.logging.info(
            f"Miner returned value {synapse.solution} {len(synapse.solution) if isinstance(synapse.solution, list) else synapse.solution}"
        )
        return synapse

    async def forwardV2(
        self, synapse: GraphV2Synapse
    ) ->  GraphV2Synapse:
        """
        Processes the incoming 'Dummy' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.Dummy): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.Dummy: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        bt.logging.info(f"received synapse with problem: {synapse.problem.get_info(verbosity=2)}")

        # hotkey = self.wallet.hotkey.ss58_address
        # dend_hotkey = synapse.dendrite.hotkey
        # log_line = f"{hotkey[:5]}_{dend_hotkey[:5]}_{synapse.problem.n_nodes}_{time.time()}\n"
        # with open("gs_logs.txt","a") as f:
        #     f.write(log_line)
        
        bt.logging.info(
            f"Miner received input to solve {synapse.problem.n_nodes}"
        )

        if isinstance(synapse.problem, GraphV2Problem):
            synapse.problem.edges = self.recreate_edges(synapse.problem)
        
        bt.logging.info(f"synapse dendrite timeout {synapse.timeout}")

        # Conditional assignment of problems to each solver
        if not isinstance(synapse.problem, GraphV2ProblemMulti):
            route = await self.solvers['large'].solve_problem(synapse.problem)
            synapse.solution = route
        else:
            # run all 3 basic algorithms and return the best scoring solution
            routes_1 = await self.solvers['multi_large_1'].solve_problem(synapse.problem)
            synapse.solution = routes_1
            score_1 = get_multi_minmax_tour_distance(synapse)
            routes_2 = await self.solvers['multi_large_2'].solve_problem(synapse.problem)
            synapse.solution = routes_2
            score_2 = get_multi_minmax_tour_distance(synapse)
            routes_3 = await self.solvers['multi_large_3'].solve_problem(synapse.problem)
            synapse.solution = routes_3
            score_3 = get_multi_minmax_tour_distance(synapse)
            routes = [routes_1, routes_2, routes_3]
            scores = [score_1, score_2, score_3]
            bt.logging.info(f"Selecting algorithm {scores.index(min(scores))}")
            synapse.solution = routes[scores.index(min(scores))]

        # empty out large distance matrix
        synapse.problem.edges = None
        
        bt.logging.info(
            f"Miner returned value {synapse.solution} {len(synapse.solution) if isinstance(synapse.solution, list) else synapse.solution}"
        )
        return synapse
    
    async def blacklist(
        self, synapse: Union[GraphV1Synapse, GraphV2Synapse]
    ) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def blacklistV1(
        self, synapse: GraphV1Synapse
    ) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def blacklistV2(
        self, synapse: GraphV2Synapse
    ) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        blacklisted = ["5HNQURvmjjYhTSksi8Wfsw676b4owGwfLR2BFAQzG7H3HhYf"]
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid] or self.metagraph.S[uid] < 2000 or synapse.dendrite.hotkey in blacklisted:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: Union[GraphV1Synapse, GraphV2Synapse]) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority

    async def priorityV1(self, synapse: GraphV1Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority

    async def priorityV2(self, synapse: GraphV2Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        start_time = time.time()
        while True:
            if time.time() - start_time >= 100:
                bt.logging.info(f"Miner running... {time.time()}")
                start_time = time.time()
            time.sleep(5)
