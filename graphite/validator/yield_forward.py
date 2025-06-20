import bittensor as bt
from bittensor import axon, dendrite

from graphite.validator.reward import ScoreYieldResponse

import time
from datetime import datetime

from graphite.base.validator import ScoreType, BaseValidatorNeuron
from graphite.yield_protocol import YieldDataRequestSynapse, MinerYield

import copy
from typing import List, Union, Optional
from pydantic import ValidationError
import asyncio
import numpy as np

async def fetch_yield_data(self: BaseValidatorNeuron):
    def create_empty_yield(hotkey: str, uid: int):
        return MinerYield(hotkey=hotkey, uid=uid, yield_data=None)
    miner_uids = await self.get_miner_uids()
    yields = [create_empty_yield(self.metagraph.hotkeys[uid], uid) for uid in miner_uids]
    request = YieldDataRequestSynapse(yields=yields)
    axons = [self.yield_axon]
    bt.logging.info(f"Axons: {axons}")
    responses = await self.dendrite(
        axons=axons,
        synapse=request,
        deserialize=True,
        timeout=12
    )
    response = responses[0]
    if not isinstance(response, YieldDataRequestSynapse) or len([miner_yield for miner_yield in response.yields if miner_yield.yield_data is not None]) == 0:
        return None
    return response


async def yield_forward(self: BaseValidatorNeuron):
    # Get the organic portfolio problem
    yield_synapse: Optional[YieldDataRequestSynapse] = await fetch_yield_data(self)    
    if yield_synapse is None:
        return
    bt.logging.info(f"Yield synapse with {len([miner_yield for miner_yield in yield_synapse.yields if miner_yield.yield_data is not None])} valid yields")

    score_handler = ScoreYieldResponse(mock_synapse=yield_synapse)
    miner_uids = [miner_yield.uid for miner_yield in yield_synapse.yields]
    rewards = score_handler.get_rewards(yield_synapse)

    for idx, yield_uid in enumerate(yield_synapse.yields):
        if yield_uid.yield_data is None:
            rewards[idx] = np.nan

    # Update the scores
    self.update_scores(rewards, miner_uids, ScoreType.YIELD)
