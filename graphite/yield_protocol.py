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

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import bittensor as bt
from bittensor.utils import is_valid_ss58_address

class LeaderPerformanceData(BaseModel):
    historical_daily_pnl: List[float] = Field(default_factory=list) # ordered by time
    sharpe_ratio: float
    max_drawdown: float
    num_copy_traders: int
    volume: int # volume in RAO of attributed rebalances
    notional_value_of_copy_traders: float

class MinerYield(BaseModel):
    uid: int
    hotkey: str
    yield_data: Optional[LeaderPerformanceData] = None

    @field_validator('hotkey')
    def validate_hotkey(cls, v):
        if not is_valid_ss58_address(v):
            raise ValueError("Invalid hotkey")
        return v

class YieldDataRequestSynapse(bt.Synapse):
    # Sends out NULL problem and receives actual problem as a response | For now, we only accept GraphV1PortfolioProblem
    yields: List[MinerYield] = Field(default_factory=list)
