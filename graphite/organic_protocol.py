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

from typing import List, Union, Optional
import bittensor as bt
from .protocol import GraphV1PortfolioProblem

class OrganicPortfolioRequestSynapse(bt.Synapse):
    # Sends out NULL problem and receives actual problem as a response | For now, we only accept GraphV1PortfolioProblem
    problem: Optional[GraphV1PortfolioProblem] = None 
    job_id: Optional[str] = None # hash of the job

class OrganicPortfolioResponseSynapse(bt.Synapse):
    # sends out synapse to trigger rebalancing on yield_server
    problem: GraphV1PortfolioProblem
    solution: Optional[Union[List[List[Union[int, float]]], bool]] = None
    accepted: bool = False
    job_id: Optional[str] = None # hash of the job