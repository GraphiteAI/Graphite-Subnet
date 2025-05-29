import requests
import bittensor as bt
from typing import Optional
from .constants import DEFAULT_REBALANCING_INFO_ENDPOINT, DEFAULT_PERFORMANCE_INFO_ENDPOINT

def fetch_rebalancing_axon() -> Optional[bt.AxonInfo]:
    # Fetch the yield axon from the yield endpoint
    response = requests.get(f"{DEFAULT_REBALANCING_INFO_ENDPOINT}", timeout=10)
    if response.status_code != 200:
        bt.logging.error(f"Failed to fetch rebalancing axon from {DEFAULT_REBALANCING_INFO_ENDPOINT}")
        return None
    return bt.AxonInfo(**response.json())

def fetch_performance_axon() -> Optional[bt.AxonInfo]:
    # Fetch the yield axon from the yield endpoint
    response = requests.get(f"{DEFAULT_PERFORMANCE_INFO_ENDPOINT}", timeout=10)
    if response.status_code != 200:
        bt.logging.error(f"Failed to fetch performance axon from {DEFAULT_PERFORMANCE_INFO_ENDPOINT}")
        return None
    return bt.AxonInfo(**response.json())