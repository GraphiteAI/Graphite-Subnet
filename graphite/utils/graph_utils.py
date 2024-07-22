# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2023 Graphite-AI

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

import math
from typing import List, Union
import numpy as np
from graphite.protocol import GraphProblem, GraphSynapse
from functools import wraps, partial
import bittensor as bt
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
import time

### Generic functions for neurons
def is_valid_path(path:List[int])->bool:
    # a valid path should have at least 3 return values and return to the source
    return (len(path)>=3) and (path[0]==path[-1])

def get_tour_distance(synapse:GraphSynapse)->float:
    '''
    Returns the total tour distance for the TSP or graph-traversal problem as a float.

    Takes a synapse as its only argument
    '''
    problem = synapse.problem
    if 'TSP' not in problem.problem_type:
        raise ValueError(f"get_tour_distance is an invalid function for processing {problem.problem_type}")
    
    if not synapse.solution:
        return np.inf
    distance=np.nan
    if problem.directed:
        # This is a General TSP problem
        # check if path and edges are of the appropriate size
        edges=problem.edges
        path=synapse.solution
        if isinstance(path,list):
            assert is_valid_path(path), ValueError('Provided path is invalid')
            assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')

            distance = 0
            for i, source in enumerate(path[:-1]):
                destination = path[i+1]
                distance += edges[source][destination]
    else:
        # This is a metric TSP problem
        # check if path and coordinates are of the appropriate size
        coordinates=problem.nodes
        path=synapse.solution
        
        if isinstance(path,list):
            assert is_valid_path(path), ValueError('Provided path is invalid')
            assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')

            # sort cities into pairs
            pairs = [(path[i], path[i+1]) for i in range(len(path)-1)]
            distance = 0
            for pair in pairs:
                distance += math.hypot(coordinates[pair[0]][0] - coordinates[pair[1]][0], coordinates[pair[0]][1] - coordinates[pair[1]][1])
    return distance if not np.isnan(distance) else np.inf

def normalize_coordinates(coordinates:List[List[Union[int,float]]]):
    '''
    Normalizes all coordinates against the max x or y value.

    Normalized coordinate values are required my some open-source algorithms/models.

    Assumes all coordinates are non-negative (in line with the current GraphProblem Formulation)
    '''
    coordinate_arr = np.array(coordinates)
    max_val = 0
    if np.max(coordinate_arr) != np.inf:
        max_val = np.max(coordinate_arr)
        assert max_val>0, ValueError("Received coordinates where the max value is not positive")
        return coordinate_arr / max_val
    else:
        # get largest non inf value
        coordinate_arr[coordinate_arr == np.inf] = np.nan
        max_finite_value = np.nanmax(coordinate_arr)
        # print(f'Maximum finite value of: {max_finite_value}')

        # Get the indices of NaN values
        nan_indices = np.isnan(coordinate_arr)

        # Generate random numbers close to 0, around the order of 10^-8
        random_values = 1 - 10**-8 * np.random.random(np.sum(nan_indices))

        # divide all coordinates by this value
        if max_finite_value:
            max_ratio = sys.float_info.max/max_finite_value
            normalized_coordinates = np.array(coordinate_arr) / (max_finite_value*min(10, max_ratio))
        else:
            normalized_coordinates = coordinate_arr
        normalized_coordinates[nan_indices] = random_values
    return normalized_coordinates

def generate_city_coordinates(n_cities, grid_size=1000):
    '''
    Helper function for generating random coordinates for MetricTSP problems.
    '''
    # Create a grid of coordinates
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xv, yv = np.meshgrid(x, y)
    
    coordinates = np.column_stack((xv.ravel(), yv.ravel()))
    
    # Sample n_cities coordinates from the grid
    sampled_indices = np.random.choice(coordinates.shape[0], n_cities, replace=False)
    sampled_coordinates = coordinates[sampled_indices]
    np.random.shuffle(sampled_coordinates)

    del coordinates

    sampled_coordinates_list = sampled_coordinates.tolist()
    sampled_coordinates_list = [[int(coord[0]), int(coord[1])] for coord in sampled_coordinates_list]
    
    return sampled_coordinates

def check_nodes(solution:List[int], n_cities:int):
    return not set(solution).difference(set(list(range(n_cities)))) and not set(list(range(n_cities))).difference(set(solution))

def start_and_end(solution:List[int]):
    return solution[0] == solution[-1]

def is_valid_solution(problem:GraphProblem, solution:List[int]):
    if solution == None:
        return False
    if isinstance(solution, bool):
        return False
    if problem.to_origin == True:
        if problem.visit_all == True:
            return check_nodes(solution, problem.n_nodes) and start_and_end(solution)
        else:
            return start_and_end(solution)
    else:
        if problem.visit_all == True:
            return check_nodes(solution, problem.n_nodes)
        else:
            return True

def valid_problem(problem:GraphProblem)->bool:
    if problem.problem_type == 'Metric TSP':
        if (problem.directed==False) and (problem.visit_all==True) and (problem.to_origin==True) and (problem.objective_function=='min'):
            return True
        else:
            bt.logging.info(f"Received an invalid Metric TSP problem")
            bt.logging.info(problem.get_info(verbosity=2))
            return False
        
    elif problem.problem_type == 'General TSP':
        if (problem.directed==True) and (problem.visit_all==True) and (problem.to_origin==True) and (problem.objective_function=='min'):
            return True
        else:
            bt.logging.info(f"Received an invalid General TSP problem")
            bt.logging.info(problem.get_info(verbosity=2))
            return False
        
    else:
        bt.logging.info(f"Received an invalid problem of type: {problem.problem_type}")
        return False

def timeout(seconds=30, error_message="Solver timed out"):
    '''
    basic implementation of async function timeout.
    '''
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                return result
            except asyncio.TimeoutError:
                bt.logging.info(f"{error_message} at {seconds} seconds")
                print(f"{error_message} at {seconds} seconds with elapsed time {time.time()-start_time}")
                return False
        return async_wrapper
    return decorator
