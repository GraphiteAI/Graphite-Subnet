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
from graphite.protocol import GraphV1Problem, GraphV1Synapse, GraphV2Problem, GraphV2Synapse, GraphV2ProblemMulti, GraphV2ProblemMultiConstrained, GraphV1PortfolioSynapse, GraphV1PortfolioProblem
from functools import wraps, partial
import bittensor as bt
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
import time
from graphite.base.subnetPool import SubnetPool

### Generic functions for neurons
def is_valid_path(path:List[int])->bool:
    # a valid path should have at least 3 return values and return to the source
    return (len(path)>=3) and (path[0]==path[-1])

def is_valid_null_path(path:List[int])->bool:
    # a valid path should have at least 3 return values and return to the source
    return (len(path)==2) and (path[0]==path[-1])

def is_valid_multi_path(paths: List[List[int]], depots: List[int], num_cities)->bool:
    '''
    Arguments:
    paths: list of paths where each path is a list of nodes that start and end at the same node
    depots: list of nodes indicating the valid sources for each traveling salesman

    Output:
    boolean indicating if the paths are valid or not
    '''
    assert len(paths) == len(depots), ValueError("Received unequal number of paths to depots. Note that if you choose to not use a salesman, you must still return a corresponding empty path: [depot, depot].")
    # check that each subpath is valid --> "empty" paths must be represented as [depot, depot]
    if not all([is_valid_path(path) or is_valid_null_path(path) for path in paths]):
        return False

    # check if the start and end of each subtour match the depots
    if not all([(path[0]==depots[i] and path[-1]==depots[i]) for i, path in enumerate(paths)]):
        return False
        
    # check that each city is only visited once across all salesmen
    all_non_depot_nodes = []
    for path in paths:
        all_non_depot_nodes.extend(path[1:-1])
    assert len(all_non_depot_nodes) == len(set(all_non_depot_nodes)), ValueError("Duplicate Visits")
    assert set(all_non_depot_nodes) == set(list(range(num_cities))).difference(set(depots)), ValueError("Invalid number of cities visited")
    return True

def get_tour_distance(synapse:Union[GraphV1Synapse, GraphV2Synapse])->float:
    '''
    Returns the total tour distance for the TSP problem as a float.

    Takes a synapse as its only argument
    '''
    problem = synapse.problem
    if 'TSP' not in problem.problem_type:
        raise ValueError(f"get_tour_distance is an invalid function for processing {problem.problem_type}")
    
    if not synapse.solution:
        return np.inf
    distance=np.nan
    if problem.directed or isinstance(synapse.problem, GraphV2Problem):
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
            try:
                path_is_valid = is_valid_path(path)
            except AssertionError as e:
                bt.logging.trace(f"Error while validating solution: {e}")
            # assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')
            if path_is_valid:
                # sort cities into pairs
                pairs = [(path[i], path[i+1]) for i in range(len(path)-1)]
                distance = 0
                for pair in pairs:
                    distance += math.hypot(coordinates[pair[0]][0] - coordinates[pair[1]][0], coordinates[pair[0]][1] - coordinates[pair[1]][1])
    return distance if not np.isnan(distance) else np.inf

def get_multi_minmax_tour_distance(synapse: GraphV2Synapse)->float:
    '''
    Returns the maximum tour distance across salesmen for the mTSP as a float.

    Takes a synapse as its only argument
    '''
    problem = synapse.problem
    if 'mTSP' not in problem.problem_type:
        raise ValueError(f"get_multi_tour_distance is an invalid function for processing {problem.problem_type}")

    if not synapse.solution:
        return np.inf
    distance=np.nan
    assert isinstance(problem, GraphV2ProblemMulti) or isinstance(problem, GraphV2ProblemMultiConstrained), ValueError(f"Attempting to use multi-path function for problem of type: {type(problem)}")
    
    assert len(problem.edges) == len(problem.edges[0]) and len(problem.edges)==problem.n_nodes, ValueError(f"Wrong distance matrix shape of: ({len(problem.edges[0])}, {len(problem.edges)}) for problem of n_nodes: {problem.n_nodes}")
    edges=problem.edges
    paths=synapse.solution
    depots=problem.depots

    if isinstance(paths,list):
        paths_are_valid =  is_valid_multi_path(paths, depots, problem.n_nodes)
        if paths_are_valid:
            # assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')
            distances = []
            for path in paths:
                distance = 0
                for i, source in enumerate(path[:-1]):
                    destination = path[i+1]
                    try:
                        distance += edges[source][destination]
                    except IndexError as e:
                        print(f"IndexError with source: {source}, destination: {destination}, distance_mat_shape: {np.array(edges).shape}")
                distances.append(distance)
            max_distance = max(distances)
        else:
            bt.logging.trace(f"Received invalid paths: {paths}")
    return max_distance if not np.isnan(distance) else np.inf

def get_portfolio_distribution_similarity(synapse: GraphV1PortfolioSynapse):
    '''
    Returns the number of swaps and objective score

    Takes a synapse as its only argument
    '''
    problem = synapse.problem

    if not synapse.solution:
        print("Solution not found")
        return 1000000, 0
    
    swaps = synapse.solution
    initial_portfolio_np = np.array(problem.initialPortfolios)
    
    if not all([len(swap)==4 for swap in swaps]):
        print("Incorrect swap length")
        return 1000000, 0

    def instantiate_pools(problem: Union[GraphV1PortfolioProblem]):
        current_pools: List[SubnetPool] = []
        for netuid, pool in enumerate(problem.pools):
            current_pools.append(SubnetPool(pool[0], pool[1], netuid))
        return current_pools
    
    current_pools = instantiate_pools(problem)

    ### iterate through all the swaps in the solution
    for swap in swaps:
        # [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_alpha_tokens]
        portfolio_idx = swap[0]
        from_subnet_idx = swap[1]
        to_subnet_idx = swap[2]
        from_num_alpha_tokens = swap[3]
        if from_num_alpha_tokens == 0:
            # no swap to be made
            continue
        if initial_portfolio_np[portfolio_idx, from_subnet_idx] >= from_num_alpha_tokens or np.isclose(initial_portfolio_np[portfolio_idx, from_subnet_idx], from_num_alpha_tokens, atol=1e-2):
            tao_emitted = current_pools[from_subnet_idx].swap_alpha_to_tao(from_num_alpha_tokens)
            alpha_emitted = current_pools[to_subnet_idx].swap_tao_to_alpha(tao_emitted)
            if np.isclose(initial_portfolio_np[portfolio_idx, from_subnet_idx], from_num_alpha_tokens, atol=1e-7):
                initial_portfolio_np[portfolio_idx, from_subnet_idx] = 0
            else:
                initial_portfolio_np[portfolio_idx, from_subnet_idx] -= from_num_alpha_tokens
            initial_portfolio_np[portfolio_idx, to_subnet_idx] += alpha_emitted
        else:
            if abs(initial_portfolio_np[portfolio_idx, from_subnet_idx]-from_num_alpha_tokens)/from_num_alpha_tokens < 0.01:
                # if the difference is less than 1% then we can assume that the swap is valid
                tao_emitted = current_pools[from_subnet_idx].swap_alpha_to_tao(from_num_alpha_tokens)
                alpha_emitted = current_pools[to_subnet_idx].swap_tao_to_alpha(tao_emitted)
                if np.isclose(initial_portfolio_np[portfolio_idx, from_subnet_idx], from_num_alpha_tokens, atol=1e-7):
                    initial_portfolio_np[portfolio_idx, from_subnet_idx] = 0
                else:
                    initial_portfolio_np[portfolio_idx, from_subnet_idx] -= from_num_alpha_tokens
                initial_portfolio_np[portfolio_idx, to_subnet_idx] += alpha_emitted
            else:
                # print(swap, initial_portfolio_np[portfolio_idx, from_subnet_idx], from_num_alpha_tokens)
                # print("Not enough alpha")
                return 1000000, 0

    ### calculate the new distribution of tao
    total_tokens_in_each_subnet_in_portfolios = initial_portfolio_np.sum(axis=0)
    updated_poools_np = np.array([[pool.num_tao_tokens, pool.num_alpha_tokens] for pool in current_pools])
    total_equivalent_tao_in_each_subnet_in_portfolios = np.append(np.array([total_tokens_in_each_subnet_in_portfolios[0]]), updated_poools_np[1:, 0] - updated_poools_np[1:, 0]*updated_poools_np[1:, 1] / (updated_poools_np[1:, 1] + total_tokens_in_each_subnet_in_portfolios[1:]))
    final_tao_total = sum(total_equivalent_tao_in_each_subnet_in_portfolios)
    final_tao_distribution = total_equivalent_tao_in_each_subnet_in_portfolios/final_tao_total * 100

    ### penalize any deviation
    objective_score = 100
    assert len(final_tao_distribution)==len(problem.constraintValues) , ValueError('len(final_tao_distribution) != len(problem.constraintValues)')
    for netuid, constraintValue in enumerate(problem.constraintValues):
        constraintType = problem.constraintTypes[netuid]
        ## penalize differences squared
        if constraintType == "ge":
            if final_tao_distribution[netuid] < constraintValue:
                if constraintValue - final_tao_distribution[netuid] > 1: # 1% deviation threshold
                    deviation = constraintValue - final_tao_distribution[netuid]
                    return 1000000, 0
                objective_score -= (constraintValue - final_tao_distribution[netuid])**2
        elif constraintType == "eq":
            if final_tao_distribution[netuid] != constraintValue:
                if abs(constraintValue - final_tao_distribution[netuid]) > 1: # 1% deviation threshold
                    deviation = constraintValue - final_tao_distribution[netuid]
                    return 1000000, 0
                objective_score -= abs(constraintValue - final_tao_distribution[netuid])**2
        elif constraintType == "le":
            if final_tao_distribution[netuid] > constraintValue:
                if final_tao_distribution[netuid] - constraintValue > 1: # 1% deviation threshold
                    deviation = constraintValue - final_tao_distribution[netuid]
                    return 1000000, 0
                objective_score -= (final_tao_distribution[netuid] - constraintValue)**2
    
    return len(swaps), max(0, objective_score)**3

def normalize_coordinates(coordinates:List[List[Union[int,float]]]):
    '''
    Normalizes all coordinates against the max x or y value.

    Normalized coordinate values are required my some open-source algorithms/models.

    Assumes all coordinates are non-negative (in line with the current GraphV1Problem Formulation)
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

def is_valid_solution(problem:Union[GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2ProblemMultiConstrained, GraphV1PortfolioProblem], solution:Union[List[List[Union[float, int]]],List[int]]):
    # nested function to validate solution type
    def is_valid_solution_type(solution, problem_type):
        try:
            assert isinstance(solution, list)
            if "mTSP" in problem_type:
                assert all(isinstance(row, list) and all(isinstance(x, int) for x in row) for row in solution)
            elif "TSP" in problem_type:
                assert all(isinstance(row, int) for row in solution)
            elif "PortfolioReallocation" in problem_type:
                assert all(isinstance(row[0], int) for row in solution)
                assert all(isinstance(row[1], int) for row in solution)
                assert all(isinstance(row[2], int) for row in solution)
                assert all(isinstance(row[3], int) or isinstance(row[3], float) for row in solution)
                assert all(len(row)==4 for row in solution)
            else:
                # received invalid problem type
                return False
        except AssertionError as e:
            # invalid solution format
            return False
        return True

    if is_valid_solution_type(solution, problem.problem_type):
        if "cmTSP" in problem.problem_type:
            # This is an cmTSP
            # check if there are as many paths as salesmen
            if len(solution) != problem.n_salesmen:
                return False
            if not all([len(path)>=1 for path in solution]):
                # disallow empty list. If salesman not assigned city, return [depot, depot]
                return False
            for idx, path in enumerate(solution):
                if sum([problem.demand[i] for i in path]) > problem.constraint[idx]:
                    return False
            if all([path[0]==problem.depots[i] for i, path in enumerate(solution)]):
                if problem.to_origin == True:
                    if not all([path[0]==path[-1] and len(path)>=2 for path in solution]):
                        return False
            else:
                # not all paths start and end at their respective depots
                return False
            
            if problem.visit_all == True: 
                all_non_depot_nodes = []
                for path in solution:
                    # path[1:-1] is an empty list for [depot, depot]
                    all_non_depot_nodes.extend(path[1:-1])

                # check if every node has been visited exactly once;
                if not (len(all_non_depot_nodes) == len(set(all_non_depot_nodes)) \
                    and set(all_non_depot_nodes) == set(range(problem.n_nodes)).difference(problem.depots)):
                    return False
            return True
        elif "mTSP" in problem.problem_type:
            # This is an mTSP
            # check if there are as many paths as salesmen
            if len(solution) != problem.n_salesmen:
                return False
            if not all([len(path)>=1 for path in solution]):
                # disallow empty list. If salesman not assigned city, return [depot, depot]
                return False
            if all([path[0]==problem.depots[i] for i, path in enumerate(solution)]):
                if problem.to_origin == True:
                    if not all([path[0]==path[-1] and len(path)>=2 for path in solution]):
                        return False
            else:
                # not all paths start and end at their respective depots
                return False
            
            if problem.visit_all == True: 
                all_non_depot_nodes = []
                for path in solution:
                    # path[1:-1] is an empty list for [depot, depot]
                    all_non_depot_nodes.extend(path[1:-1])

                # check if every node has been visited exactly once;
                if not (len(all_non_depot_nodes) == len(set(all_non_depot_nodes)) \
                    and set(all_non_depot_nodes) == set(range(problem.n_nodes)).difference(problem.depots)):
                    return False
            return True
        elif "PortfolioReallocation" in problem.problem_type:
            return True
        else:
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
    else:
        return False

def is_valid_portfolio_solution(problem:Union[GraphV1PortfolioProblem], solution):

    if not all([len(swap)==4 for swap in solution]):
        return False
    if not all([swap[0]<problem.n_portfolio for swap in solution]):
        return False
    if not all([swap[1]<len(problem.pools) for swap in solution]):
        return False
    if not all([swap[2]<len(problem.pools) for swap in solution]):
        return False
    
    return True

def valid_problem(problem:Union[GraphV1Problem, GraphV2Problem, GraphV1PortfolioProblem])->bool:
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
    
    elif problem.problem_type == 'Metric mTSP':
        if (problem.directed==False) \
            and (problem.visit_all==True) \
            and (problem.to_origin==True) \
            and (problem.objective_function=='min') \
            and (problem.n_salesmen > 1) \
            and (len(problem.depots)==problem.n_salesmen):
            if problem.single_depot == True:
                # assert that all the depots be at source city #0
                return True if all([depot==0 for depot in problem.depots]) else False
            else:
                # assert that all depots are different
                return True if len(set(problem.depots)) == len(problem.depots) else False
        else:
            bt.logging.info(f"Received an invalid Metric mTSP problem")
            bt.logging.info(problem.get_info(verbosity=2))
            return False
        
    elif problem.problem_type == 'General mTSP':
        if (problem.directed==True) \
            and (problem.visit_all==True) \
            and (problem.to_origin==True) \
            and (problem.objective_function=='min') \
            and (problem.n_salesmen > 1) \
            and (len(problem.depots)==problem.n_salesmen):
            if problem.single_depot == True:
                # assert that all the depots be at source city #0
                return True if all([depot==0 for depot in problem.depots]) else False
            else:
                # assert that all depots are different
                return True if len(set(problem.depots)) == len(problem.depots) else False
        else:
            bt.logging.info(f"Received an invalid General mTSP problem")
            bt.logging.info(problem.get_info(verbosity=2))
            return False

    elif problem.problem_type == 'Metric cmTSP':
        if (problem.directed==False) \
            and (problem.visit_all==True) \
            and (problem.to_origin==True) \
            and (problem.objective_function=='min') \
            and (problem.n_salesmen > 1) \
            and (len(problem.depots)==problem.n_salesmen) \
            and (len(problem.demand) == problem.n_nodes) \
            and (len(problem.constraint) == problem.n_salesmen) \
            and (sum(problem.demand) <= sum(problem.constraint)):
            if problem.single_depot == True:
                # assert that all the depots be at source city #0
                return True if all([depot==0 for depot in problem.depots]) else False
            else:
                # assert that all depots are different
                return True if len(set(problem.depots)) == len(problem.depots) else False
        else:
            bt.logging.info(f"Received an invalid Metric mTSP problem")
            bt.logging.info(problem.get_info(verbosity=2))
            return False
        
    elif problem.problem_type == 'General cmTSP':
        if (problem.directed==True) \
            and (problem.visit_all==True) \
            and (problem.to_origin==True) \
            and (problem.objective_function=='min') \
            and (problem.n_salesmen > 1) \
            and (len(problem.depots)==problem.n_salesmen) \
            and (len(problem.demand) == problem.n_nodes) \
            and (len(problem.constraint) == problem.n_salesmen) \
            and (sum(problem.demand) <= sum(problem.constraint)):
            if problem.single_depot == True:
                # assert that all the depots be at source city #0
                return True if all([depot==0 for depot in problem.depots]) else False
            else:
                # assert that all depots are different
                return True if len(set(problem.depots)) == len(problem.depots) else False
        else:
            bt.logging.info(f"Received an invalid General mTSP problem")
            bt.logging.info(problem.get_info(verbosity=2))
            return False
        
    elif problem.problem_type == 'PortfolioReallocation':
        def assert_portfolio_count(problem):
            if not len(problem.initialPortfolios) == problem.n_portfolio:
                return False
            return True
        def assert_portfolio_subnet_count(problem):
            subnetsCount = len(problem.initialPortfolios[0])
            if not all([len(portfolio)==subnetsCount for portfolio in problem.initialPortfolios]):
                return False
            if not all([[subnetToken>=0 for subnetToken in portfolio] for portfolio in problem.initialPortfolios]):
                return False
            if not all([[type(subnetToken)==int for subnetToken in portfolio] for portfolio in problem.initialPortfolios]):
                return False
            return True
        def assert_constraintValues_type_count(problem):
            if problem.constraintValues:
                if not len(problem.constraintValues) == len(problem.initialPortfolios[0]):
                    return False
                if not all([type(constraintValue)==float or type(constraintValue)==int for constraintValue in problem.constraintValues]):
                    return False
                if not all([constraintValue>=0 for constraintValue in problem.constraintValues]):
                    return False
            return True
        def assert_constraintTypes_type_count(problem):
            if problem.constraintTypes:
                if not len(problem.constraintTypes) == len(problem.initialPortfolios[0]):
                    return False
                if not all([constraintType=="eq" or constraintType=="ge" or constraintType=="le" for constraintType in problem.constraintTypes]):
                    return False
            return True
        def assert_pool_type_count(problem):
            if problem.pools:
                if not len(problem.pools) == len(problem.initialPortfolios[0]):
                    return False
                if not all([len(pool)==2 for pool in problem.pools]):
                    return False
                if not all([(type(pool[0])==int) and (type(pool[1])==int) for pool in problem.pools]):
                    return False
                if not all([pool[0]>=0 and pool[1]>=0 for pool in problem.pools]):
                    return False
            return True
        def assert_feasible_constraint(problem):
            if problem.constraintTypes and problem.constraintValues:

                def check_constraints_feasibility(types, values):
                    """
                    types: list of 10 strings, each one of 'eq', 'ge', 'le'
                    values: list of 10 numbers (floats or ints), each representing a percentage
                    Returns True if types are feasible (some valid assignment can total 100%), False otherwise
                    """

                    eq_total = sum(t for c, t in zip(types, values) if c == "eq")
                    min_total = sum(t if c in ("eq", "ge") else 0 for c, t in zip(types, values))
                    max_total = sum(t if c in ("eq", "le") else 100 for c, t in zip(types, values))

                    return (
                        round(eq_total, 2) <= 100 and
                        round(min_total, 2) <= 100 and
                        round(max_total, 2) >= 100
                    )
                
                if not check_constraints_feasibility(problem.constraintTypes, problem.constraintValues) == True:
                    return False
            return True
        if (assert_portfolio_count(problem) \
            and assert_portfolio_subnet_count(problem) \
            and assert_constraintValues_type_count(problem) \
            and assert_constraintTypes_type_count(problem) \
            and assert_pool_type_count(problem) \
            and assert_feasible_constraint(problem)):
            return True
        else:
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
