import unittest
from pydantic import ValidationError
from graphite.protocol import GraphV2ProblemMulti, GraphV2Synapse
from graphite.data.dataset_generator_v2 import MetricMTSPV2Generator
from graphite.data.dataset_utils import load_default_dataset
from graphite.validator.reward import ScoreResponse
from graphite.solvers import NearestNeighbourMultiSolver
from graphite.utils.graph_utils import get_multi_minmax_tour_distance, is_valid_multi_path, is_valid_solution
import numpy as np
from itertools import cycle, islice
import random
import copy
import asyncio
from enum import Enum

class Mock:
    def __init__(self): 
        pass

class SolutionEnum(Enum):
    VALID = 0
    TOO_FEW_CITIES = 1
    TOO_MANY_CITIES = 2
    REPEAT_CITIES = 3
    INVALID_DEPOT = 4
    WRONG_DEPOT_ORDER = 5
    VISITED_DEPOT = 6 # visit another depot
    DEPOT_TO_DEPOT = 7 # [0, 0] is not a valid path
    NO_RETURN = 8
    MIXED_TYPE = 9
    FLOAT_VALUES = 10
    EMPTY_ROUTES = 11

mock = Mock()
load_default_dataset(mock)

def generate_mdmtsp_test_response(problem:GraphV2ProblemMulti, solution_type:SolutionEnum):
    assert isinstance(solution_type, SolutionEnum)
    n_nodes = problem.n_nodes
    depots = problem.depots
    def generate_random_initial_solution(n_nodes, depots):
        # initialize start routes
        routes = [[x,x] for x in depots]
        remaining_cities = list(set(range(n_nodes)).difference(depots))
        random.shuffle(remaining_cities)
        route_indices = cycle(range(len(depots)))
        route_assignments = islice(route_indices, len(remaining_cities))
        for city in remaining_cities:
            routes[next(route_assignments)].insert(1, city)
        return routes
    # Generate a valid initial node
    initial_solution = generate_random_initial_solution(n_nodes, depots)
    if solution_type == SolutionEnum.TOO_FEW_CITIES:
        # too few cities
        initial_solution[0] = [initial_solution[0][0]] + initial_solution[0][2:] # remove the element in the first index of the first route

    elif solution_type == SolutionEnum.TOO_MANY_CITIES:
        # too many cities
        initial_solution[0].insert(1, n_nodes) # insert an extra node 

    elif solution_type == SolutionEnum.REPEAT_CITIES:
        # repeated visit to a city
        initial_solution[0].insert(1, initial_solution[0][1]) # insert repeat node

    elif solution_type == SolutionEnum.INVALID_DEPOT:
        # invalid city
        invalid_depot = initial_solution[0][1]
        temp = initial_solution[0][0]
        initial_solution[0][0] = invalid_depot
        initial_solution[0][-1] = invalid_depot
        initial_solution[0][1] = temp

    elif solution_type == SolutionEnum.WRONG_DEPOT_ORDER:
        temp = initial_solution[0].copy()
        initial_solution[0] = initial_solution[1]
        initial_solution[1] = temp

    elif solution_type == SolutionEnum.VISITED_DEPOT:
        initial_solution[0].insert(1, initial_solution[1][0])

    elif solution_type == SolutionEnum.DEPOT_TO_DEPOT:
        initial_solution[0] = [initial_solution[0][0]] + initial_solution[1][1:-1] + initial_solution[0][1:]
        initial_solution[1] = [initial_solution[1][0], initial_solution[1][-1]]

    elif solution_type == SolutionEnum.NO_RETURN:
        initial_solution[0] = initial_solution[0][:-1]

    elif solution_type == SolutionEnum.MIXED_TYPE:
        initial_solution[0][1] = "wrong_type"
        initial_solution[0][2] = False

    elif solution_type == SolutionEnum.FLOAT_VALUES:
        initial_solution[0][1] = float(initial_solution[0][1]) + 0.5

    elif solution_type == SolutionEnum.EMPTY_ROUTES:
        initial_solution[0] = [initial_solution[0][0]] + initial_solution[1][1:-1] + initial_solution[0][1:]
        initial_solution[1] = []

    return initial_solution


graph_problem = MetricMTSPV2Generator.generate_one_sample(mock.loaded_datasets, 2000, 4, single_depot=False)
graph_problem.depots[0] = graph_problem.depots[1]
graph_synapse = GraphV2Synapse(problem=graph_problem)
graph_synapse.solution = generate_mdmtsp_test_response(graph_problem, SolutionEnum.DEPOT_TO_DEPOT)
print(f"Validity Check: {is_valid_multi_path(graph_synapse.solution,graph_synapse.problem.depots,graph_synapse.problem.n_nodes)}")
print(f"Solution Score: {get_multi_minmax_tour_distance(graph_synapse)}")
print(f"Solution Validator Check: {is_valid_solution(graph_problem, graph_synapse.solution)}")