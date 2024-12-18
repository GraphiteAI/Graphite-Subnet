'''
This script tests the scoring function of the ScoreResponse class which is responsible for handling the scoring of miner responses.
'''
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

class MDMTSPSolutionEnum(Enum):
    VALID = 0
    TOO_FEW_CITIES = 1
    TOO_MANY_CITIES = 2
    REPEAT_CITIES = 3
    INVALID_DEPOT = 4
    WRONG_DEPOT_ORDER = 5
    VISITED_DEPOT = 6 # visit another depot which is invalid
    DEPOT_TO_DEPOT = 7
    NO_RETURN = 8
    MIXED_TYPE = 9
    FLOAT_VALUES = 10
    EMPTY_ROUTES = 11

mock = Mock()
load_default_dataset(mock)

def generate_mdmtsp_test_response(problem:GraphV2ProblemMulti, solution_type:MDMTSPSolutionEnum):
    assert isinstance(solution_type, MDMTSPSolutionEnum)
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
    if solution_type == MDMTSPSolutionEnum.TOO_FEW_CITIES:
        # too few cities
        initial_solution[0] = [initial_solution[0][0]] + initial_solution[0][2:] # remove the element in the first index of the first route

    elif solution_type == MDMTSPSolutionEnum.TOO_MANY_CITIES:
        # too many cities
        initial_solution[0].insert(1, n_nodes) # insert an extra node 

    elif solution_type == MDMTSPSolutionEnum.REPEAT_CITIES:
        # repeated visit to a city
        initial_solution[0].insert(1, initial_solution[0][1]) # insert repeat node

    elif solution_type == MDMTSPSolutionEnum.INVALID_DEPOT:
        # invalid city
        invalid_depot = initial_solution[0][1]
        temp = initial_solution[0][0]
        initial_solution[0][0] = invalid_depot
        initial_solution[0][-1] = invalid_depot
        initial_solution[0][1] = temp

    elif solution_type == MDMTSPSolutionEnum.WRONG_DEPOT_ORDER:
        temp = initial_solution[0].copy()
        initial_solution[0] = initial_solution[1]
        initial_solution[1] = temp

    elif solution_type == MDMTSPSolutionEnum.VISITED_DEPOT:
        initial_solution[0].insert(1, initial_solution[1][0])

    elif solution_type == MDMTSPSolutionEnum.DEPOT_TO_DEPOT:
        initial_solution[0] = [initial_solution[0][0]] + initial_solution[1][1:-1] + initial_solution[0][1:]
        initial_solution[1] = [initial_solution[1][0], initial_solution[1][-1]]

    elif solution_type == MDMTSPSolutionEnum.NO_RETURN:
        initial_solution[0] = initial_solution[0][:-1]

    elif solution_type == MDMTSPSolutionEnum.MIXED_TYPE:
        initial_solution[0][1] = "wrong_type"
        initial_solution[0][2] = False

    elif solution_type == MDMTSPSolutionEnum.FLOAT_VALUES:
        initial_solution[0][1] = float(initial_solution[0][1]) + 0.5

    elif solution_type == MDMTSPSolutionEnum.EMPTY_ROUTES:
        initial_solution[0] = [initial_solution[0][0]] + initial_solution[1][1:-1] + initial_solution[0][1:]
        initial_solution[1] = []

    return initial_solution


class TestScoringFunction(unittest.TestCase):
    def setUp(self):
        # create an md-mTSP instance
        self.graph_problem = MetricMTSPV2Generator.generate_one_sample(mock.loaded_datasets, 2000, 4, single_depot=False)
        self.graph_rp_depot_problem = copy.deepcopy(self.graph_problem)
        self.graph_rp_depot_problem.depots[1] = self.graph_rp_depot_problem.depots[0]
        self.graph_synapse = GraphV2Synapse(problem=self.graph_problem)
        self.graph_synapse_rp = GraphV2Synapse(problem=self.graph_rp_depot_problem)
        self.score_handler = ScoreResponse(self.graph_synapse)
        self.score_handler_rp = ScoreResponse(self.graph_synapse_rp)
        self.solver = NearestNeighbourMultiSolver()
    
    ### Set of tests to ensure that the scoring function is applied correctly for the multi-depot instance
    # Did not visit all cities
    def test_too_few_cities(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.TOO_FEW_CITIES)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Visited invalid node value
    def test_too_many_nodes(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.TOO_MANY_CITIES)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Repeated Visit to cities
    def test_repeat_node(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.REPEAT_CITIES)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Valid null path
    def test_no_visit_node(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.DEPOT_TO_DEPOT)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertLess(test_score, np.inf)
    
    # float in solution
    def test_float_node(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.FLOAT_VALUES)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Non-numeric in solution
    def test_mixed_type_node(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.MIXED_TYPE)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Visited wrong order of depots (salesman i start from depot[i'] where i' != i)
    def test_wrong_order(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.WRONG_DEPOT_ORDER)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Did not return to source
    def test_no_return(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.NO_RETURN)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Choose not to use salesman i (this is a valid choice in the event that salesman i happens to be a large outlier from the rest of the cities to service)
    # In context, this might be used for determining the value of maintaining some fulfilment centers from a legacy network
    # Invalid return format for empty routes
    def test_empty_route(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.EMPTY_ROUTES)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_valid_route(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_problem, MDMTSPSolutionEnum.VALID)
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertLess(test_score, np.inf)

    # No response
    def test_null_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = None
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    # False response
    def test_false_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = False
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # True response
    def test_true_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = True
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Invalid solution type
    def test_inf_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = np.inf
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Invalid data type
    def test_wrong_type_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = "some_wrong_type"
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    ### Repeat tests for repeated depots problem
    # Did not visit all cities
    def test_too_few_cities_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.TOO_FEW_CITIES)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Visited invalid node value
    def test_too_many_nodes_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.TOO_MANY_CITIES)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Repeated Visit to cities
    def test_repeat_node_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.REPEAT_CITIES)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Valid null subpath
    def test_no_visit_node_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.DEPOT_TO_DEPOT)
        print(f"rp null: {get_multi_minmax_tour_distance(mock_synapse)} {is_valid_solution(mock_synapse.problem, mock_synapse.solution)}")
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertLess(test_score, np.inf)
    
    # float in solution
    def test_float_node_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.FLOAT_VALUES)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Non-numeric in solution
    def test_mixed_type_node_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.MIXED_TYPE)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Did not return to source
    def test_no_return_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.NO_RETURN)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Choose not to use salesman i (this is a valid choice in the event that salesman i happens to be a large outlier from the rest of the cities to service)
    # In context, this might be used for determining the value of maintaining some fulfilment centers from a legacy network
    def test_empty_route_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.EMPTY_ROUTES)
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_valid_route_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = generate_mdmtsp_test_response(self.graph_rp_depot_problem, MDMTSPSolutionEnum.VALID)
        print(f"rp valid: {get_multi_minmax_tour_distance(mock_synapse)} {is_valid_solution(mock_synapse.problem, mock_synapse.solution)}")
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertLess(test_score, np.inf)

    # No response
    def test_null_response_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = None
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    # False response
    def test_false_response_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = False
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # True response
    def test_true_response_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        mock_synapse.solution = True
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Invalid solution type
    def test_inf_response_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = np.inf
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    # Invalid data type
    def test_wrong_type_response_rp(self):
        mock_synapse = copy.deepcopy(self.graph_synapse_rp)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = "some_wrong_type"
        test_score = self.score_handler_rp.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)


if __name__=="__main__":
    unittest.main()