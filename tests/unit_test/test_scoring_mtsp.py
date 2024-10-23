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
from graphite.utils.graph_utils import get_multi_minmax_tour_distance
import numpy as np
import copy
import asyncio

class Mock:
    def __init__(self): 
        pass

mock = Mock()
load_default_dataset(mock)

class TestScoringFunction(unittest.TestCase):
    def setUp(self):
        self.graph_problem = MetricMTSPV2Generator.generate_one_sample(mock.loaded_datasets, 2000, 4)
        self.graph_synapse = GraphV2Synapse(problem=self.graph_problem)
        self.score_handler = ScoreResponse(self.graph_synapse)
        self.solver = NearestNeighbourMultiSolver()
    
    def test_too_few_cities(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [list(range(500))+[0], [0]+list(range(500,1000)) + [0], [0] + list(range(1000,1500)) + [0], [0] + list(range(1500,1999)) + [0]]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_too_many_nodes(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [list(range(500))+[0], [0]+list(range(500,1000)) + [0], [0] + list(range(1000,1500)) + [0], [0] + list(range(1500,2001)) + [0]]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_repeat_node(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [list(range(500))+[0], [0]+list(range(500,1000)) + [0], [0] + list(range(1000,1500)) + [0], [0] + list(range(1500,1999)) + [1998, 0]]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    def test_no_return(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [list(range(500))+[1], [0]+list(range(500,1000)) + [0], [0] + list(range(1000,1500)) + [0], [0] + list(range(1500,2000)) + [0]]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    def test_null_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = None
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    def test_inf_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = np.inf
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_wrong_type_response(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        with self.assertRaises(ValidationError) as context:
            mock_synapse.solution = "some_wrong_type"
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

if __name__=="__main__":
    unittest.main()