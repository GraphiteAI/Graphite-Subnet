'''
This script tests the scoring function of the ScoreResponse class which is responsible for handling the scoring of miner responses.

'''
import unittest
from pydantic import ValidationError
from graphite.protocol import GraphV1Synapse, GraphV1Problem
from graphite.validator.reward import ScoreResponse
from graphite.solvers import NearestNeighbourSolver
from graphite.utils.graph_utils import is_valid_path
import numpy as np
import copy
import asyncio

class TestScoringFunction(unittest.TestCase):
    def setUp(self):
        self.graph_problem = GraphV1Problem(n_nodes=10)
        self.graph_synapse = GraphV1Synapse(problem=self.graph_problem)
        self.score_handler = ScoreResponse(self.graph_synapse)
        self.solver = NearestNeighbourSolver()
    
    def test_too_few_cities(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [0,1,2,3,4]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_too_many_nodes(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [0,1,2,3,4,5,6,7,8,9,10,11,0]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

    def test_repeat_node(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [0,1,1,1,4,5,6,7,8,9,1,0]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    def test_no_return(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = [0,1,2,3,4,5,6,7,8,9,10,1]
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)
    
    def test_good_metric_answer(self):
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = asyncio.run(self.solver.solve_problem(copy.deepcopy(self.graph_problem)))
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertNotEqual(test_score, None)
        self.assertNotEqual(test_score, np.inf)
    
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

    def test_unsolvable_response(self):
        # when given a problem that the solver is unable to solve, the default return value is False
        mock_synapse = copy.deepcopy(self.graph_synapse)
        mock_synapse.solution = False
        test_score = self.score_handler.score_response(mock_synapse)
        self.assertEqual(test_score, np.inf)

if __name__=="__main__":
    unittest.main()