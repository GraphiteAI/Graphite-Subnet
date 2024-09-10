'''
script for testing rewards function
'''

import unittest
from graphite.validator.reward import scaled_rewards
from graphite.protocol import GraphV1Problem, GraphV1Synapse
from graphite.solvers import NearestNeighbourSolver, DPSolver, HPNSolver, BeamSearchSolver
import numpy as np

class TestRewards(unittest.TestCase):
    ### Suite of tests for problems that benchmark could not solve
    def test_unsolved_all_inf(self):
        mock_scores = [np.inf, np.inf, np.inf]
        mock_benchmark = np.inf
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        self.assertEqual(mock_rewards, [0,0,0])
        
    def test_unsolved_all_better(self):
        mock_scores = [1, 6, 11]
        mock_benchmark = np.inf
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[1,0.6,0.2])]
    
    def test_unsolved_all_better_negative(self):
        mock_scores = [-1, -6, -11]
        mock_benchmark = np.inf
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[0.2,0.6,1])]

    def test_unsolved_all_better_mixed(self):
        mock_scores = [-5, 0, 5]
        mock_benchmark = np.inf
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[1,0.6,0.2])]

    def test_unsolved_some_better(self):
        mock_scores = [-5, 0, np.inf, -5]
        mock_benchmark = np.inf
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[1,0.2,0,1])]

    ### Suite of tests for problems that benchmark could solve
    def test_solved_all_inf(self):
        mock_scores = [np.inf, np.inf, np.inf]
        mock_benchmark = 100
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        self.assertEqual(mock_rewards, [0,0,0])
    
    def test_solved_all_better(self):
        mock_scores = [1, 6, 11]
        mock_benchmark = 100
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[1,0.6,0.2])]

    def test_solved_some_better(self):
        mock_scores = [1, 6, 12]
        mock_benchmark = 11
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[1,0.6,0])]

    def test_solved_all_worse(self):
        mock_scores = [1, 6, 12]
        mock_benchmark = 0
        mock_rewards = scaled_rewards(mock_scores, mock_benchmark, 'min')
        [self.assertAlmostEqual(val1,val2) for val1, val2 in zip(mock_rewards,[0,0,0])] 

if __name__=="__main__":
    unittest.main()
