'''
This file is a comprehensive test for the pydantic BaseModel
'''

import unittest
from pydantic import ValidationError
from typing import List, Union, Optional
import numpy as np
from graphite.protocol import GraphV1Problem, IsAlive, GraphV1Synapse
import bittensor as bt

class TestProtocol(unittest.TestCase):
    def test_random_generation_initialization(self):
        print(f"Testing random generation")
        random_metric_tsp = GraphV1Problem(n_nodes=8)
        self.assertEqual(random_metric_tsp.n_nodes, 8)
        self.assertEqual(len(random_metric_tsp.nodes), 8)
        self.assertEqual(len(random_metric_tsp.edges), 8)
        self.assertEqual(len(random_metric_tsp.edges[0]), 8)

    def test_fixed_coordinate_initialization(self):
        print(f"Testing fixed coordinate input initialization")
        metric_tsp = GraphV1Problem(n_nodes=3, nodes=[[1, 0], [2, 0], [3, 5]])
        self.assertEqual(metric_tsp.n_nodes, 3)
        self.assertEqual(len(metric_tsp.nodes), 3)
        self.assertEqual(len(metric_tsp.edges), 3)
        self.assertEqual(len(metric_tsp.edges[0]), 3)
        self.assertEqual(metric_tsp.nodes, [[1, 0], [2, 0], [3, 5]])

    def test_invalid_weight_initialization(self):
        print(f"Testing invalid weight initialization")
        with self.assertRaises(ValueError):
           GraphV1Problem(n_nodes=3, edges=[[1, 0, 5], [2, 0, 7]])

    def test_nodes_length_mismatch_initialization(self):
        print(f"Testing nodes length mismatch input initialization")
        with self.assertRaises(ValueError):
            GraphV1Problem(n_nodes=2, nodes=[[1, 5], [2, 7], [3, 2]])

    def test_invalid_coordinate_initialization(self):
        print(f"Testing erroneous coordinate input initialization")
        with self.assertRaises(ValueError):
            GraphV1Problem(n_nodes=3, nodes=[[1, 2, 5], [2, 6, 7], [3, 1, 2]])

    def test_negative_coordinate_input_initialization(self):
        print(f"Testing negative coordinate input initialization")
        with self.assertRaises(ValidationError):
            GraphV1Problem(n_nodes=3, nodes=[[-1, 5], [2, 7], [-3.2, 2]])

    def test_generate_random_coordinates(self):
        print(f"Testing random coordinate generation")
        problem = GraphV1Problem(n_nodes=10)
        coordinates = problem.generate_random_coordinates(10)
        self.assertEqual(len(coordinates), 10)
        self.assertTrue(all(len(coord) == 2 for coord in coordinates))
        self.assertTrue(all(coord[0] >= 0 and coord[1] for coord in coordinates))

    def test_get_distance_matrix(self):
        print(f"Testing distance matrix generation")
        problem = GraphV1Problem(n_nodes=3, nodes=[[0, 0], [3, 4], [6, 8]])
        distance_matrix = problem.get_distance_matrix(problem.nodes)
        self.assertEqual(len(distance_matrix), 3)
        self.assertAlmostEqual(distance_matrix[0][1], 5.0)
        self.assertAlmostEqual(distance_matrix[1][2], 5.0)

    def test_generate_edges(self):
        print(f"Testing edge weight generation")
        problem = GraphV1Problem(n_nodes=4, directed=True)
        edges = problem.generate_edges(4)
        self.assertEqual(len(edges), 4)
        self.assertEqual(len(edges[0]), 4)

if __name__ == '__main__':
    unittest.main()
