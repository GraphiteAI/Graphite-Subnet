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

from typing import List, Union
import matplotlib.pyplot as plt
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver_multi import NearestNeighbourMultiSolver
from graphite.solvers.greedy_solver_multi_2 import NearestNeighbourMultiSolver2
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.utils.graph_utils import timeout, get_multi_minmax_tour_distance
from graphite.data.dataset_utils import load_default_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class InsertionMultiSolver(BaseSolver):
    '''
    This solver is a constructive nearest_neighbour algorithm that assigns cities to subtours based on the min increase in objective function value.
    '''
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    def get_random_valid_start(self, depot_id, distance_matrix, taken_nodes:list[int]=[], selection_range:int=5) -> int:
        distances = [(city_id, distance) for city_id, distance in enumerate(distance_matrix[depot_id].copy())]
        # reverse sort the copied list and pop from it
        assert (selection_range + len(taken_nodes)) < len(distances)
        distances.sort(reverse=True, key=lambda x: x[1])
        closest_cities = []
        while len(closest_cities) < selection_range:
            selected_city = None
            while not selected_city:
                city_id, distance = distances.pop()
                if city_id not in taken_nodes:
                    selected_city = city_id
            closest_cities.append(selected_city)
        return random.choice(closest_cities)
        
    def get_starting_tours(self, depots, distance_matrix):
        taken_nodes = depots.copy()
        initial_incomplete_tours = []
        for depot in depots:
            first_visit = self.get_random_valid_start(depot, distance_matrix, taken_nodes)
            initial_incomplete_tours.append([depot, first_visit, depot])
            taken_nodes.append(first_visit)
        return initial_incomplete_tours
    
    async def solve(self, formatted_problem, future_id:int)->List[int]:
        def get_cheapest_insertion(distance_matrix, original_distance, city, subtour):
            # returns the subtour with the cheapest insertion value
            min_distance = np.inf
            insertion_index = None
            for index in range(1, len(subtour)):
                new_distance = original_distance - distance_matrix[subtour[index-1]][subtour[index]] + distance_matrix[subtour[index-1]][city] + distance_matrix[city][subtour[index]]
                if new_distance < min_distance:
                    min_distance = new_distance
                    insertion_index = index
            new_subtour = subtour.copy()
            new_subtour.insert(insertion_index, city)
            net_increase = min_distance - original_distance
            return new_subtour, net_increase
        
        def subtour_distance(distance_matrix, subtour):
            subtour = np.array(subtour)
            next_points = np.roll(subtour, -1) 
            distances = distance_matrix[subtour, next_points] 
            total_distance = np.sum(distances)
            return total_distance
        
        # construct m tours
        m = formatted_problem.n_salesmen
        distance_matrix = np.array(formatted_problem.edges)
        unvisited = [city for city in range(len(distance_matrix)) if city not in set(formatted_problem.depots)]
        tours = self.get_starting_tours(formatted_problem.depots, distance_matrix)
        for _, first_city, _ in tours:
            unvisited.remove(first_city)

        distances = [subtour_distance(distance_matrix, subtour) for subtour in tours]
        while unvisited:
            # Choose a random city to derive the cheapest insertion
            insertion_city = random.sample(unvisited, 1)[0]
            new_distance_increases = []
            new_cheapest_subtours = []
            for index, subtour in enumerate(tours):
                if distances[index] != max(distances):
                    new_subtour, net_increase = get_cheapest_insertion(distance_matrix, distances[index], insertion_city, subtour)
                    new_distance_increases.append(net_increase)
                    new_cheapest_subtours.append(new_subtour)
                else:
                    new_distance_increases.append(np.inf)
                    new_cheapest_subtours.append(None)
            # Set the subtour to assign the insertion to
            chosen_index = new_distance_increases.index(min(new_distance_increases))
            assert new_cheapest_subtours[chosen_index] is not None, "Error: Chosen subtour is None"
            tours[chosen_index] = new_cheapest_subtours[chosen_index]
            distances[chosen_index] += min(new_distance_increases)
            unvisited.remove(insertion_city)
        
        return tours
    
    def problem_transformations(self, problem: Union[GraphV2ProblemMulti]):
        return problem

if __name__=="__main__":
    # runs the solver on a test Metric mTSP
    class Mock:
        def __init__(self) -> None:
            pass        

        def recreate_edges(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
            if problem.cost_function == "Geom":
                return geom_edges(node_coords)
            elif problem.cost_function == "Euclidean2D":
                return euc_2d_edges(node_coords)
            elif problem.cost_function == "Manhatten2D":
                return man_2d_edges(node_coords)
            else:
                return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
    mock = Mock()
    load_default_dataset(mock)

    n_nodes = 2000
    m = 10

    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=[0]*m)
    test_problem.edges = mock.recreate_edges(test_problem)
    solver1 = InsertionMultiSolver(problem_types=[test_problem])
    start_time = time.time()
    route1 = asyncio.run(solver1.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route1)
    score1 = get_multi_minmax_tour_distance(test_synapse)
    solver2 = NearestNeighbourMultiSolver(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route2)
    score2 = get_multi_minmax_tour_distance(test_synapse)
    # print(f"{solver.__class__.__name__} Solution: {route1}")
    print(f"{solver1.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time} and Salesmen: {m}")
    print(f"Insertion scored: {score1} while Multi scored: {score2}")
    
    n_nodes = 2000
    m = 10

    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=random.sample(range(n_nodes),m), single_depot=False)
    test_problem.edges = mock.recreate_edges(test_problem)
    solver1 = InsertionMultiSolver(problem_types=[test_problem])
    start_time = time.time()
    route1 = asyncio.run(solver1.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route1)
    score1 = get_multi_minmax_tour_distance(test_synapse)
    solver2 = NearestNeighbourMultiSolver2(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route2)
    score2 = get_multi_minmax_tour_distance(test_synapse)
    # print(f"{solver.__class__.__name__} Solution: {route1}")
    print(f"{solver1.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time} and Salesmen: {m}")
    print(f"Insertion scored: {score1} while Multi2 scored: {score2}")
    