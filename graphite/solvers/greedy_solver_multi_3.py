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
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2ProblemMultiConstrained, GraphV2Synapse
from graphite.utils.graph_utils import timeout, get_multi_minmax_tour_distance
from graphite.data.dataset_utils import load_default_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import numpy as np
import time
import asyncio
import random
import math

import bittensor as bt

class NearestNeighbourMultiSolver3(BaseSolver):
    '''
    This solver is a constructive nearest_neighbour algorithm that assigns cities to subtours based on the min increase in objective function value.
    '''
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMultiConstrained()]):
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
            initial_incomplete_tours.append([depot, first_visit])
            taken_nodes.append(first_visit)
        return initial_incomplete_tours
    
    async def solve(self, formatted_problem, future_id:int)->List[int]:
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

        for _, first_city in tours:
            unvisited.remove(first_city)

        distances = [subtour_distance(distance_matrix, subtour) for subtour in tours]
        constraints = formatted_problem.constraint # List of constraints per salesman

        while unvisited:
            # print("TOUR LENS", [len(tour) for tour in tours])
            chosen_index = distances.index(min(distances))
            chosen_subtour = tours[chosen_index]

            min_distance = np.inf
            chosen_city = None
            for city in unvisited:
                new_distance = distances[chosen_index] - distance_matrix[chosen_subtour[-1]][0] + distance_matrix[chosen_subtour[-1]][city] + distance_matrix[city][0]
                if new_distance < min_distance:
                    chosen_city = city
                    min_distance = new_distance
            if chosen_city is not None and chosen_city in unvisited:
                distances[chosen_index] = min_distance
                tours[chosen_index] = chosen_subtour + [chosen_city]
                unvisited.remove(chosen_city)

            # Ensure we do not exceed constraint
            if len(tours[chosen_index]) - 1 >= constraints[chosen_index]:  # Exclude depot
                distances[chosen_index] = np.inf
            
        return [tour + [depot] for tour, depot in zip(tours, formatted_problem.depots.copy())] # complete each subtour back to source depot
    
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

    n_nodes = 500
    m = 5

    constraint = []
    # depots = sorted(random.sample(list(range(n_nodes)), k=m))
    depots = [11, 32, 55, 54, 78]
    demand = [1]*n_nodes
    for depot in depots:
        demand[depot] = 0
    visitable_nodes = n_nodes - m
    # for i in range(m-1):
    #     # ensure each depot should at least have 2 nodes to visit
    #     constraint.append(random.randint(2, visitable_nodes-sum(constraint)-2*(m-i-1)))
    # constraint.append(visitable_nodes - sum(constraint))
    # # add a layer of flexibility
    # constraint = [con+random.randint(0, 100) for con in constraint]

    # constraint = [(math.ceil(n_nodes/m) + random.randint(0, int(n_nodes/m * 0.3)) - random.randint(0, int(n_nodes/m * 0.2))) for _ in range(m-1)]
    # constraint += [(math.ceil(n_nodes/m) + random.randint(0, int(n_nodes/m * 0.3)) - random.randint(0, int(n_nodes/m * 0.2)))] if sum(constraint) > n_nodes - (math.ceil(n_nodes/m) - random.randint(0, int(n_nodes/m * 0.2))) else [(n_nodes - sum(constraint) + random.randint(int(n_nodes/m * 0.2), int(n_nodes/m * 0.3)))]
    constraint = [114, 104, 96, 120, 123]
    
    test_problem = GraphV2ProblemMultiConstrained(problem_type="Metric cmTSP", 
                                            n_nodes=n_nodes, 
                                            selected_ids=random.sample(list(range(100000)),n_nodes), 
                                            cost_function="Geom", 
                                            dataset_ref="Asia_MSB", 
                                            n_salesmen=m, 
                                            depots=depots, 
                                            single_depot=False,
                                            demand=demand,
                                            constraint=constraint)
    test_problem.edges = mock.recreate_edges(test_problem)

    print("depots", depots)
    print("constraints", constraint, sum(constraint), n_nodes, sum(constraint) >= n_nodes)

    print("Running NNMS3")
    solver1 = NearestNeighbourMultiSolver3(problem_types=[test_problem])
    start_time = time.time()
    route1 = asyncio.run(solver1.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route1)
    score1 = get_multi_minmax_tour_distance(test_synapse)

    print(f"{solver1.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time} and Salesmen: {m}")
    print(f"Multi2 scored: {score1}")
    print("len of routes", [len(route) for route in route1])
    [print(route) for route in route1]
    