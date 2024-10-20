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
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.utils.graph_utils import timeout
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class GeneticSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        population_size = 100  # Size of the population
        generations = 500  # Number of generations
        mutation_rate = 0.01  # Mutation rate
        n = len(distance_matrix)

        # Step 1: Initialize the population with random routes
        population = [self.random_route(n) for _ in range(population_size)]

        # Step 2: Evolve the population over generations
        for generation in range(generations):
            if self.future_tracker.get(future_id):
                return None

            # Step 3: Evaluate fitness of each route (lower distance is better)
            population_fitness = [
                (route, self.calculate_total_distance(route, distance_matrix))
                for route in population
            ]
            population_fitness.sort(key=lambda x: x[1])  # Sort by distance

            # Step 4: Selection - take the top 50% of the population
            selected_population = [
                route for route, _ in population_fitness[: population_size // 2]
            ]

            # Step 5: Crossover - create new routes by combining pairs from the selected population
            offspring = []
            for _ in range(population_size // 2):
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child = self.crossover(parent1, parent2)
                offspring.append(child)

            # Step 6: Mutation - randomly mutate some offspring
            offspring = [self.mutate(route, mutation_rate) for route in offspring]

            # Step 7: Create the new population by combining selected parents and offspring
            population = selected_population + offspring

        # Step 8: Choose the best route from the final population
        best_route = min(
            population,
            key=lambda route: self.calculate_total_distance(route, distance_matrix),
        )
        best_distance = self.calculate_total_distance(best_route, distance_matrix)

        # Step 9: Return to the starting node to complete the cycle
        best_route.append(best_route[0])  # Ensure route ends at the start
        print(f"Total Distance GeneticAlgorithm: {best_distance}")
        return best_route

    def random_route(self, n: int) -> List[int]:
        """Generate a random route (a permutation of nodes)."""
        route = list(range(n))
        random.shuffle(route)
        return route

    def calculate_total_distance(
        self, route: List[int], distance_matrix: List[List[int]]
    ) -> float:
        """Calculate the total distance of the route."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += distance_matrix[route[-1]][route[0]]  # Return to the start
        return total_distance

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Crossover two parents to produce a child using order crossover."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        # Copy a slice from the first parent
        child[start:end] = parent1[start:end]
        # Fill the remaining cities from parent2
        parent2_genes = [gene for gene in parent2 if gene not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = parent2_genes[idx]
                idx += 1
        return child

    def mutate(self, route: List[int], mutation_rate: float) -> List[int]:
        """Randomly swap two cities in the route with a small probability."""
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

if __name__=="__main__":
    # # runs the solver on a test MetricTSP
    # n_nodes = 100
    # test_problem = GraphV1Problem(n_nodes=n_nodes)
    # solver = NearestNeighbourSolver(problem_types=[test_problem])
    # start_time = time.time()
    # route = asyncio.run(solver.solve_problem(test_problem))
    # print(f"{solver.__class__.__name__} Solution: {route}")
    # print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")


    ## Test case for GraphV2Problem
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    loaded_datasets = {}
    with np.load('dataset/Asia_MSB.npz') as f:
        loaded_datasets["Asia_MSB"] = np.array(f['data'])
    def recreate_edges(problem: GraphV2Problem):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
      
    n_nodes = random.randint(2000, 5000)
    # randomly select n_nodes indexes from the selected graph
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")
    if isinstance(test_problem, GraphV2Problem):
        test_problem.edges = recreate_edges(test_problem)
    print("Problem", test_problem)
    solver = NearestNeighbourSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")

    solver = NearestNeighbourSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
