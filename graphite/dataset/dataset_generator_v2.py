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

# Defines class for create a dataset based on the synthetic distribution
from abc import ABC, abstractmethod
from typing import List
from graphite.solvers import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import os
import random
import json
from collections import Counter
import asyncio
import numpy as np
from graphite.data.distance import euc_2d, geom, man_2d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def print_value_count(strings):
    # Use Counter to count the occurrences of each string in the list
    counter = Counter(strings)
    
    # Print the counts
    for value, count in counter.items():
        print(f"{value}: {count}")

class DatasetGenerator(ABC):
    @classmethod
    @abstractmethod
    def generate_n_samples(cls, n:int, problem_model:GraphV1Problem, **kwargs)->List[GraphV1Problem]:
        '''
        Returns a list of dictionaries containing the parameters required to initialize each problem
        This can be used in-situ to generate a dataset without saving it
        '''
        ...

    @classmethod
    def generate_and_save_dataset(cls, n_samples: int, file_name: str=None, save_dir: str=None):
        '''
        generates n sample GraphV1Problems and saves it as a json file
        '''
        if save_dir is None:
            save_dir = cls.save_dir
        if file_name is None:
            file_name = cls.file_name
        
        # Your logic to generate and save the dataset
        # For now, just a placeholder print statement
        # print(f"Generating {n_samples} samples and saving to {os.path.join(save_dir, file_name)}")
        
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate samples
        problems, sizes = cls.generate_n_samples(n_samples)
        
        # serialize and save json string with meta data of the dataset
        output_json = cls.serialize_dataset(problems, sizes)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write(output_json)
        
        # print("Generated and saved dataset with:")
        # print_value_count(sizes)

        # also return the generated samples
        return problems, sizes

    @classmethod
    def __weighted_sampling(cls, weights:List):
        """
        Perform weighted sampling from four categories and select a random integer from the corresponding range.

        Args:
        weights (list): List of weights for the four categories (small, medium, large, very large).

        Returns:
        int: A randomly selected integer from the selected category's range.
        """
        categories = ['small', 'medium', 'large', 'very large']
        ranges = {
            'small': (10, 14),
            'medium': (15, 25),
            'large': (26, 99),
            'very large': (100, 250)
        }

        # Select a category based on the provided weights
        selected_category = random.choices(categories, weights=weights, k=1)[0]

        # Select a random integer from the selected category's range
        selected_range = ranges[selected_category]
        selected_integer = random.randint(selected_range[0], selected_range[1])

        return selected_integer, selected_category
    
    @classmethod
    def serialize_dataset(cls, problems: List[GraphV1Problem], sizes:List[str]):
        # the data should be json serializable given the coerced data types defined in the GraphV1Problem
        meta_data = {
            'problem_type': 'Metric TSP',
            'n_samples': len(problems),
            'problems': [problem.model_dump_json() for problem in problems],
            'problem_sizes': sizes
        }
        # return the json string
        return json.dumps(meta_data)
    
    @classmethod
    def load_dataset(cls, filepath=None):
        if filepath == None:
            filepath = os.path.join(cls.save_dir, cls.file_name)
        with open(filepath,'r') as f:
            raw_data = json.load(f)
        
        # rebuild the problems
        dataset = raw_data.copy()
        dataset['problems'] = [GraphV1Problem(**json.loads(data)) for data in dataset['problems']]
        return dataset
    

class MetricTSPGenerator(DatasetGenerator):
    save_dir = os.path.join(BASE_DIR,'metric_tsp')
    file_name = os.path.join('dataset.json')
    problem_weights = [0.1, 0.4, 0.4, 0.1] # probability of choosing a small, medium, large, or very large problem
    
    def recreate_edges(problem: GraphV2Problem):
        # Select dataset to load
        loaded_datasets = {}
        with np.load(f"dataset/{problem.dataset_ref}.npz") as f:
            loaded_datasets[problem.dataset_ref] = np.array(f['data'])

        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = [node_coords_np[i][1:] for i in problem.selected_ids]
        num_nodes = len(node_coords)
        edge_matrix = np.zeros((num_nodes, num_nodes))
        if problem.cost_function == "Geom":
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i != j:
                        distance = geom(node_coords[i], node_coords[j])
                        edge_matrix[i][j] = distance
                        edge_matrix[j][i] = distance  # Since it's symmetric
        if problem.cost_function == "Euclidean2D":
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i != j:
                        distance = euc_2d(node_coords[i], node_coords[j])
                        edge_matrix[i][j] = distance
                        edge_matrix[j][i] = distance  # Since it's symmetric
        if problem.cost_function == "Manhatten2D":
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i != j:
                        distance = man_2d(node_coords[i], node_coords[j])
                        edge_matrix[i][j] = distance
                        edge_matrix[j][i] = distance  # Since it's symmetric

        problem.edges = edge_matrix

    @classmethod
    def generate_n_samples(cls, n: int):
        # generate 15 cities as a default with random coordinates
        problems = []
        sizes = []
        for _ in range(n):
            # sample n_nodes
            n_nodes, category= cls._DatasetGenerator__weighted_sampling(cls.problem_weights)

            loaded_datasets = {}
            with np.load('dataset/Asia_MSB.npz') as f:
                loaded_datasets["Asia_MSB"] = np.array(f['data'])
            with np.load('dataset/World_TSP.npz') as f:
                loaded_datasets["World_TSP"] = np.array(f['data'])
        
            n_nodes = random.randint(2000, 5000)
            # randomly select n_nodes indexes from the selected graph
            prob_select = random.randint(0, len(list(loaded_datasets.keys())-1))
            dataset_ref = list(loaded_datasets.keys())[prob_select]
            selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref])), n_nodes)
            test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)
            if isinstance(test_problem, GraphV2Problem):
                problem =  cls.recreate_edges(test_problem) 

            sizes.append(category)
            problems.append(GraphV1Problem(n_nodes=n_nodes))
        return problems, sizes


if __name__=="__main__":
    print('________________________')
    print('Testing MetricTSPGenerator')
    MetricTSPGenerator.generate_and_save_dataset(100)
    dataset = MetricTSPGenerator.load_dataset()
    sample_problem = dataset['problems'][0]
    solver = NearestNeighbourSolver()
    print(type(sample_problem))
    print('Sample metric tsp path: ', end='')
    print(asyncio.run(solver.solve_problem(sample_problem)))


