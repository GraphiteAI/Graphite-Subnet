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
from typing import List, Union, Dict
from graphite.solvers import NearestNeighbourSolver
from graphite.protocol import GraphV2Problem
import os
import random
import json
from collections import Counter
import asyncio
import numpy as np
from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges

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
    def generate_n_samples(cls, n:int, problem_model:GraphV2Problem, loaded_datasets, **kwargs)->List[GraphV2Problem]:
        '''
        Returns a list of dictionaries containing the parameters required to initialize each problem
        This can be used in-situ to generate a dataset without saving it
        '''
        ...

    @classmethod
    def generate_and_save_dataset(cls, n_samples: int, file_name: str=None, save_dir: str=None, loaded_datasets: Dict={}):
        '''
        generates n sample GraphV2Problems and saves it as a json file
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
        problems = cls.generate_n_samples(n_samples, loaded_datasets)
        
        # serialize and save json string with meta data of the dataset
        output_json = cls.serialize_dataset(problems)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write(output_json)
        
        # also return the generated samples
        return problems

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
    def serialize_dataset(cls, problems: List[GraphV2Problem]):
        # the data should be json serializable given the coerced data types defined in the GraphV2Problem
        meta_data = {
            'problem_type': 'Metric TSP',
            'n_samples': len(problems),
            'problems': [problem.model_dump_json() for problem in problems],
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
        dataset['problems'] = [GraphV2Problem(**json.loads(data)) for data in dataset['problems']]
        return dataset
    

class MetricTSPGenerator(DatasetGenerator):
    save_dir = os.path.join(BASE_DIR,'metric_tsp')
    file_name = os.path.join('dataset.json')
    problem_weights = [0.1, 0.4, 0.4, 0.1] # probability of choosing a small, medium, large, or very large problem
    
    def recreate_edges(problem: GraphV2Problem, loaded_datasets):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            problem.edges = geom_edges(node_coords).tolist()
        elif problem.cost_function == "Euclidean2D":
            problem.edges = euc_2d_edges(node_coords).tolist()
        elif problem.cost_function == "Manhatten2D":
            problem.edges = man_2d_edges(node_coords).tolist()
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."

    @classmethod
    def generate_n_samples(cls, n: int, loaded_datasets):
        # generate 15 cities as a default with random coordinates
        problems = []
        for _ in range(n):
            n_nodes = random.randint(2000, 5000)
            # randomly select n_nodes indexes from the selected graph
            prob_select = random.randint(0, len(list(loaded_datasets.keys()))-1)
            dataset_ref = list(loaded_datasets.keys())[prob_select]
            selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref])), n_nodes)
            test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)
            cls.recreate_edges(test_problem, loaded_datasets)
            problems.append(test_problem)
        return problems


if __name__=="__main__":
    print('________________________')
    print('Testing MetricTSPGenerator V2')
    loaded_datasets = {}
    try:
        with np.load('dataset/Asia_MSB.npz') as f:
            loaded_datasets["Asia_MSB"] = np.array(f['data'])
    except:
        pass
    try:
        with np.load('dataset/World_TSP.npz') as f:
            loaded_datasets["World_TSP"] = np.array(f['data'])
    except:
        pass
    MetricTSPGenerator.generate_and_save_dataset(n_samples=1, loaded_datasets=loaded_datasets)
    dataset = MetricTSPGenerator.load_dataset()
    sample_problem = dataset['problems'][0]
    solver = NearestNeighbourSolver()
    # print(sample_problem)
    print(type(sample_problem))
    print('Sample metric tsp path: ', end='')
    print(asyncio.run(solver.solve_problem(sample_problem)))


