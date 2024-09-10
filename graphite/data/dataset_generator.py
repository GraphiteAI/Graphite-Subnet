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
from graphite.protocol import GraphV1Problem
import os
import random
import json
from collections import Counter
from pathlib import Path
import asyncio

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("dataset")

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
    save_dir = os.path.join(DATASET_DIR,'metric_tsp')
    file_name = os.path.join('dataset.json')
    problem_weights = [0.1, 0.4, 0.4, 0.1] # probability of choosing a small, medium, large, or very large problem
    
    @classmethod
    def generate_n_samples(cls, n: int):
        # generate 15 cities as a default with random coordinates
        problems = []
        sizes = []
        for _ in range(n):
            # sample n_nodes
            n_nodes, category= cls._DatasetGenerator__weighted_sampling(cls.problem_weights)
            sizes.append(category)
            problems.append(GraphV1Problem(n_nodes=n_nodes))
        return problems, sizes

class GeneralTSPGenerator(DatasetGenerator):
    save_dir = os.path.join(DATASET_DIR,'general_tsp')
    file_name = os.path.join('dataset.json')
    problem_weights = [0.1, 0.4, 0.4, 0.1] # probability of choosing a small, medium, large, or very large problem
    
    @classmethod
    def generate_n_samples(cls, n: int):
        # generate 15 cities as a default with random coordinates
        problems = []
        sizes = []
        for _ in range(n):
            # sample n_nodes
            n_nodes, category= cls._DatasetGenerator__weighted_sampling(cls.problem_weights)
            sizes.append(category)
            problems.append(GraphV1Problem(n_nodes=n_nodes, directed=True))
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

    print('\n\n\n________________________')
    print('Testing GeneralTSPGenerator')
    GeneralTSPGenerator.generate_and_save_dataset(100)
    dataset = MetricTSPGenerator.load_dataset()
    sample_problem = dataset['problems'][0]
    print(type(sample_problem))
    print('Sample general tsp path: ', end='')
    print(asyncio.run(solver.solve_problem(sample_problem)))
