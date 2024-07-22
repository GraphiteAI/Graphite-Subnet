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

from pydantic import BaseModel, Field, model_validator, conint, confloat, ValidationError, field_validator
from typing import List, Union, Optional, Literal
import numpy as np
import bittensor as bt
import pprint
import math

class IsAlive(bt.Synapse):
    answer: Optional[str] = None
    completion: str = Field(
        "",
        title="Completion",
        description="Checks if axon is alive."
                    "This attribute is mutable and can be updated.",
    )

class GraphProblem(BaseModel):
    problem_type: Literal['Metric TSP', 'General TSP'] = Field('Metric TSP', description="Problem Type")
    objective_function: str = Field('min', description="Objective Function")
    visit_all: bool = Field(True, description="Visit All Nodes")
    to_origin: bool = Field(True, description="Return to Origin")
    n_nodes: conint(ge=2) = Field(10, description="Number of Nodes (must be >= 2)")
    nodes: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]],None] = Field(default_factory=list, description="Node Coordinates")  # If not none, nodes represent the coordinates of the cities
    edges: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]],None] = Field(default_factory=list, description="Edge Weights")  # If not none, this represents a square matrix of edges where edges[source;row][destination;col] is the cost of a given edge
    directed: bool = Field(False, description="Directed Graph")  # boolean for whether the graph is directed or undirected / Symmetric or Asymmetric
    simple: bool = Field(True, description="Simple Graph")  # boolean for whether the graph contains any degenerate loop
    weighted: bool = Field(False, description="Weighted Graph")  # boolean for whether the value in the edges matrix represents cost
    repeating: bool = Field(False, description="Allow Repeating Nodes")  # boolean for whether the nodes in the problem can be revisited

    @model_validator(mode='after') # checks and generates missing data if not passed in by user
    def initialize_nodes_and_edges(self):
        if self.directed == False: # we should generate a set of coordinates and corresponding distances for this Metric TSP
            if not self.nodes and not self.edges: # we do not want to allow setting of edges for metric TSP... only setting coordinates
                self.nodes = self.generate_random_coordinates(self.n_nodes)
                self.edges = self.get_distance_matrix(self.nodes)
            elif self.edges and not self.nodes: # convert problem into a General TSP and solve instead
                raise ValueError("Undirected graph should have defined nodes(coordinates) instead of edges")
            elif not self.edges and self.nodes:
                self.edges = self.get_distance_matrix(self.nodes)
        else: # we need to generate a set of random edge weights for this General TSP
            self.problem_type = 'General TSP'
            if not self.edges:
                self.edges = self.generate_edges(self.n_nodes)
        return self

    @model_validator(mode='after')
    def validate_length(self):
        if not self.directed and len(self.nodes) != self.n_nodes:
            raise ValueError(f"The length of nodes {len(self.nodes)} does not match n_nodes {self.n_nodes}")
        
        if not self.directed and not all(len(row)==2 for row in self.nodes):
            raise ValueError(f"the input coordinates are not all 2-dimensional")

        if len(self.edges) != self.n_nodes:
            raise ValueError(f"The length of edges {len(self.edges)} does not match n_nodes {self.n_nodes}")

        if not all(len(row)==self.n_nodes for row in self.edges):
            raise ValueError(f"The length of edges {len(self.edges)} does not match n_nodes {self.n_nodes}")
        return self
    
    @model_validator(mode='after')
    def validate_values(self):
        if self.directed:
            # check edges; Non-diagonal must be valid number
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        assert isinstance(self.edges[i][j], (float, int)), TypeError(f"Found value in edges at index {(i,j)} of type {type(self.edges[i][j])}")
                        assert (math.isfinite(self.edges[i][j]) and np.isfinite(self.edges[i][j])), ValueError(f"Found non-finite value in edges at index {(i,j)} of value {self.edges[i][j]}")
        else:
            # check nodes
            for i, node in enumerate(self.nodes):
                for j, coordinate_val in enumerate(node):
                    assert isinstance(coordinate_val, (float, int)), TypeError(f"Found value in nodes at index {(i,j)} of type {type(coordinate_val)}")
                    assert (math.isfinite(self.edges[i][j]) and np.isfinite(self.edges[i][j])), ValueError(f"Found non-finite value in nodes at index {(i,j)} of value {coordinate_val}")
        return self
    
    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric TSP', 'General TSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self

    def generate_random_coordinates(self, n_cities, grid_size=1000):
        # Create a grid of coordinates
        x = np.arange(grid_size)
        y = np.arange(grid_size)
        xv, yv = np.meshgrid(x, y)
        
        coordinates = np.column_stack((xv.ravel(), yv.ravel()))
        
        # Sample n_cities coordinates from the grid
        sampled_indices = np.random.choice(coordinates.shape[0], n_cities, replace=False)
        sampled_coordinates = coordinates[sampled_indices]
        np.random.shuffle(sampled_coordinates)

        # Convert the array to a list of lists of integers
        sampled_coordinates_list = sampled_coordinates.tolist()
        sampled_coordinates_list = [[int(coord) for coord in pair] for pair in sampled_coordinates_list]
        
        return sampled_coordinates_list
    
    def get_distance_matrix(self, coordinates:List[List[int]]):
        '''
        Receives an array of the coordinate values and returns a square numpy array of euclidean distances
        '''
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    distance_matrix[i, j] = np.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2)
        distance_list = distance_matrix.tolist()
        distance_list = [[float(x) for x in row] for row in distance_list]
        return distance_list
    
    @staticmethod
    def generate_edges(num_cities, min_weight=1, max_weight=100):
        # Initialize a matrix with zeros
        edges = np.zeros((num_cities, num_cities), dtype=int)
        for i in range(num_cities):
            for j in range(0, num_cities):
                weight = np.random.randint(min_weight, max_weight)
                edges[i][j] = weight
        edges_list = edges.tolist()
        edges_list = [[int(cost) for cost in row] for row in edges_list]
        return edges_list
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

class GraphSynapse(bt.Synapse):
    '''
    Implement necessary serialization and deserialization checks
    '''
    problem: GraphProblem
    solution: Optional[Union[List[int], bool]] = None
    
if __name__=="__main__":
    # run to check that each field behaves in the correct manner
    print(f"Testing random metric initialization")
    print(f"_____________________________")
    random_metric_tsp = GraphProblem(n_nodes=8)
    pprint.pprint(random_metric_tsp.get_info(3))
    output = GraphSynapse(problem=random_metric_tsp)
    print("HEREEEE", output)
    print(type(random_metric_tsp))
    print(type(output))

    print('\n\n_____________________________')
    print(f"Testing fixed coordinate initialization")
    metric_tsp = GraphProblem(n_nodes=3, nodes=[[1,0],[2,0],[3,5]])
    pprint.pprint(metric_tsp.get_info(3))

    print('\n\n_____________________________')
    print(f"Testing error infinite coordinate initialization")
    try:
        metric_tsp = GraphProblem(n_nodes=3, nodes=[[1,np.inf],[2,0],[3,5]])
    except ValidationError as e:
        print(e)

    print('\n\n_____________________________')
    print(f"Testing erroneous edges input initialization")
    try:
        false_metric_tsp = GraphProblem(n_nodes=3, edges=[[1,0,5],[2,0,7],[3,5,2]])
    except ValueError as e:
        print(e)

    print('\n\n_____________________________')
    print(f"Testing erroneous nodes input initialization")
    try:
        false_metric_tsp = GraphProblem(n_nodes=2, nodes=[[1,5],[2,7],[3,2]])
        pprint.pprint(false_metric_tsp.get_info(3))
    except ValidationError as e:
        print(e)
    
    print('\n\n_____________________________')
    print(f"Testing erroneous coordinate input initialization")
    try:
        false_metric_tsp = GraphProblem(n_nodes=3, nodes=[[1,2,5],[2,6,7],[3,1,2]])
        pprint.pprint(false_metric_tsp.get_info(3))
    except ValidationError as e:
        print(e)

    print('\n\n_____________________________')
    print(f"Testing negative coordinate input initialization")
    try:
        false_metric_tsp = GraphProblem(n_nodes=3, nodes=[[-1,5],[2,7],[-3.2,2]])
        pprint.pprint(false_metric_tsp.get_info(3))
    except ValidationError as e:
        print(e)

    print('\n\n_____________________________')
    print(f"Testing enforce objective function")
    try:
        false_metric_tsp = GraphProblem(n_nodes=3, objective_function='max')
        pprint.pprint(false_metric_tsp.get_info(3))
    except ValueError as e:
        print(e)
