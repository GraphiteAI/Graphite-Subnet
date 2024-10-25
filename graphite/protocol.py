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
from typing import List, Union, Optional, Literal, Iterable
import numpy as np
import bittensor as bt
import pprint
import math
import json
import base64
import sys
import random

is_alive_path = "graphite/is_alive.json"
with open(is_alive_path, "r") as f:
    ISALIVE_SCHEMA = json.load(f)

rel_v1_path = "graphite/schema_v1.json"
with open(rel_v1_path, "r") as f:
    MODEL_V1_SCHEMA = json.load(f)

rel_v2_path = "graphite/schema_v2.json"
with open(rel_v2_path, "r") as f:
    MODEL_V2_SCHEMA = json.load(f)

class IsAlive(bt.Synapse):
    answer: Optional[str] = None
    completion: str = Field(
        "",
        title="Completion",
        description="Checks if axon is alive."
                    "This attribute is mutable and can be updated.",
    )

    def to_headers(self) -> dict:
        """
        Converts the state of a Synapse instance into a dictionary of HTTP headers.

        This method is essential for
        packaging Synapse data for network transmission in the Bittensor framework, ensuring that each key aspect of
        the Synapse is represented in a format suitable for HTTP communication.

        Process:

        1. Basic Information: It starts by including the ``name`` and ``timeout`` of the Synapse, which are fundamental for identifying the query and managing its lifespan on the network.
        2. Complex Objects: The method serializes the ``axon`` and ``dendrite`` objects, if present, into strings. This serialization is crucial for preserving the state and structure of these objects over the network.
        3. Encoding: Non-optional complex objects are serialized and encoded in base64, making them safe for HTTP transport.
        4. Size Metrics: The method calculates and adds the size of headers and the total object size, providing valuable information for network bandwidth management.

        Example Usage::

            synapse = Synapse(name="ExampleSynapse", timeout=30)
            headers = synapse.to_headers()
            # headers now contains a dictionary representing the Synapse instance

        Returns:
            dict: A dictionary containing key-value pairs representing the Synapse's properties, suitable for HTTP communication.
        """
        # Initializing headers with 'name' and 'timeout'
        headers = {"name": self.name, "timeout": str(self.timeout)}

        # Adding headers for 'axon' and 'dendrite' if they are not None
        if self.axon:
            headers.update(
                {
                    f"bt_header_axon_{k}": str(v)
                    for k, v in self.axon.model_dump().items()
                    if v is not None
                }
            )
        if self.dendrite:
            headers.update(
                {
                    f"bt_header_dendrite_{k}": str(v)
                    for k, v in self.dendrite.model_dump().items()
                    if v is not None
                }
            )

        # Getting the fields of the instance
        instance_fields = self.model_dump()

        required = ISALIVE_SCHEMA.get("required", [])
        # Iterating over the fields of the instance
        for field, value in instance_fields.items():
            # If the object is not optional, serializing it, encoding it, and adding it to the headers

            # Skipping the field if it's already in the headers or its value is None
            if field in headers or value is None:
                continue

            elif required and field in required:
                try:
                    # create an empty (dummy) instance of type(value) to pass pydantic validation on the axon side
                    serialized_value = json.dumps(value.__class__.__call__())
                    encoded_value = base64.b64encode(serialized_value.encode()).decode(
                        "utf-8"
                    )
                    headers[f"bt_header_input_obj_{field}"] = encoded_value
                except TypeError as e:
                    raise ValueError(
                        f"Error serializing {field} with value {value}. Objects must be json serializable."
                    ) from e

        # Adding the size of the headers and the total size to the headers
        headers["header_size"] = str(sys.getsizeof(headers))
        headers["total_size"] = str(self.get_total_size())
        headers["computed_body_hash"] = self.body_hash

        return headers

class GraphV2Problem(BaseModel):
    problem_type: Literal['Metric TSP', 'General TSP'] = Field('Metric TSP', description="Problem Type")
    objective_function: str = Field('min', description="Objective Function")
    visit_all: bool = Field(True, description="Visit All Nodes")
    to_origin: bool = Field(True, description="Return to Origin")
    n_nodes: conint(ge=2000, le=5000) = Field(2000, description="Number of Nodes (must be between 2000 and 5000)")
    selected_ids: List[int] = Field(default_factory=list, description="List of selected node positional indexes")
    cost_function: Literal['Geom', 'Euclidean2D', 'Manhatten2D', 'Euclidean3D', 'Manhatten3D'] = Field('Geom', description="Cost function")
    dataset_ref: Literal['Asia_MSB', 'World_TSP'] = Field('Asia_MSB', description="Dataset reference file")
    nodes: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Node Coordinates")  # If not none, nodes represent the coordinates of the cities
    edges: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Edge Weights")  # If not none, this represents a square matrix of edges where edges[source;row][destination;col] is the cost of a given edge
    directed: bool = Field(False, description="Directed Graph")  # boolean for whether the graph is directed or undirected / Symmetric or Asymmetric
    simple: bool = Field(True, description="Simple Graph")  # boolean for whether the graph contains any degenerate loop
    weighted: bool = Field(False, description="Weighted Graph")  # boolean for whether the value in the edges matrix represents cost
    repeating: bool = Field(False, description="Allow Repeating Nodes")  # boolean for whether the nodes in the problem can be revisited

    ### Expensive check only needed for organic requests
    # @model_validator(mode='after')
    # def unique_select_ids(self):
    #     # ensure all selected ids are unique
    #     self.selected_ids = list(set(self.selected_ids))

    #     # ensure the selected_ids are < len(file)
    #     with np.load(f"dataset/{self.dataset_ref}.npz") as f:
    #         node_coords_np = np.array(f['data'])
    #         largest_possible_id = len(node_coords_np) - 1

    #     self.selected_ids = [id for id in self.selected_ids if id <= largest_possible_id]
    #     self.n_nodes = len(self.selected_ids)

    #     return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric TSP', 'General TSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
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

# Constants for problem formulation
MAX_SALESMEN = 10

class GraphV2ProblemMulti(GraphV2Problem):
    problem_type: Literal['Metric mTSP', 'General mTSP'] = Field('Metric mTSP', description="Problem Type")
    n_nodes: conint(ge=500, le=2000) = Field(500, description="Number of Nodes (must be between 500 and 2000) for mTSP")
    n_salesmen: conint(ge=2, le=MAX_SALESMEN) = Field(2, description="Number of Salesmen in the mTSP formulation")
    # Note that in this initial problem formulation, we will start with a single depot structure
    single_depot: bool = Field(True, description="Whether problem is a single or multi depot formulation")
    depots: List[int] = Field([0,0], description="List of selected 'city' indices for which the respective salesmen paths begin")
    dataset_ref: Literal['Asia_MSB', 'World_TSP'] = Field('Asia_MSB', description="Dataset reference file")

    ### Expensive check only needed for organic requests
    # @model_validator(mode='after')
    # def unique_select_ids(self):
    #     # ensure all selected ids are unique
    #     self.selected_ids = list(set(self.selected_ids))

    #     # ensure the selected_ids are < len(file)
    #     with np.load(f"dataset/{self.dataset_ref}.npz") as f:
    #         node_coords_np = np.array(f['data'])
    #         largest_possible_id = len(node_coords_np) - 1

    #     self.selected_ids = [id for id in self.selected_ids if id <= largest_possible_id]
    #     self.n_nodes = len(self.selected_ids)

    #     return self
    @model_validator(mode='after')
    def assert_salesmen_depot(self):
        assert len(self.depots) == self.n_salesmen, ValueError('Number of salesmen must match number of depots')
        return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric mTSP', 'General mTSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
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


class GraphV2Synapse(bt.Synapse):
    '''
    Implement necessary serialization and deserialization checks
    '''
    problem: Union[GraphV2Problem, GraphV2ProblemMulti]
    solution: Optional[Union[List[List[int]], List[int], bool]] = None

    def to_headers(self) -> dict:
        """
        Converts the state of a Synapse instance into a dictionary of HTTP headers.

        This method is essential for
        packaging Synapse data for network transmission in the Bittensor framework, ensuring that each key aspect of
        the Synapse is represented in a format suitable for HTTP communication.

        Process:

        1. Basic Information: It starts by including the ``name`` and ``timeout`` of the Synapse, which are fundamental for identifying the query and managing its lifespan on the network.
        2. Complex Objects: The method serializes the ``axon`` and ``dendrite`` objects, if present, into strings. This serialization is crucial for preserving the state and structure of these objects over the network.
        3. Encoding: Non-optional complex objects are serialized and encoded in base64, making them safe for HTTP transport.
        4. Size Metrics: The method calculates and adds the size of headers and the total object size, providing valuable information for network bandwidth management.

        Example Usage::

            synapse = Synapse(name="ExampleSynapse", timeout=30)
            headers = synapse.to_headers()
            # headers now contains a dictionary representing the Synapse instance

        Returns:
            dict: A dictionary containing key-value pairs representing the Synapse's properties, suitable for HTTP communication.
        """
        # Initializing headers with 'name' and 'timeout'
        headers = {"name": self.name, "timeout": str(self.timeout)}

        # Adding headers for 'axon' and 'dendrite' if they are not None
        if self.axon:
            headers.update(
                {
                    f"bt_header_axon_{k}": str(v)
                    for k, v in self.axon.model_dump().items()
                    if v is not None
                }
            )
        if self.dendrite:
            headers.update(
                {
                    f"bt_header_dendrite_{k}": str(v)
                    for k, v in self.dendrite.model_dump().items()
                    if v is not None
                }
            )

        # Getting the fields of the instance
        instance_fields = self.model_dump()
        required = MODEL_V2_SCHEMA.get("required", [])
        # Iterating over the fields of the instance
        for field, value in instance_fields.items():
            # If the object is not optional, serializing it, encoding it, and adding it to the headers

            # Skipping the field if it's already in the headers or its value is None
            if field in headers or value is None:
                continue

            elif required and field in required:
                try:
                    # create an empty (dummy) instance of type(value) to pass pydantic validation on the axon side
                    serialized_value = json.dumps(value.__class__.__call__())
                    encoded_value = base64.b64encode(serialized_value.encode()).decode(
                        "utf-8"
                    )
                    headers[f"bt_header_input_obj_{field}"] = encoded_value
                except TypeError as e:
                    raise ValueError(
                        f"Error serializing {field} with value {value}. Objects must be json serializable."
                    ) from e

        # Adding the size of the headers and the total size to the headers
        headers["header_size"] = str(sys.getsizeof(headers))
        headers["total_size"] = str(self.get_total_size())
        headers["computed_body_hash"] = self.body_hash

        return headers

class GraphV1Problem(BaseModel):
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

class GraphV1Synapse(bt.Synapse):
    '''
    Implement necessary serialization and deserialization checks
    '''
    problem: GraphV1Problem
    solution: Optional[Union[List[int], bool]] = None

    def to_headers(self) -> dict:
        """
        Converts the state of a Synapse instance into a dictionary of HTTP headers.

        This method is essential for
        packaging Synapse data for network transmission in the Bittensor framework, ensuring that each key aspect of
        the Synapse is represented in a format suitable for HTTP communication.

        Process:

        1. Basic Information: It starts by including the ``name`` and ``timeout`` of the Synapse, which are fundamental for identifying the query and managing its lifespan on the network.
        2. Complex Objects: The method serializes the ``axon`` and ``dendrite`` objects, if present, into strings. This serialization is crucial for preserving the state and structure of these objects over the network.
        3. Encoding: Non-optional complex objects are serialized and encoded in base64, making them safe for HTTP transport.
        4. Size Metrics: The method calculates and adds the size of headers and the total object size, providing valuable information for network bandwidth management.

        Example Usage::

            synapse = Synapse(name="ExampleSynapse", timeout=30)
            headers = synapse.to_headers()
            # headers now contains a dictionary representing the Synapse instance

        Returns:
            dict: A dictionary containing key-value pairs representing the Synapse's properties, suitable for HTTP communication.
        """
        # Initializing headers with 'name' and 'timeout'
        headers = {"name": self.name, "timeout": str(self.timeout)}

        # Adding headers for 'axon' and 'dendrite' if they are not None
        if self.axon:
            headers.update(
                {
                    f"bt_header_axon_{k}": str(v)
                    for k, v in self.axon.model_dump().items()
                    if v is not None
                }
            )
        if self.dendrite:
            headers.update(
                {
                    f"bt_header_dendrite_{k}": str(v)
                    for k, v in self.dendrite.model_dump().items()
                    if v is not None
                }
            )

        # Getting the fields of the instance
        instance_fields = self.model_dump()

        required = MODEL_V1_SCHEMA.get("required", [])
        # Iterating over the fields of the instance
        for field, value in instance_fields.items():
            # If the object is not optional, serializing it, encoding it, and adding it to the headers

            # Skipping the field if it's already in the headers or its value is None
            if field in headers or value is None:
                continue

            elif required and field in required:
                try:
                    # create an empty (dummy) instance of type(value) to pass pydantic validation on the axon side
                    serialized_value = json.dumps(value.__class__.__call__())
                    encoded_value = base64.b64encode(serialized_value.encode()).decode(
                        "utf-8"
                    )
                    headers[f"bt_header_input_obj_{field}"] = encoded_value
                except TypeError as e:
                    raise ValueError(
                        f"Error serializing {field} with value {value}. Objects must be json serializable."
                    ) from e

        # Adding the size of the headers and the total size to the headers
        headers["header_size"] = str(sys.getsizeof(headers))
        headers["total_size"] = str(self.get_total_size())
        headers["computed_body_hash"] = self.body_hash

        return headers


if __name__=="__main__":
    # # run to check that each field behaves in the correct manner
    # print(f"Testing random metric initialization")
    # print(f"_____________________________")
    # random_metric_tsp = GraphV1Problem(n_nodes=8)
    # pprint.pprint(random_metric_tsp.get_info(3))
    # output = GraphV1Synapse(problem=random_metric_tsp)
    # print(type(random_metric_tsp))
    # print(type(output))

    # print('\n\n_____________________________')
    # print(f"Testing fixed coordinate initialization")
    # metric_tsp = GraphV1Problem(n_nodes=3, nodes=[[1,0],[2,0],[3,5]])
    # pprint.pprint(metric_tsp.get_info(3))

    # print('\n\n_____________________________')
    # print(f"Testing error infinite coordinate initialization")
    # try:
    #     metric_tsp = GraphV1Problem(n_nodes=3, nodes=[[1,np.inf],[2,0],[3,5]])
    # except ValidationError as e:
    #     print(e)

    # print('\n\n_____________________________')
    # print(f"Testing erroneous edges input initialization")
    # try:
    #     false_metric_tsp = GraphV1Problem(n_nodes=3, edges=[[1,0,5],[2,0,7],[3,5,2]])
    # except ValueError as e:
    #     print(e)

    # print('\n\n_____________________________')
    # print(f"Testing erroneous nodes input initialization")
    # try:
    #     false_metric_tsp = GraphV1Problem(n_nodes=2, nodes=[[1,5],[2,7],[3,2]])
    #     pprint.pprint(false_metric_tsp.get_info(3))
    # except ValidationError as e:
    #     print(e)
    
    # print('\n\n_____________________________')
    # print(f"Testing erroneous coordinate input initialization")
    # try:
    #     false_metric_tsp = GraphV1Problem(n_nodes=3, nodes=[[1,2,5],[2,6,7],[3,1,2]])
    #     pprint.pprint(false_metric_tsp.get_info(3))
    # except ValidationError as e:
    #     print(e)

    # print('\n\n_____________________________')
    # print(f"Testing negative coordinate input initialization")
    # try:
    #     false_metric_tsp = GraphV1Problem(n_nodes=3, nodes=[[-1,5],[2,7],[-3.2,2]])
    #     pprint.pprint(false_metric_tsp.get_info(3))
    # except ValidationError as e:
    #     print(e)

    # print('\n\n_____________________________')
    # print(f"Testing enforce objective function")
    # try:
    #     false_metric_tsp = GraphV1Problem(n_nodes=3, objective_function='max')
    #     pprint.pprint(false_metric_tsp.get_info(3))
    # except ValueError as e:
    #     print(e)

    print('\n\n_____________________________')
    print(f"Testing creating graph problem large")
    loaded_datasets = {}
    with np.load('dataset/Asia_MSB.npz') as f: # "dataset/Asia_MSB.npz"
        node_coords_np = f['data']
    loaded_datasets["Asia_MSB"] = np.array(node_coords_np)
    # determine the number of nodes to select
    n_nodes = random.randint(2000, 5000)
    # randomly select n_nodes indexes from the selected graph
    prob_select = random.randint(0, len(list(loaded_datasets.keys()))-1)
    selected_node_idxs = random.sample(range(len(loaded_datasets[list(loaded_datasets.keys())[prob_select]])), n_nodes)
    large_metric_tsp = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Euclidean", dataset_ref="Asia_MSB")
    print(large_metric_tsp)