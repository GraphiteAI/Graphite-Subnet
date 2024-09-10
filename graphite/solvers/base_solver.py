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

from abc import ABC, abstractmethod
from typing import List
from graphite.utils.graph_utils import valid_problem, timeout
from graphite.protocol import GraphV1Problem, GraphV2Problem
import bittensor as bt
import asyncio
import concurrent.futures
import time
from typing import Union
import numpy as np

DEFAULT_SOLVER_TIMEOUT = 20

class BaseSolver(ABC):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]):
        self.problem_types = [problem.problem_type for problem in problem_types] # defining what problems the solver is equipped to solve
        self.future_tracker = {}
    
    #TODO evolve the abstract method to handle the different problem classes and objective functions
    @abstractmethod
    async def solve(self, formatted_problem, future_id, *args, **kwargs)->List[int]:
        '''
        Abstract class that handles the solving of GraphV1Problems contained within the Synapse.

        Solvers can be developed to handle multiple types of graph traversal problems.

        It takes a formatted_problem (post-transformation) as an input and returns the optimal path based on the objective function (optional)
        '''
        ...
    
    @abstractmethod
    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        '''
        This abstract class applies any necessary transformation to the problem to convert it to the form required for the solve method
        '''
        ...

    def is_valid_problem(self, problem):
        '''
        checks if the solver is supposed to be able to solve the given problem and that the problem specification is valid.
        Note that this does not guarantee the problem has a solution. For example: the TSP problem might be a partially connected graph with no hamiltonian cycle
        '''
        return valid_problem(problem) and problem.problem_type in self.problem_types

    async def solve_problem(self, problem: Union[GraphV1Problem, GraphV2Problem], timeout:int=DEFAULT_SOLVER_TIMEOUT):
        '''
        This method implements the security checks
        Then it makes the necessary transformations to the problem
        and passes it on to the solve method

        Checks for the integrity of the data (that the problem is legitimate) are handled outside the forward function
        '''
        if self.is_valid_problem(problem):

            future_id = id(problem)
            self.future_tracker[future_id] = False

            transformed_problem = self.problem_transformations(problem)
            
            loop = asyncio.get_running_loop()
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit the asynchronous task to the executor
                future = loop.run_in_executor(executor, lambda: asyncio.run(self.solve(transformed_problem,future_id)))
                try:
                    result = await asyncio.wait_for(future, timeout)
                    return result
                except asyncio.TimeoutError:
                    print(f"Task {future_id} timed out after: {time.time() - start_time}, with timeout set to {timeout}")
                    self.future_tracker[future_id] = True
                    return False
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
                    return False
        else:
            bt.logging.error(f"current solver: {self.__class__.__name__} cannot handle received problem: {problem.problem_type}")
            return False
