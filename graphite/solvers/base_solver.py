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
from graphite.protocol import GraphProblem
import bittensor as bt
import asyncio

DEFAULT_SOLVER_TIMEOUT = 30

class BaseSolver(ABC):
    def __init__(self, problem_variants:List[str]):
        self.problem_types = [problem.problem_type for problem in problem_variants] # defining what problems the solver is equipped to solve
    
    #TODO evolve the abstract method to handle the different problem classes and objective functions
    @abstractmethod
    async def solve(self, formatted_problem, *args, **kwargs)->List[int]:
        '''
        Abstract class that handles the solving of GraphProblems contained within the Synapse.

        Solvers can be developed to handle multiple types of graph traversal problems.

        It takes a formatted_problem (post-transformation) as an input and returns the optimal path based on the objective function (optional)
        '''
        ...
    
    @abstractmethod
    def problem_transformations(self, problem: GraphProblem):
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
    
    async def solve_problem(self, problem: GraphProblem, timeout:int=DEFAULT_SOLVER_TIMEOUT):
        '''
        This method implements the security checks
        Then it makes the necessary transformations to the problem
        and passes it on to the solve method

        Checks for the integrity of the data (that the problem is legitimate) are handled outside the forward function
        '''
        if self.is_valid_problem(problem):
            task = self.loop.create_task(self.solve(self.problem_transformations(problem)))
            result = self.loop.run_until_complete(
                asyncio.wait_for(task, timeout=timeout)
                )
            return result
        else:
            # solver is unable to solve the given problem
            bt.logging.error(f"current solver: {self.__class__.__name__} cannot handle received problem: {problem.problem_type}")
            return False
