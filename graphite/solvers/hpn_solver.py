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
from graphite.protocol import GraphV1Problem
from graphite.models.hybrid_pointer_network import HPN
from graphite.utils.graph_utils import normalize_coordinates, timeout
from scipy.spatial import distance
import numpy as np
import torch
import time
import asyncio

class HPNSolver(BaseSolver):
    '''
    implement solve method and necessary transformations
    '''
    def __init__(self, problem_types:List[Union[GraphV1Problem]]=[GraphV1Problem(n_nodes=2)], weights_fp:str = 'graphite/models/model_weights/hpn_base_model.pkl'):
        super().__init__(problem_types=problem_types)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # assign device to run the model on
        self.critic = HPN(n_feature=2, n_hidden=128) # instantiate model to handle metric 2-d geographic TSP problems
        self.critic = self.critic.to(self.device)
        self.critic.eval() # set the model to evaluate
        checkpoint = torch.load(weights_fp, map_location=self.device)
        self.critic.load_state_dict(checkpoint['model_baseline']) # load in base model

    def post_process(self,B,route,graph,size,visited_indices):
        # apply 2-opt post-processing to find local optima based on the solution provided by the pointer network
        # this is an O(n^2) procedure
        best_solutions = []
        for b in range(B):
            best = route.copy()
            graph_ = graph[:, b, :].copy()
            dmatrix = distance.cdist(graph_, graph_, 'euclidean')
            improved = True

            while improved:
                improved = False
                for i in range(size):
                    for j in range(i + 2, size + 1):
                        old_dist = dmatrix[best[i], best[i + 1]] + dmatrix[best[j], best[j - 1]]
                        new_dist = dmatrix[best[j], best[i + 1]] + dmatrix[best[i], best[j - 1]]
                        if new_dist < old_dist:
                            best[i + 1:j] = best[j - 1:i:-1]
                            improved = True

            new_tour_len = 0
            for k in range(size):
                new_tour_len += dmatrix[best[k], best[k + 1]]  # Calculate the length of the tour for this batch
                best_solutions.append(best)

        # convert the visited indices to 1-d list
        best_indices = [visited_indices[idx] for idx in best_solutions[0]]
        return best_indices

    async def solve(self, formatted_problem, future_id:int, post_process:bool=False)->List[int]:
        coordinates = formatted_problem
        size = len(coordinates)
        B = 1 # batch size set to 1 as default
        X = torch.from_numpy(coordinates).float().to(self.device)
        X = X.unsqueeze(0)
        mask = torch.zeros(B,size).to(self.device)

        solution = []

        Y = X.view(B, size, 2)  # to the same batch size
        x = Y[:, 0, :]
        h = None
        c = None
        Transcontext = None
        visited_indices = []  # To store the sequence of node indices

        for k in range(size):
            if self.future_tracker.get(future_id):
                return None
            Transcontext, output, h, c, _ = self.critic(Transcontext, x=x, X_all=X, h=h, c=c, mask=mask)
            idx = torch.argmax(output, dim=1)
            x = Y[[i for i in range(B)], idx.data]
            solution.append(x.cpu().numpy())
            visited_indices.append(idx.cpu().numpy())  # Store the index of the chosen node
            mask[[i for i in range(B)], idx.data] += -np.inf

        solution.append(solution[0])
        visited_indices.append(visited_indices[0])  # Return to the start index
        graph = np.array(solution)
        route = [x for x in range(size)] + [0]

        if post_process==True:
            # we apply the post-processing step to possibly obtain a better solution
            best_indices = self.post_process(B,route,graph,size,visited_indices)
            tour_1d = [int(arr[0]) for arr in best_indices]
        else:
            # convert the visited indices to 1-d list
            tour_1d = [int(arr[0]) for arr in visited_indices]
        # translate the start city to 0 for consistency with other TSP solver solutions (which start at the 0th index)
        start_index = tour_1d.index(0)
        min_path = tour_1d[start_index:] + tour_1d[1:start_index+1]
        return min_path

    def problem_transformations(self, problem: Union[GraphV1Problem]):
        # normalize values in the coordinates
        formatted_problem = normalize_coordinates(problem.nodes)
        return formatted_problem
        
if __name__=='__main__':
    n_nodes = 1000
    test_problem = GraphV1Problem(n_nodes=n_nodes)
    solver = HPNSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
