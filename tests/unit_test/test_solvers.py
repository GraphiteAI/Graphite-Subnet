'''
This script tests all the solvers against each other. Note that this test takes while to execute.
'''

import unittest
from graphite.solvers import NearestNeighbourSolver, BeamSearchSolver, DPSolver, HPNSolver
from graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
from graphite.protocol import GraphV1Synapse, GraphV1Problem
from graphite.validator.reward import ScoreResponse
from graphite.utils.graph_utils import is_valid_solution
import asyncio

class TestSolvers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.metric_tsp = GraphV1Problem(n_nodes=10)
        self.large_tsp = GraphV1Problem(n_nodes=100)
        self.general_tsp = GraphV1Problem(n_nodes=10, directed=True)
    
    async def test_dp_solver(self):
        solver = DPSolver()
        metric_solution = await solver.solve_problem(self.metric_tsp)
        general_solution = await solver.solve_problem(self.general_tsp)
        large_solution = await solver.solve_problem(self.large_tsp)
        self.assertEqual(is_valid_solution(self.metric_tsp, metric_solution),True)
        self.assertEqual(is_valid_solution(self.general_tsp, general_solution),True)
        self.assertEqual(is_valid_solution(self.large_tsp, large_solution),False)

    async def test_greedy_solver(self):
        solver = NearestNeighbourSolver()
        metric_solution = await solver.solve_problem(self.metric_tsp)
        general_solution = await solver.solve_problem(self.general_tsp)
        large_solution = await solver.solve_problem(self.large_tsp)
        self.assertEqual(is_valid_solution(self.metric_tsp, metric_solution),True)
        self.assertEqual(is_valid_solution(self.general_tsp, general_solution),True)
        self.assertEqual(is_valid_solution(self.large_tsp, large_solution),True)

    async def test_greedy_solver(self):
        solver = NearestNeighbourSolverVali()
        metric_solution = await solver.solve_problem(self.metric_tsp)
        general_solution = await solver.solve_problem(self.general_tsp)
        large_solution = await solver.solve_problem(self.large_tsp)
        self.assertEqual(is_valid_solution(self.metric_tsp, metric_solution),True)
        self.assertEqual(is_valid_solution(self.general_tsp, general_solution),True)
        self.assertEqual(is_valid_solution(self.large_tsp, large_solution),True)

    async def test_beam_solver(self):
        solver = BeamSearchSolver()
        metric_solution = await solver.solve_problem(self.metric_tsp)
        general_solution = await solver.solve_problem(self.general_tsp)
        large_solution = await solver.solve_problem(self.large_tsp)
        self.assertEqual(is_valid_solution(self.metric_tsp, metric_solution),True)
        self.assertEqual(is_valid_solution(self.general_tsp, general_solution),True)
        self.assertEqual(is_valid_solution(self.large_tsp, large_solution),True)

    async def test_hpn_solver(self):
        solver = HPNSolver()
        metric_solution = await solver.solve_problem(self.metric_tsp)
        general_solution = await solver.solve_problem(self.general_tsp)
        large_solution = await solver.solve_problem(self.large_tsp)
        self.assertEqual(is_valid_solution(self.metric_tsp, metric_solution),True)
        self.assertEqual(is_valid_solution(self.general_tsp, general_solution),False)
        self.assertEqual(is_valid_solution(self.large_tsp, large_solution),True)

    async def test_solvers(self):
        # check to assert that heuristics are not better than exact solver
        solvers = [DPSolver(), NearestNeighbourSolver(), BeamSearchSolver(), HPNSolver()]
        metric_solutions = {solver.__class__.__name__: await solver.solve_problem(self.metric_tsp) for solver in solvers}
        metric_synapses = {solver_type: GraphV1Synapse(problem=self.metric_tsp, solution=solution) for solver_type, solution in metric_solutions.items()}
        general_solutions = {solver.__class__.__name__: await solver.solve_problem(self.general_tsp) for solver in solvers}
        general_synapses = {solver_type: GraphV1Synapse(problem=self.general_tsp, solution=solution) for solver_type, solution in general_solutions.items()}
        large_solutions = {solver.__class__.__name__: await solver.solve_problem(self.large_tsp) for solver in solvers}
        large_synapses = {solver_type: GraphV1Synapse(problem=self.large_tsp, solution=solution) for solver_type, solution in large_solutions.items()}
        metric_score_handler = ScoreResponse(GraphV1Synapse(problem=self.metric_tsp))
        await metric_score_handler.get_benchmark()
        general_score_handler = ScoreResponse(GraphV1Synapse(problem=self.general_tsp))
        await general_score_handler.get_benchmark()
        large_score_handler = ScoreResponse(GraphV1Synapse(problem=self.large_tsp))
        await large_score_handler.get_benchmark()
        metric_scores = {solver_name: metric_score_handler.get_score(synapse) for solver_name, synapse in metric_synapses.items()}
        general_scores = {solver_name: general_score_handler.get_score(synapse) for solver_name, synapse in general_synapses.items()}
        large_scores = {solver_name: large_score_handler.get_score(synapse) for solver_name, synapse in large_synapses.items()}

        # For the metric and the general scoring, we want to assert 
        for compared_solver in [NearestNeighbourSolver, BeamSearchSolver, HPNSolver]:
            self.assertLessEqual(round(metric_scores[DPSolver.__name__],5), round(metric_scores[compared_solver.__name__],5))
            self.assertLessEqual(round(general_scores[DPSolver.__name__],5), round(general_scores[compared_solver.__name__],5))
            self.assertGreater(large_scores[DPSolver.__name__], large_scores[compared_solver.__name__])

if __name__=="__main__":
    unittest.main()

