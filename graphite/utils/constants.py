# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2023 Graphite-AI

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

from graphite.solvers import NearestNeighbourSolver, NearestNeighbourMultiSolver, BeamSearchSolver, HPNSolver, DPSolver
from graphite.utils.graph_utils import get_tour_distance, get_multi_minmax_tour_distance

BENCHMARK_SOLUTIONS = {
    'Metric TSP': NearestNeighbourSolver,
    'General TSP': NearestNeighbourSolver,
    'Metric mTSP': NearestNeighbourMultiSolver,
    'General mTSP': NearestNeighbourMultiSolver
} # mapping benchmark solvers to each problem

COST_FUNCTIONS = {
    'Metric TSP': get_tour_distance,
    'General TSP': get_tour_distance,
    'Metric mTSP': get_multi_minmax_tour_distance,
    'General mTSP': get_multi_minmax_tour_distance
}

HEURISTIC_SOLVERS = [NearestNeighbourSolver, BeamSearchSolver, HPNSolver]

EXACT_SOLVERS = [DPSolver]
