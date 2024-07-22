from enum import Enum

class ProblemType(Enum):
    METRIC_TSP = 'Metric TSP'
    GENERAL_TSP = 'General TSP'

class ObjectiveFunction(Enum):
    MIN = 'min'
    MAX = 'max'