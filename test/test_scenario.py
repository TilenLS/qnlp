import pytest
import numpy as np
from contextuality.model import Scenario, CyclicScenario

@pytest.fixture
def triangluar_scenario():
    contexts = [('x1', 'x2'), ('x2', 'x3'), ('x3', 'x1')]
    num_outcome = 2
    return Scenario(contexts, num_outcome)

def test_incidence_matrix(triangluar_scenario):
    expected = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1],
                         [1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1],
                         [1, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 1]])
    assert np.all(triangluar_scenario.incidence_matrix == expected)

def test_observables(triangluar_scenario):
    expected = ['x1', 'x2', 'x3']
    assert triangluar_scenario.observables == expected

def test_cyclic_scenario():
    observables = ['x1', 'x2', 'x3']
    expected = [('x1', 'x2'), ('x2', 'x3'), ('x3', 'x1')]
    num_outcome = 2

    s = CyclicScenario(observables, num_outcome)
    assert s.contexts == expected

def test_scenario_eq(triangluar_scenario):
    contexts = [('x1', 'x2'), ('x2', 'x3'), ('x3', 'x1')]
    num_outcome = 2
    s = Scenario(contexts, num_outcome)
    assert triangluar_scenario == triangluar_scenario
    assert triangluar_scenario == s

def test_scenario_repr(triangluar_scenario):
    assert eval(triangluar_scenario.__repr__()) == triangluar_scenario

def test_scneario_str(triangluar_scenario):
    string = triangluar_scenario.__str__()
    for ob in triangluar_scenario.observables:
        assert ob in string

