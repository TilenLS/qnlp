import pytest
from pytest import approx
from contextuality.model import Scenario, CyclicScenario, random_pr_like_model, random_model
from contextuality.utils import sum_odd

def test_probability():
    model = random_pr_like_model(n=3)
    assert model._distributions[0][0] == model.probability(('x1', 'x2'), (0, 0))
    assert model._distributions[0][1] == model.probability(('x1', 'x2'), (0, 1))
    assert model._distributions[0][2] == model.probability(('x1', 'x2'), (1, 0))
    assert model._distributions[0][3] == model.probability(('x1', 'x2'), (1, 1))

    assert model._distributions[1][0] == model.probability(('x2', 'x3'), (0, 0))
    assert model._distributions[2][0] == model.probability(('x3', 'x1'), (0, 0))

def test_CbD_direction_influence():
    for _ in range(10):
        model = random_pr_like_model(n=3)
        dist = model._distributions
        assert approx(model.CbD_direct_influence()) == (2 * (abs(dist[0][0] - dist[1][0]) + 
            abs(dist[1][0] - dist[2][1]) + 
            abs(dist[2][1] + dist[0][0] - 1)))

def test_CbD_measure():
    for _ in range(10):
        model = random_pr_like_model(n=3)
        dist = model._distributions
        assert approx(model.CbD_measure()) == 2 - (2 * (abs(dist[0][0] - dist[1][0]) + 
            abs(dist[1][0] - dist[2][1]) + 
            abs(dist[2][1] + dist[0][0] - 1)))

    import numpy as np
    def get_s1(model, ctx):
        """
        Calculates the s_odd term in the CbD inequality.
        """
        n = len(ctx)
        prods = [np.sum(np.array([[1,-1], [-1,1]])*np.asarray(model[ii]).reshape((2,2))) for ii in range(n)]
        return sum_odd(prods)

    scenario = CyclicScenario(['x1', 'x2', 'x3'], 2)
    n = len(scenario.contexts)
    for _ in range(20):
        model = random_model(scenario)
        Deltas = model.CbD_direct_influence()
        s1s = get_s1(model._distributions, scenario.contexts)
        return n - 2 + Deltas - s1s

def test_sum_odd():
    import numpy as np 
    def s_odd(the_list):
        """Calculate the s_odd function of the list"""
        size = len(the_list)
        max_number_neg = int((size + np.mod(size,2))/2)
        sorted_list = np.sort(the_list)
        sign = np.array([1 for n in range(size)])
        sign[0] = -1
        s_odd_value = np.sum(sign*sorted_list)
        for k in range(max_number_neg):
            odd = 2*k+1
            sign = [int(n>=odd) - np.mod(int(n>=odd)+1,2) for n in range(size)]
            new_value = np.sum(sign*sorted_list)
            if new_value>=s_odd_value:
                s_odd_value = new_value
            else:
                break
        return s_odd_value
    test_cases = []
    test_cases.append([0])
    test_cases.append([0, 0])
    test_cases.append([0, 1])
    test_cases.append([0, -1])
    test_cases.append([0, -1, -1])

    for _ in range(100):
        l = np.random.randint(1, 22)
        test_cases.append(np.random.randint(0, 5, l))
        test_cases.append(np.random.randint(-5, 5, l))
        test_cases.append(np.random.randint(-5, 0, l))
        test_cases.append(-1 * np.random.rand(l) + 0.5)

    for arr in test_cases:
        assert approx(s_odd(arr)) == sum_odd(arr)

def test_direct_inflence():
    import numpy as np
    def get_delta(model, ctx):
        """
        Calculates Delta
        NOTE: model is the array of probability distribution
              ctx is the labels of contexts. 
        For example, for ctx=[("a", "b"), ("a'", "b"), ("a", "b'"), ("a'", "b'")]
        the model is a (4,4) array s.t.:
               |(0,0)|(0,1)|(1,0)|(1,1)|
        (a,b)  | p11 | p12 | p13 | p14 |
        (a,b') | p21 | p22 | p23 | p24 |
        (a',b) | p31 | p32 | p33 | p34 |
        (a',b')| p41 | p42 | p43 | p44 |
        will be input as :
        [
            [p11, p12, p13, p14],
            [p21, p22, p23, p24],
            [p31, p32, p33, p34],
            [p41, p42, p43, p44]
        ]
        """
        new_dists = np.asarray([np.asarray(x).reshape((2,2)) for x in model])
        contents = set([c[0] for c in ctx] + [c[1] for c in ctx])
        delta = 0
        for cN in contents:
            assert len(np.argwhere(np.asarray(ctx).flatten()==cN).flatten())==2
            contexts = [int(x) for x in np.argwhere(np.asarray(ctx).flatten()==cN).flatten()/2]
            axes = [np.mod(np.argwhere(np.asarray(ctx[c])==cN).flatten()[0]+1, 2) for c in contexts]
            marginals = [np.sum(new_dists[contexts[ii]], axis=axes[ii]) for ii in range(2)]
            delta += np.absolute(np.sum(np.array([-1,1])*(marginals[0] - marginals[1])))
        return delta
    scenario = CyclicScenario(['x1', 'x2', 'x3'], 2)
    for _ in range(20):
        model = random_model(scenario)
        assert model.CbD_direct_influence() == approx(get_delta(model._distributions, scenario.contexts))

