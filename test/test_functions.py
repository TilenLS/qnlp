import pytest

def test_print_model_cbd():
    from contextuality import chsh_scenario, random_model, print_model_cbd
    scenario = chsh_scenario()
    for _ in range(10):
        model = random_model(scenario)
        print_model_cbd(model)

def test_symmetrized_model():
    from contextuality import symmetrized_model
    from contextuality import pr_model, random_pr_like_model, random_model, chsh_scenario
    from itertools import product

    for n in range(3, 10):
        assert symmetrized_model(pr_model(n)) == pr_model(n)
        assert symmetrized_model(random_pr_like_model(n)) == pr_model(n)

    for _ in range(10):
        scenario = chsh_scenario()
        model = symmetrized_model(random_model(scenario))
        for context in scenario.contexts:
            for outcome in product((0, 1), repeat=2):
                sym_outcome = tuple(1-out for out in outcome)
                prob = model.probability(context, outcome)
                sym_prob = model.probability(context, sym_outcome)
                assert prob == pytest.approx(sym_prob)
                

        
    