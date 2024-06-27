from contextuality import Model

def print_model_cbd(model: Model):
    from prettytable import PrettyTable
    from contextuality import CyclicScenario

    outcomes = model.scenario.outcomes
    assert isinstance(model.scenario, CyclicScenario)
    for i, context in enumerate(model.scenario.contexts):
        pt = PrettyTable()
        pt.field_names = [context] + outcomes
        for o1, outcome1 in enumerate(outcomes):
            row = []
            row.append(outcome1)
            for o2, outcome2 in enumerate(outcomes):
                row.append(f"{model.probability(context, (o1, o2)):.4f}")
            pt.add_row(row)
        print(pt)
                
def symmetrized_model(model: Model) -> Model:
    from numpy import flip
    if len(model.scenario.outcomes) != 2:
        raise ValueError("Only model with binary outcomes can be symmetrised.")
    sym_model = Model(model.scenario, flip(model._distributions, 1))
    return model.mix(sym_model, 0.5)