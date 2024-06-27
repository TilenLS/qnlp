from __future__ import annotations
from typing import Union, List, Tuple, Optional
import numpy
import itertools
import picos
from contextuality.utils import sum_odd

class Scenario:
    """
    Defines a measurement scenario.

    Parameters
    ----------
    contexts : list
        A list of context where each context is a tuple of observables.
        An observable is just a string which represent its name.
    num_outcome : int
        The number of possible outcome for every observables.
    """

    def __init__(self, contexts: List[Tuple[str, ...]], outcomes: Union[list, int]):
        self.contexts = contexts
        if isinstance(outcomes, int):
            self.num_outcome = outcomes
            self.outcomes = list(range(outcomes))
        else:
            self.num_outcome = len(outcomes)
            self.outcomes = outcomes

        # Caches
        self._observables = None
        self._incidence_matrix = None

    @property
    def observables(self) -> list:
        """Returns a list of observables."""
        if self._observables:
            return self._observables

        out = []
        for context in self.contexts:
            for observable in context:
                if observable not in out:
                    out.append(observable)
        return out

    @property
    def incidence_matrix(self) -> numpy.ndarray:
        """Compute the incidence matrix which maps a global distribution
        to the empirical model via matrix multiplication.
        """
        if self._incidence_matrix:
            return self._incidence_matrix
        global_assigns = list(
            itertools.product(range(self.num_outcome), repeat=len(self.observables))
        )

        M_context = []
        for context in self.contexts:
            local_assigns = list(
                itertools.product(range(self.num_outcome), repeat=len(context))
            )

            ob_idx = [self.observables.index(ob) for ob in context]
            g_context = [tuple([g[idx] for idx in ob_idx]) for g in global_assigns]

            M_context_this = numpy.zeros(
                (len(local_assigns), len(global_assigns)), dtype=int
            )
            for i in range(len(local_assigns)):
                for j in range(len(g_context)):
                    M_context_this[i, j] = local_assigns[i] == g_context[j]

            M_context.append(M_context_this)

        M = numpy.vstack(M_context)
        return M

    def __eq__(self, other):
        return self.contexts == other.contexts and self.num_outcome == other.num_outcome

    def __repr__(self):
        class_name = self.__class__.__name__
        contexts_repr = self.contexts.__repr__()
        return f"{class_name}({contexts_repr}, {self.num_outcome})"

    def __str__(self):
        observable_str = ", ".join([ob for ob in self.observables])
        context_str = ", ".join([str(c).replace("'", "") for c in self.contexts])
        outcome_str = ", ".join(map(str, range(self.num_outcome)))
        out = ""
        out += f"Observables\t[{observable_str}]\n"
        out += f"Contexts\t[{context_str}]\n"
        out += f"Outcomes\t[{outcome_str}]"
        return out


class CyclicScenario(Scenario):
    """Defines a cyclic scenario which is a special kind of
    measurement scenario.
    Each observable appears in exactly two contexts and each
    context contains exactly two observables.

    Cyclic scenarios allows quicker computations compared to
    a general scenario. Therefore, it is recommended to use
    CyclicScenario whenever possible.

    Parameters
    ----------
    observables : list of strings
        A list of observables. Every pair of observables appear
        next to each other is considered as a context. The last
        and the first observable together also form a context.
    num_outcome : int
        The number of possible outcome for every observables.
    """

    def __init__(self, observables: List[str], outcomes: Union[list, int]):
        contexts = list(zip(observables, observables[1:] + observables[:1]))
        super().__init__(contexts, outcomes)

    def __repr__(self):
        class_name = self.__class__.__name__
        observables_repr = self.observables.__repr__()
        return f"{class_name}({observables_repr}, {self.num_outcome})"


class Model:
    """Defines a empirical model which specifies a probability
    distribution for each context in a given measurement scenario.

    Parameters
    ----------
    scenario : Scenario
        A measurement scenario.
    distributions : iterable of iterables
        A list of distributions for each context. The distributions
        should have the same ordering as the contexts in the
        scenario. A distribution for N observables should be a list of
        probabilities of length num_outcome**N.
    """

    def __init__(self, scenario: Scenario, distributions):
        if len(distributions) != len(scenario.contexts):
            raise ValueError(
                "Number of distributions must be equal to the number of contexts"
            )

        self.scenario = scenario
        self._distributions = []

        if isinstance(scenario, CyclicScenario):
            self._distributions = numpy.array(distributions)
            num_outcome = scenario.num_outcome
            rank = len(scenario.contexts)
            expected_shape = (rank, num_outcome**2)
            if self._distributions.shape != expected_shape:
                raise ValueError(
                    "The distributions provided are not compatible with the scenario."
                )
        else:
            for context, distribution in zip(scenario.contexts, distributions):
                self._distributions.append(numpy.array(distribution))

                if len(distribution) != scenario.num_outcome ** len(context):
                    raise ValueError(
                        f"The distribution provided for context {context} has the incorrect number of probabilities"
                    )

    def probability(self, context: Tuple[str, ...], outcome: tuple) -> float:
        if context not in self.scenario.contexts:
            raise ValueError(f"The given context {context} does not exists.")
        context_idx = self.scenario.contexts.index(context)
        outcome_idx = sum(
            out * self.scenario.num_outcome**i for i, out in enumerate(outcome)
        )
        return self._distributions[context_idx][outcome_idx]

    @property
    def vector(self) -> numpy.ndarray:
        """Returns the distributions in a flattened vector.
        The distribution of the first context goes first in
        the vector. The distribution of the second context goes
        second in the vector, etc.
        """
        return numpy.array([p for dist in self._distributions for p in dist])

    def _decomposition(self, mode: str) -> float:
        """_decomposition.

        Parameters
        ----------
        mode : str
            A string specifying the type of decomposition wanted.
            The options are:
                'cf': contextual fraction
                'sf': signalling_fraction

        Returns
        -------
        float
            The maximum fraction of the restricted part of the
            decomposition, e.g. the non-contextual part.
        """
        M = self.scenario.incidence_matrix
        v = self.vector
        num_global_assign = M.shape[1]
        one = numpy.ones(num_global_assign)

        problem = picos.Problem()
        problem.options.solver = "mosek"

        b = picos.RealVariable("b", num_global_assign)

        problem.set_objective("max", one | b)

        problem.add_constraint(M * b <= v)

        if mode == "cf":
            problem.add_constraint(b >= 0)
        if mode == "sf":
            problem.add_constraint(M * b >= 0)

        solution = problem.solve()
        weight_N = solution.value
        return float(weight_N)

    def contextual_fraction(self) -> float:
        """Compute the contextual fraction of the model."""
        weight_NC = self._decomposition("cf")
        weight_SC = 1 - weight_NC
        return weight_SC

    def signalling_fraction(self) -> float:
        """Compute the signalling fraction of the model."""
        weight_NS = self._decomposition("sf")
        weight_SS = 1 - weight_NS
        return weight_SS

    def CbD_direct_influence(self) -> float:
        # Currently only for cyclic scenario with binary outcome
        assert isinstance(self.scenario, CyclicScenario)
        assert self.scenario.num_outcome == 2

        dist = self._distributions

        parity_in = numpy.array([1, 1, -1, -1])
        parity_out = numpy.array([1, -1, 1, -1])
        ev_in = [numpy.dot(d, parity_in) for d in dist]
        ev_out = [numpy.dot(d, parity_out) for d in dist]

        ret = sum(numpy.abs(numpy.array(ev_in) + -numpy.roll(ev_out, 1)))
        return float(ret)

    def CbD_measure(self) -> float:
        """Compute the CbD measure for binary cyclic scenarios."""
        # Currently only for cyclic scenario with binary outcome
        assert isinstance(self.scenario, CyclicScenario)
        assert self.scenario.num_outcome == 2

        contexts = self.scenario.contexts
        dist = self._distributions

        parity = numpy.array([1, -1, -1, 1])
        corr = [numpy.dot(d, parity) for d in dist]

        ret = sum_odd(corr) - self.CbD_direct_influence() - (len(contexts) - 2)
        return float(ret)

    def mix(self, other: Model, weight: float) -> Model:
        """Return the convex sum of this model and another model.

        Parameters
        ----------
        other :
            other
        weight : float
            The weight given to the other model.

        Returns
        -------
        Model
            The convex sum.
        """
        if self.scenario != other.scenario:
            raise ValueError("Cannot mix two models with non-identical scenarios")

        self_dist = self._distributions
        other_dist = other._distributions

        mix_dist = []
        for i in range(len(self_dist)):
            mix_dist.append((1 - weight) * self_dist[i] + weight * other_dist[i])

        mix_model = Model(self.scenario, mix_dist)
        return mix_model

    def __distributions_binary_operation(self, other, operation):
        """A helper function to perform binary operation on the
        distributions of two models with the same measurement scenario.

        Parameters
        ----------
        other : Model
            The other model.
        operation : callable
            A function or lambda with two arguments.
        """
        if self.scenario is not other.scenario and self.scenario != other.scenario:
            raise ValueError("Incompatible scenarios.")
        is_self_cyclic = isinstance(self.scenario, CyclicScenario)
        is_other_cyclic = isinstance(other.scenario, CyclicScenario)
        if is_self_cyclic or is_other_cyclic:
            dist = operation(self._distributions, other._distributions)
            if is_self_cyclic:
                return Model(self.scenario, dist)
            else:
                return Model(other.scenario, dist)
        else:
            dist = []
            self_dist = self._distributions
            other_dist = other._distributions
            for i in range(len(self.scenario.contexts)):
                dist.append(operation(self_dist[i], self_dist[i]))
            return Model(self.scenario, dist)

    def summary(self) -> str:
        output = ""
        output += f"CF: {self.contextual_fraction():.6f}\n"
        output += f"SF: {self.signalling_fraction():.6f}\n"
        output += f"Δ: {self.CbD_direct_influence():.6f}\n"
        output += f"CbD: {self.CbD_measure():.6f}"
        return output

    def __add__(self, other):
        return self.__distributions_binary_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.__distributions_binary_operation(other, lambda x, y: x - y)

    def __mul__(self, scaler):
        if isinstance(self.scenario, CyclicScenario):
            dist = self._distributions * scaler
        else:
            dist = [d * scaler for d in self._distributions]
        return Model(self.scenario, self._distributions * scaler)

    def __rmul__(self, scaler):
        return self.__mul__(scaler)

    def __str__(self):
        num_outcome = self.scenario.num_outcome
        contexts = self.scenario.contexts
        out = ""
        if isinstance(self.scenario, CyclicScenario):
            context_strs = [str(c).replace("'", "") + " " for c in contexts]
            context_col_width = max(map(len, context_strs))
            assigns = map(
                str, itertools.product(self.scenario.outcomes, repeat=num_outcome)
            )
            assigns_str = " ".join(assigns)
            out += " " * context_col_width + assigns_str + "\n"
            for i, c in enumerate(contexts):
                dist_str = " ".join([f"{p:>6.4f}" for p in self._distributions[i]])
                out += context_strs[i] + dist_str + "\n"
        else:
            for i, c in enumerate(contexts):
                assigns = map(str, itertools.product(range(num_outcome), repeat=len(c)))
                assigns_str = " ".join(assigns)
                context_str = str(c).replace("'", "") + "->"
                dist_str = " ".join([f"{p:>6.4f}" for p in self._distributions[i]])

                out += context_str + assigns_str + "\n"
                out += " " * len(context_str) + dist_str + "\n"
        return out

    def __eq__(self, other):
        if self.scenario != other.scenario:
            return False
        if isinstance(self.scenario, CyclicScenario) or isinstance(
            other.scenario, CyclicScenario
        ):
            return numpy.allclose(self._distributions, other._distributions)

    def __repr__(self):
        class_name = self.__class__.__name__
        scenario_repr = self.scenario.__repr__()
        dist_repr = self._distributions.__repr__()
        return f"{class_name}({scenario_repr}, {dist_repr})"

    cf = contextual_fraction
    sf = signalling_fraction
    delta = CbD_direct_influence
    cbd = CbD_measure


def chsh_scenario() -> CyclicScenario:
    """Create a CHSH/Bell measurement scenario.
    Returns
    -------
    CyclicScenario
    """
    observables = ["a1", "b1", "a2", "b2"]
    return CyclicScenario(observables, 2)


def cyclic_scenario(n: int = 3, ob_name: str = "x"):
    """Create a cyclic scenario.

    Parameters
    ----------
    n : int
        The number of observables.
    ob_name: str
        The common name given to the observables.
    """
    observables = [f"{ob_name}{i+1}" for i in range(n)]
    return CyclicScenario(observables, 2)


def get_model_from_vector(scenario, vector) -> Model:
    """Return a empirical model given the distributions
    in a flattened vector.

    Parameters
    ----------
    scenario : Scenario
    vector :
        The vector containing all the distributions for
        each context in the scenario.

    Returns
    -------
    Model
        The empirical model constructed from the vector.
    """
    num_assign = [scenario.num_outcome ** len(c) for c in scenario.contexts]
    context_last_idx = numpy.add.accumulate(num_assign)
    dist = numpy.split(vector, context_last_idx[:-1])
    return Model(scenario, dist)


def pr_model(n: int = 4) -> Model:
    """Create a (general) PR box empirical model.
    If the number of observable is not specified, the
    traditional PR box with 4 observables is created,
    and the scenario used is the CHSH scenario.

    Parameters
    ----------
    n : int
        Number of observables. I.e. rank.

    Returns
    -------
    Model
    """
    if not n:
        table = [
            [1 / 2, 0, 0, 1 / 2],
            [1 / 2, 0, 0, 1 / 2],
            [1 / 2, 0, 0, 1 / 2],
            [0, 1 / 2, 1 / 2, 0],
        ]
        return Model(chsh_scenario(), table)
    else:
        table = numpy.zeros((n, 4))
        table[:-1, 0] = 1 / 2
        table[:-1, -1] = 1 / 2
        table[-1, 1] = 1 / 2
        table[-1, 2] = 1 / 2
        return Model(cyclic_scenario(n), table)


def random_pr_like_model(n: int = 3):
    """Generate a PR-like model.
    PR-like models are those with the same support
    as PR-like. In other words, whenever there is a
    zero probability in the PR box, the corresponding probability
    in a PR-like model is also zero.

    The probabilities are generated by first uniformly
    sample real numbers between 0 and 1, and then
    each distribution is normalised to one.

    Parameters
    ----------
    n : int
        The number of observables.
    """
    table = numpy.zeros((n, 4))
    table[:-1, 0] = numpy.random.rand(n - 1)
    table[:-1, -1] = numpy.random.rand(n - 1)
    table[-1, 1] = numpy.random.rand()
    table[-1, 2] = numpy.random.rand()
    table = table / numpy.sum(table, axis=1, keepdims=True)
    return Model(cyclic_scenario(n), table)


def bell_model() -> Model:
    """Create a Bell empirical model."""
    scenario = chsh_scenario()
    table = [
        [4 / 8, 0 / 8, 0 / 8, 4 / 8],
        [3 / 8, 1 / 8, 1 / 8, 3 / 8],
        [3 / 8, 1 / 8, 1 / 8, 3 / 8],
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    ]
    return Model(chsh_scenario(), table)


def random_model(scenario: Scenario, scaling=None):
    """Generate a random empirical model given a measurement
    scenario.

    The probabilities are generated by first uniformly
    sample real numbers between 0 and 1, and then
    each distribution is normalised to one.

    A scaling mask can be provided if you want to control
    the relative strength given to each probability.

    Parameters
    ----------
    scenario : Scenario
    scaling :
        A scaling masks which should have the same shape
        as the distributions table one supplies to the
        constructor of a Model object.
    """
    contexts = scenario.contexts
    num_outcome = scenario.num_outcome
    table = []
    for i in range(len(contexts)):
        context = contexts[i]
        rand_dist = numpy.random.rand(num_outcome ** len(context))
        if scaling:
            rand_dist = rand_dist * scaling[i]
        rand_dist = rand_dist / sum(rand_dist)
        table.append(rand_dist)
    return Model(scenario, table)


if __name__ == "__main__":
    s = CyclicScenario(["x1", "x2", "x3"], 2)
    print(s)
    dist1 = [[1 / 2, 0, 0, 1 / 2], [1 / 2, 0, 0, 1 / 2], [0, 1 / 2, 1 / 2, 0]]
    dist2 = [[1 / 2, 0, 0, 1 / 2], [1 / 2, 0, 0, 1 / 2], [1 / 2, 0, 0, 1 / 2]]
    m1 = Model(s, dist1)
    m2 = Model(s, dist2)
    # m = m1.mix(m2, 0.9)
    weight = 0.9
    m = weight * m1 + (1 - weight) * m2
    print(str(m))
    print(f"CF:\t{m.contextual_fraction():.5f}")
    print(f"SF:\t{m.signalling_fraction():.5f}")
    print(f"CbD:\t{m.CbD_measure():.5f}")
    print(m.__repr__())

    print(get_model_from_vector(s, m.vector))
