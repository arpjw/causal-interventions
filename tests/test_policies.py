"""Independent policy-tree enumeration and hand-worked decision core checks."""
import itertools
import math
import unittest
from dataclasses import dataclass

from causal_interventions.beliefs import BeliefModel, normalize
from causal_interventions.policies import (ExactPolicy, GreedyInformationGainPolicy,
    GreedyQueryRiskPolicy, RandomPolicy, expected_policy_risk)
from causal_interventions.scoring import bernoulli_kl, excess_log_loss, scored_forecasts
from causal_interventions.types import Intervention, Query

A = Intervention("X", 0)
B = Intervention("Y", 0)
Q = Query(Intervention("X", 1), "Z")


@dataclass(frozen=True)
class TableWorld:
    a: tuple[float, ...]
    b: tuple[float, ...]
    query: float

    def intervention_distribution(self, action):
        return self.a if action == A else self.b

    def probability(self, query):
        return self.query


def trees(actions, depth):
    """Enumerate every deterministic binary-outcome policy tree, including STOP."""
    yield None
    if depth:
        children = list(trees(actions, depth - 1))
        for a in actions:
            for left, right in itertools.product(children, repeat=2):
                yield (a, left, right)


def tree_risk(model, tree):
    """Evaluate leaves from joint path likelihoods without using belief methods."""
    def visit(node, masses):
        mass = sum(masses)
        if mass == 0:
            return 0.0
        if node is None:
            prediction = sum(m * w.query for m, w in zip(masses, model.worlds)) / mass
            return sum(m * bernoulli_kl(w.query, prediction)
                       for m, w in zip(masses, model.worlds) if m)
        a, left, right = node
        return sum(visit(child, tuple(m * w.intervention_distribution(a)[o]
                                     for m, w in zip(masses, model.worlds)))
                   for o, child in enumerate((left, right)))
    return visit(tree, model.prior)


class DecisionCoreTests(unittest.TestCase):
    def setUp(self):
        self.model = BeliefModel((TableWorld((0.8, 0.2), (0.5, 0.5), 0.9),
                                 TableWorld((0.2, 0.8), (0.5, 0.5), 0.1)), (B, A), (Q,))

    def test_hand_posterior_and_predictions(self):
        m = self.model
        self.assertEqual(m.prior, (0.5, 0.5))
        self.assertEqual(m.posterior(m.prior, A, 0), (0.8, 0.2))
        self.assertAlmostEqual(m.predictive((0.8, 0.2))[0], 0.74)
        self.assertAlmostEqual(m.risk(m.prior), 0.9 * math.log(1.8) + 0.1 * math.log(0.2))
        for a in m.actions:
            self.assertAlmostEqual(sum(m.outcomes(m.prior, a)), 1)
            for o in (0, 1):
                self.assertAlmostEqual(sum(m.posterior(m.prior, a, o)), 1)

    def test_impossible_outcomes_and_invalid_inputs(self):
        m = BeliefModel((TableWorld((1, 0), (1, 0), 0.5),), (A,), (Q,))
        with self.assertRaises(ValueError):
            m.posterior(m.prior, A, 1)
        for weights in ((), (0, 0), (-1, 2), (math.nan, 1), (math.inf, 1)):
            with self.assertRaises(ValueError):
                normalize(weights)
        for b in ((0.1,), (-0.2, 1.2), (math.nan, 0), (0.2, 0.2)):
            with self.assertRaises(ValueError):
                self.model.risk(b)
        with self.assertRaises(ValueError):
            m.posterior(m.prior, A, -1)
        with self.assertRaises(ValueError):
            ExactPolicy(m).value(m.prior, -1)

    def test_reject_mismatched_outcome_variable_order(self):
        from causal_interventions.worlds import FiniteSCM
        joint = (0.125,) * 8
        first = FiniteSCM(("X", "Y", "Z"), ("X", "Y", "Z"), joint)
        second = FiniteSCM(("Z", "Y", "X"), ("X", "Y", "Z"), joint)
        with self.assertRaisesRegex(ValueError, "canonical variable order"):
            BeliefModel((first, second), (A,), (Q,))

    def test_greedy_hand_scores(self):
        m = self.model
        ig, qr = GreedyInformationGainPolicy(m), GreedyQueryRiskPolicy(m)
        expected_information = math.log(2) + 0.8 * math.log(0.8) + 0.2 * math.log(0.2)
        self.assertAlmostEqual(ig.scores(m.prior)[A], expected_information)
        self.assertAlmostEqual(ig.scores(m.prior)[B], 0)
        self.assertEqual(ig.choose(m.prior, 1), A)
        self.assertEqual(qr.choose(m.prior, 1), A)
        self.assertAlmostEqual(qr.scores(m.prior)[A], m.risk(m.prior) - tree_risk(m, (A, None, None)))

    def test_exhaustive_trees_budgets_one_two(self):
        models = [self.model, BeliefModel((
            TableWorld((0.9, 0.1), (0.4, 0.6), 0.95),
            TableWorld((0.6, 0.4), (0.9, 0.1), 0.5),
            TableWorld((0.1, 0.9), (0.3, 0.7), 0.05)), (A, B), (Q,), (0.2, 0.3, 0.5))]
        for m in models:
            oracle = ExactPolicy(m)
            previous = m.risk(m.prior)
            for budget in (1, 2):
                evaluated = [(tree_risk(m, t), t) for t in trees(m.actions, budget)]
                value, _ = min(evaluated, key=lambda pair: pair[0])
                optimal_roots = {None if t is None else t[0]
                                 for v, t in evaluated if abs(v - value) < 1e-12}
                self.assertAlmostEqual(oracle.value(m.prior, budget), value, places=13)
                self.assertIn(oracle.choose(m.prior, budget), optimal_roots)
                self.assertLessEqual(value, previous + 1e-14)
                previous = value
                for policy in (RandomPolicy(m), GreedyInformationGainPolicy(m), GreedyQueryRiskPolicy(m)):
                    self.assertLessEqual(value, expected_policy_risk(m, policy, m.prior, budget) + 1e-14)

    def test_structural_information_can_be_query_irrelevant(self):
        m = BeliefModel((TableWorld((1, 0), (0, 1), 0.6),
                        TableWorld((0, 1), (1, 0), 0.6)), (B, A), (Q,))
        self.assertEqual(GreedyInformationGainPolicy(m).choose(m.prior, 2), A)
        for policy in (ExactPolicy(m), GreedyQueryRiskPolicy(m)):
            self.assertIsNone(policy.choose(m.prior, 2))
        for policy in (ExactPolicy(m), GreedyQueryRiskPolicy(m), GreedyInformationGainPolicy(m), RandomPolicy(m)):
            self.assertIsNone(policy.choose(m.prior, 0))

    def test_information_gain_and_query_value_choose_different_actions(self):
        # A identifies an irrelevant latent bit exactly; B measures the query bit.
        m = BeliefModel(tuple(TableWorld((1 - nuisance, nuisance),
                                        (0.8, 0.2) if relevant == 0 else (0.2, 0.8),
                                        0.1 if relevant == 0 else 0.9)
                             for nuisance in (0, 1) for relevant in (0, 1)),
                        (A, B), (Q,))
        self.assertEqual(GreedyInformationGainPolicy(m).choose(m.prior, 1), A)
        self.assertEqual(GreedyQueryRiskPolicy(m).choose(m.prior, 1), B)
        self.assertEqual(ExactPolicy(m).choose(m.prior, 1), B)

    def test_random_reproducible_and_analytic(self):
        m = self.model
        a, b = RandomPolicy(m, 37), RandomPolicy(m, 37)
        self.assertEqual([a.choose(m.prior, 1) for _ in range(20)],
                         [b.choose(m.prior, 1) for _ in range(20)])
        expected = sum(tree_risk(m, (a, None, None)) for a in m.actions) / len(m.actions)
        self.assertAlmostEqual(expected_policy_risk(m, a, m.prior, 1), expected)

    def test_scoring_endpoints_and_clipping(self):
        self.assertEqual(bernoulli_kl(0, 0), 0)
        self.assertEqual(bernoulli_kl(1, 1), 0)
        self.assertTrue(math.isinf(bernoulli_kl(0.5, 0)))
        score = scored_forecasts((0, 1, 0.5), (0, 1, 0.5))
        self.assertEqual(score.clipping_count, 2)
        self.assertGreater(score.loss, 0)
        with self.assertRaises(ValueError):
            excess_log_loss((0.5,), (math.nan,))
        with self.assertRaises(ValueError):
            excess_log_loss((), ())


if __name__ == "__main__":
    unittest.main()
