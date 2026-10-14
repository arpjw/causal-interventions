"""Simulator invariants checked against independent rational enumeration."""

from collections import Counter
from fractions import Fraction
from itertools import permutations, product
from math import fsum
from random import Random
import unittest

from causal_interventions.types import Intervention, Query
from causal_interventions.worlds import FiniteSCM, demo_worlds


VARIABLES = ("X", "Y", "Z")
STATES = tuple(product((0, 1), repeat=3))
JOINT = (0.28, 0.12, 0.03, 0.07, 0.07, 0.03, 0.12, 0.28)


def rational_do(weights, order, action):
    """Independent conditional-ratio product using exact rational marginals."""
    table = {state: Fraction(weight, sum(weights)) for state, weight in zip(STATES, weights)}
    result = []
    for state in STATES:
        probability = Fraction(int(state[VARIABLES.index(action.target)] == action.value))
        prefix = {}
        for name in order:
            position = VARIABLES.index(name)
            denominator = sum(p for s, p in table.items() if all(s[i] == v for i, v in prefix.items()))
            prefix[position] = state[position]
            numerator = sum(p for s, p in table.items() if all(s[i] == v for i, v in prefix.items()))
            if name != action.target:
                probability *= numerator / denominator
        result.append(float(probability))
    return result


class FiniteSCMTests(unittest.TestCase):
    def test_all_orders_reproduce_joint_and_zero_observational_information(self):
        for order in permutations(VARIABLES):
            world = FiniteSCM(VARIABLES, order, JOINT)
            for state, p in zip(STATES, JOINT):
                self.assertAlmostEqual(world.observational_probability(state), p, places=14)
                self.assertAlmostEqual(world.observational_probability(state) / p, 1.0, places=14)

    def test_three_variable_hand_calculation(self):
        forward = FiniteSCM(VARIABLES, VARIABLES, JOINT)
        reverse = FiniteSCM(VARIABLES, ("Y", "X", "Z"), JOINT)
        self.assertAlmostEqual(forward.probability(Query(Intervention("X", 1), "Z")), .62)
        self.assertAlmostEqual(reverse.probability(Query(Intervention("X", 1), "Z")), .5)
        self.assertAlmostEqual(forward.probability(Query(Intervention("X", 1), "Y")), .8)
        self.assertAlmostEqual(reverse.probability(Query(Intervention("X", 1), "Y")), .5)

    def test_random_tables_match_independent_rational_oracle(self):
        rng = Random(841)
        for _ in range(12):
            weights = tuple(rng.randrange(1, 100) for _ in STATES)
            joint = tuple(w / sum(weights) for w in weights)
            for order in permutations(VARIABLES):
                world = FiniteSCM(VARIABLES, order, joint)
                for target, value in product(VARIABLES, (0, 1)):
                    action = Intervention(target, value)
                    distribution = world.intervention_distribution(action)
                    self.assertAlmostEqual(fsum(distribution), 1.0, places=14)
                    for state, actual, expected in zip(STATES, distribution, rational_do(weights, order, action)):
                        self.assertAlmostEqual(actual, expected, places=14)
                        if state[VARIABLES.index(target)] != value:
                            self.assertEqual(actual, 0.0)

    def test_independent_variables_are_negative_control(self):
        joint = tuple((.3 if x else .7) * (.6 if y else .4) * (.8 if z else .2)
                      for x, y, z in STATES)
        worlds = [FiniteSCM(VARIABLES, order, joint) for order in permutations(VARIABLES)]
        for target, value in product(VARIABLES, (0, 1)):
            action = Intervention(target, value)
            reference = worlds[0].intervention_distribution(action)
            for world in worlds:
                for actual, expected in zip(world.intervention_distribution(action), reference):
                    self.assertAlmostEqual(actual, expected, places=14)

    def test_two_variable_compatibility_and_target_query(self):
        for old in demo_worlds():
            order = ("X", "Y") if old.cause == "X" else ("Y", "X")
            world = FiniteSCM(("X", "Y"), order, old.joint)
            for target, value in product(("X", "Y"), (0, 1)):
                action = Intervention(target, value)
                self.assertAlmostEqual(world.probability(Query(action, "Y")),
                                       old.intervention_probability(target, value))
                self.assertAlmostEqual(world.probability(Query(action, target, value)), 1.0)
                self.assertEqual(world.probability(Query(action, target, 1 - value)), 0.0)

    def test_seeded_sampling_matches_exact_probabilities(self):
        world = FiniteSCM(VARIABLES, VARIABLES, JOINT)
        action = Intervention("X", 1)
        rng = Random(731)
        counts = Counter(world.sample_intervention(action, rng) for _ in range(12000))
        for state, probability in zip(STATES, world.intervention_distribution(action)):
            self.assertLess(abs(counts[state] / 12000 - probability), .015)
        self.assertEqual(world.sample_intervention(action, Random(9)),
                         world.sample_intervention(action, Random(9)))

    def test_validation_rejects_bad_worlds_and_requests(self):
        for joint in ((.125,) * 7, (.1,) * 8, (0.,) + (.125,) * 7,
                      (float("nan"),) + (.125,) * 7, (float("inf"),) + (.125,) * 7,
                      (-.1,) + (.125,) * 7):
            with self.assertRaises(ValueError):
                FiniteSCM(VARIABLES, VARIABLES, joint)
        for variables, order in ((("X", "X", "Z"), VARIABLES), (VARIABLES, ("X", "X", "Z")),
                                 (VARIABLES, ("X", "Y")), (("", "Y", "Z"), VARIABLES), ((), ())):
            with self.assertRaises(ValueError):
                FiniteSCM(variables, order, JOINT)
        world = FiniteSCM(VARIABLES, VARIABLES, JOINT)
        for state in ((0, 1), (0, 1, 2), (0, 1, float("nan")), (0.0, 1, 0)):
            with self.assertRaises(ValueError):
                world.observational_probability(state)
        with self.assertRaises(ValueError):
            world.intervention_distribution(Intervention("unknown", 1))
        with self.assertRaises(ValueError):
            world.probability(Query(Intervention("X", 0), "unknown"))


if __name__ == "__main__":
    unittest.main()
