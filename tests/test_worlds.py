import unittest
from random import Random

from causal_interventions.worlds import BinaryWorld, demo_worlds


class WorldTests(unittest.TestCase):
    def test_observational_equivalence_and_intervention_gap(self) -> None:
        forward, reverse = demo_worlds()
        for x in (0, 1):
            for y in (0, 1):
                self.assertEqual(
                    forward.observational_probability(x, y),
                    reverse.observational_probability(x, y),
                )
        self.assertAlmostEqual(forward.intervention_probability("X", 1), 0.8)
        self.assertAlmostEqual(reverse.intervention_probability("X", 1), 0.5)

    def test_invalid_joint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BinaryWorld((0.4, 0.1, 0.1, 0.0), "X")

    def test_sampling_is_reproducible(self) -> None:
        world, _ = demo_worlds()
        samples_a = [world.sample_observation(Random(seed)) for seed in range(5)]
        samples_b = [world.sample_observation(Random(seed)) for seed in range(5)]
        self.assertEqual(samples_a, samples_b)


if __name__ == "__main__":
    unittest.main()
