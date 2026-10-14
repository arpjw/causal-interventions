import unittest

from causal_interventions.types import Intervention, Query


class IdentifierTests(unittest.TestCase):
    def test_binary_values_require_integers(self) -> None:
        for value in (True, False, 0.0, 1.0, -1, 2):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Intervention("X", value)
                with self.assertRaises(ValueError):
                    Query(Intervention("X", 0), "Y", value)

    def test_valid_identifiers(self) -> None:
        query = Query(Intervention("X", 1), "Y")
        self.assertEqual(query.outcome_value, 1)


if __name__ == "__main__":
    unittest.main()
