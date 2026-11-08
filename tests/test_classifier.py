import unittest
from pathlib import Path

from classifier import NearestCentroidClassifier, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "water_samples.csv"


class WaterQualityTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 18)
        self.assertEqual(set(labels), {"low", "medium", "high"})

    def test_classifier_learns_risk_profiles(self):
        model, score = train(DATA)
        self.assertGreaterEqual(score, 0.80)
        self.assertEqual(model.predict_one([7.2, 1.2, 2.0, 18]), "low")
        self.assertEqual(model.predict_one([6.0, 12.0, 16.5, 28]), "high")

    def test_prediction_count(self):
        rows, _ = load_csv(DATA)
        model, _ = train(DATA)
        self.assertEqual(len(model.predict(rows)), 18)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            NearestCentroidClassifier().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy(["low", "high"], ["low", "low"]), 0.5)


if __name__ == "__main__":
    unittest.main()
