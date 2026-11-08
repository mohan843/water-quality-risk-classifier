"""Classify water samples as low, medium, or high contamination risk."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

FEATURES = ("ph", "turbidity_ntu", "nitrate_mg_l", "temperature_c")


class NearestCentroidClassifier:
    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> "NearestCentroidClassifier":
        if not rows or len(rows) != len(labels):
            raise ValueError("Rows and labels must be non-empty and equally sized")
        self.centroids = {}
        for label in sorted(set(labels)):
            members = [row for row, target in zip(rows, labels) if target == label]
            self.centroids[label] = [sum(row[i] for row in members) / len(members) for i in range(len(rows[0]))]
        return self

    def predict_one(self, row: Sequence[float]) -> str:
        return min(self.centroids, key=lambda label: sum((a - b) ** 2 for a, b in zip(row, self.centroids[label])))

    def predict(self, rows: Sequence[Sequence[float]]) -> list[str]:
        return [self.predict_one(row) for row in rows]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [[float(row[name]) for name in FEATURES] for row in rows], [row["risk"] for row in rows]


def accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[NearestCentroidClassifier, float]:
    rows, labels = load_csv(path)
    model = NearestCentroidClassifier().fit(rows, labels)
    return model, accuracy(labels, model.predict(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict water-quality contamination risk.")
    parser.add_argument("--data", default="data/water_samples.csv")
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--turbidity-ntu", type=float, required=True)
    parser.add_argument("--nitrate-mg-l", type=float, required=True)
    parser.add_argument("--temperature-c", type=float, required=True)
    args = parser.parse_args()
    model, score = train(args.data)
    prediction = model.predict_one([args.ph, args.turbidity_ntu, args.nitrate_mg_l, args.temperature_c])
    print(f"Training accuracy: {score:.1%}")
    print(f"Predicted contamination risk: {prediction}")
