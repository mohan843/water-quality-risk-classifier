# Water Quality Risk Classifier

An environmental machine-learning mini-project that classifies water samples as low, medium, or high contamination risk from pH, turbidity, nitrate concentration, and temperature.

## Overview

Water monitoring teams need quick, interpretable screening signals before laboratory confirmation. This project uses nearest-centroid classification to summarize labeled sample profiles and assign a new sample to the closest risk category.

## Problem statement

Given four water-quality measurements, predict a broad contamination-risk category for educational screening.

## Features

- Nearest-centroid classification from scratch.
- Multivariate Euclidean distance.
- CSV sample data and command-line inference.
- Unit tests and training accuracy reporting.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- Nearest-centroid classification
- Environmental data features and Euclidean distance
- CSV, `argparse`, and `pathlib`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/water-quality-risk-classifier.git
cd water-quality-risk-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python classifier.py --ph 7.2 --turbidity-ntu 1.2 --nitrate-mg-l 2.0 --temperature-c 18
```

High-risk example:

```bash
python classifier.py --ph 6.0 --turbidity-ntu 12 --nitrate-mg-l 16.5 --temperature-c 28
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 100.0%
Predicted contamination risk: low
```

## Responsible use and limitations

The dataset is synthetic and the categories are educational only. This model must not replace certified laboratory testing, regulatory thresholds, or public-health guidance. Real systems require calibrated sensors, local standards, uncertainty estimates, and expert review.

## Future improvements

- Add region-specific regulatory thresholds.
- Compare nearest neighbors, trees, and probabilistic models.
- Add missing-value handling and sensor anomaly detection.
- Validate against laboratory measurements with time-based splits.
- Build a monitoring dashboard with alerts and audit trails.
