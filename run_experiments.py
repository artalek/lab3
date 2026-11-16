import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

ROOT = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2


def main():
    students = pd.read_csv(ROOT / "Student_Performance.csv")
    student_target = students.pop("Performance Index")
    students["Extracurricular Activities"] = students[
        "Extracurricular Activities"
    ].map({"No": 0, "Yes": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        students, student_target, test_size=TEST_SIZE, random_state=SEED
    )
    regression = train_linear_regression(RegressionInput(x_train, y_train, x_test, y_test))
    mse = float(np.mean((regression.predictions - y_test.to_numpy()) ** 2))

    german = pd.read_csv(ROOT / "german.csv", sep=";")
    credit_target = german.pop("Creditability")
    x_train, x_test, y_train, y_test = train_test_split(
        german, credit_target, test_size=TEST_SIZE, random_state=SEED, stratify=credit_target
    )
    credit_split = ClassificationInput(x_train, y_train, x_test, y_test)
    own = train_logistic_classifier(credit_split)
    own_metrics = classification_metrics(y_test, own.probabilities, own.predictions)
    baselines = compare_classical_models(credit_split)

    results = {
        "linear_regression": {"test_mse": mse},
        "classification": {"own_logistic": asdict(own_metrics), **asdict(baselines)},
    }
    results_json = json.dumps(results, indent=2, allow_nan=False)
    (ROOT / "experiment_results.json").write_text(results_json + "\n", encoding="utf-8")

    for name, history in (("Train MSE", regression.loss_history), ("Train BCE", own.loss_history)):
        checkpoints = (1, 10, 100, len(history))
        print(name, {epoch: history[epoch - 1] for epoch in checkpoints})
    print(results_json)


if __name__ == "__main__":
    main()
