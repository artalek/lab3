import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _scaled_features(train_features, test_features):
    """Fit the scaling on the training rows and reuse it for the test rows."""
    scaler = StandardScaler()
    train = scaler.fit_transform(np.asarray(train_features, dtype=float))
    test = scaler.transform(np.asarray(test_features, dtype=float))
    return train, test, scaler


def _original_scale(weights, bias, scaler):
    weights = weights / scaler.scale_
    return weights, float(bias - scaler.mean_ @ weights)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Fit ordinary least squares by full-batch gradient descent."""
    train, test, scaler = _scaled_features(data.train_features, data.test_features)
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    weights = np.zeros(train.shape[1])
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        residual = train @ weights + bias - target
        weights -= (2 * data.learning_rate / len(train)) * (train.T @ residual)
        bias -= 2 * data.learning_rate * residual.mean()
        updated_residual = train @ weights + bias - target
        loss_history.append(float(np.mean(updated_residual ** 2)))

    predictions = test @ weights + bias
    weights, bias = _original_scale(weights, bias, scaler)
    return RegressionResult(weights, bias, loss_history, predictions)


def _sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Minimize binary cross-entropy by full-batch gradient descent."""
    train, test, scaler = _scaled_features(data.train_features, data.test_features)
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    weights = np.zeros(train.shape[1])
    bias = 0.0
    loss_history = []
    signs = 1 - 2 * target

    for _ in range(data.epochs):
        residual = _sigmoid(train @ weights + bias) - target
        weights -= (data.learning_rate / len(train)) * (train.T @ residual)
        bias -= data.learning_rate * residual.mean()
        logits = train @ weights + bias
        loss_history.append(float(np.logaddexp(0, signs * logits).mean()))

    probabilities = _sigmoid(test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    weights, bias = _original_scale(weights, bias, scaler)
    return ClassificationResult(weights, bias, loss_history, probabilities, predictions)


def classification_metrics(target, probabilities, predictions) -> ClassificationMetrics:
    target = np.asarray(target).reshape(-1)
    auc = roc_auc_score(target, probabilities) if np.unique(target).size == 2 else float("nan")
    return ClassificationMetrics(
        roc_auc=float(auc),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Evaluate three sklearn baselines on the split supplied by the caller."""
    train, test, _ = _scaled_features(data.train_features, data.test_features)
    target = np.asarray(data.train_target).reshape(-1)

    logistic = LogisticRegression(max_iter=1000)
    tree = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(train)))
    for model in (logistic, tree, knn):
        model.fit(train, target)

    return BaselineComparisonResult(
        logistic_regression=classification_metrics(
            data.test_target, logistic.predict_proba(test)[:, 1], logistic.predict(test)
        ),
        decision_tree=classification_metrics(
            data.test_target, tree.predict_proba(test)[:, 1], tree.predict(test)
        ),
        knn=classification_metrics(
            data.test_target, knn.predict_proba(test)[:, 1], knn.predict(test)
        ),
    )
