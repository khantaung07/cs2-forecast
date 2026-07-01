import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    accuracy: float
    log_loss: float
    brier_score: float


def clamp_probability(p: float, eps: float = 1e-15) -> float:
    return min(max(p, eps), 1.0 - eps)


def compute_binary_metrics(y_true: list[int], y_prob: list[float]) -> BinaryMetrics:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    if not y_true:
        return BinaryMetrics(
            n=0,
            accuracy=0.0,
            log_loss=0.0,
            brier_score=0.0,
        )

    correct = 0
    total_log_loss = 0.0
    total_brier = 0.0

    for actual, prob in zip(y_true, y_prob, strict=True):
        p = clamp_probability(prob)

        predicted = 1 if p >= 0.5 else 0
        if predicted == actual:
            correct += 1

        total_log_loss += -(actual * math.log(p) + (1 - actual) * math.log(1 - p))
        total_brier += (p - actual) ** 2

    n = len(y_true)

    return BinaryMetrics(
        n=n,
        accuracy=correct / n,
        log_loss=total_log_loss / n,
        brier_score=total_brier / n,
    )
