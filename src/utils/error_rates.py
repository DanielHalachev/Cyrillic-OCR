from Levenshtein import distance  # Install python-Levenshtein


def char_error_rate(truth: str, pred: str) -> float:
    if not truth and not pred:
        return 0.0
    if not truth:
        return 1.0
    dist = distance(truth, pred)
    return dist / len(truth)


import re
from Levenshtein import distance


def word_error_rate(truth: str, prediction: str) -> float:
    truth_words = re.split(r"\s+", truth.strip()) if truth.strip() else []
    prediction_words = (
        re.split(r"\s+", prediction.strip()) if prediction.strip() else []
    )

    if not truth_words and not prediction_words:
        return 0.0
    if not truth_words:
        return 1.0
    if not prediction_words:
        return 1.0

    word_distance = distance(" ".join(truth_words), " ".join(prediction_words))
    return word_distance / len(truth_words)
