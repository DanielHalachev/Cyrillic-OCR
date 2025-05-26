from Levenshtein import distance  # Install python-Levenshtein


def char_error_rate(truth, pred):
    # Compute the Levenshtein distance between the truth and prediction
    dist = distance(truth, pred)
    # Compute the length of the truth string
    length = len(truth)
    # Compute the CER
    cer = dist / length
    return cer


def word_error_rate(truth: str, prediction: str) -> float:
    truth_words = truth.strip().split()
    prediction_words = prediction.strip().split()

    if not truth_words and not prediction_words:
        return 0.0
    if not truth_words:
        return len(prediction_words)
    if not prediction_words:
        return len(truth_words)

    dist = distance(" ".join(truth_words), " ".join(prediction_words))
    return dist / len(truth_words)
