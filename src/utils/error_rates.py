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
    """
    Calculate word error rate using word-level Levenshtein distance.

    Args:
        truth: Reference text
        prediction: Predicted text

    Returns:
        Word Error Rate as a float
    """
    truth_words = truth.strip().split()
    prediction_words = prediction.strip().split()

    if not truth_words and not prediction_words:
        return 0.0
    if not truth_words:
        return 1.0  # All words are insertions
    if not prediction_words:
        return 1.0  # All words are deletions

    # Calculate Levenshtein distance at the word level
    m, n = len(truth_words), len(prediction_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize borders
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if truth_words[i - 1] == prediction_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # The distance is the value in the bottom-right corner
    word_distance = dp[m][n]
    return word_distance / len(truth_words)
