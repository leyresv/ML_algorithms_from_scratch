
def evaluate_accuracy(y_gold, y_pred):
    """
    Evaluate accuracy of the predictions

    :param Y_gold: list of actual labels
    :param Y_pred: list of predicted labels
    :return: accuracy value (int)
    """
    return sum([1 for gold, pred in zip(y_gold, y_pred) if gold == pred]) / len(y_pred)

