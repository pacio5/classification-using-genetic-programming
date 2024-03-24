import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def protectedDiv(left, right):
    """
    Protected division function. It returns 1 if the denominator is 0.
    :param left: The numerator.
    :param right: The denominator.
    :return: The division result.
    """
    if right == 0:
        return 1
    return left / right


def ephemeral():
    """
    Ephimeral constant generator. It returns a random float between 0 and 1.
    """
    return random.random()


def evaluate_model(model, X_test, y_test, toolbox):
    """
    Evaluate the model on the test set.
    :param model: The model to evaluate.
    :param X_test: The input data.
    :param y_test: The target values.
    :param toolbox: The DEAP toolbox.
    """
    # Compile the model into a usable Python function.
    func = toolbox.compile(expr=model)
    # Use the compiled function to make predictions about the test set
    predictions_test = [round(func(*record)) for record in X_test]

    # Calculates various performance metrics.
    accuracy = accuracy_score(y_test, predictions_test)
    precision = precision_score(y_test, predictions_test, average='macro')
    recall = recall_score(y_test, predictions_test, average='macro', zero_division=0)
    f1 = f1_score(y_test, predictions_test, average='macro')

    # Returns metrics as a dictionary
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
