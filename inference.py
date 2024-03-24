import sys
import pickle
import numpy as np
from deap import gp
from sklearn import datasets


def load_dataset():
    """Load the iris dataset.
    :return: Tuple containing the target names, the input data and the target values.
    """
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return iris.target_names, x, y


def load_configuration(configuration_path):
    """
    Load the configuration file containing the primitive set.
    :param configuration_path: Path to the configuration file.
    :return: The primitive set.
    """
    with open(f"{configuration_path}/primitive_set.pkl", "rb") as pset_file:
        pset = pickle.load(pset_file)
    return pset


def load_model(model_path):
    """
    Load the best individual (model).
    :param model_path: Path to the model.
    :return: The best individual.
    """
    with open(model_path + "/best_individual.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)
    return cp['halloffame'][0]  # Ritorna il miglior individuo


if __name__ == "__main__":
    """
    Execute the inference on the best individual.
    Usage: python inference.py <path_to_model> <optional_n_samples>
    """
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_model>")
        sys.exit()

    model_path = sys.argv[1]

    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])
    else:
        n_samples = 5

    # Load primitives
    pset = load_configuration(model_path)
    # Load the model and compile the expression tree.
    best_individual = load_model(model_path)
    func = gp.compile(expr=best_individual, pset=pset)

    # Load the dataset
    class_names, x, y = load_dataset()

    # Select n_samples random samples
    random_indices = np.random.choice(range(len(x)), size=n_samples, replace=False)

    # Do inference on random samples and print the actual class and prediction.
    for i in random_indices:
        sample = x[i]
        real_class = y[i]
        prediction = round(func(*sample))
        print(f"Sample {i}: Real class = {class_names[real_class]}, Prediction = {class_names[prediction]}")
