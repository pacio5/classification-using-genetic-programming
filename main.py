import operator
import os
import random
import functools
import sys
import time
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, gp, algorithms
from utils import evaluate_model, protectedDiv, ephemeral
from visualize import visualize_tree


# Loading and preparation of the Iris dataset.
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evalIris(individual):
    """
    Fitness function that evaluates an individual's accuracy.
    :param individual: The individual to evaluate.
    :return: The accuracy of the individual.
    """
    func = toolbox.compile(expr=individual)
    predictions = np.array([round(func(*record)) for record in X_train], dtype=np.int64)
    return accuracy_score(y_train, predictions),


# Configuring the DEAP environment for genetic programming.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Configuring the toolbox with the necessary functions and parameters.
toolbox = base.Toolbox()
# Defining the primitive set.
pset = gp.PrimitiveSet("MAIN", arity=4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addEphemeralConstant("rand101", functools.partial(ephemeral))
# Defining the input arguments.
pset.renameArguments(ARG0='sepal_length', ARG1='sepal_width', ARG2='petal_length', ARG3='petal_width')

# Registering the necessary functions in the toolbox.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalIris)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Population initialization and evolutionary algorithm parameters.
pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
# Statistics to collect during evolution.
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Check if a path was passed as a parameter to save the checkpoint, otherwise use the output folder.
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
else:
    checkpoint_path = "output" + str(int(time.time()))

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Logbook initialization
logbook = tools.Logbook()
logbook.header = ['gen', 'evals'] + (stats.fields if stats else [])

# Evolutionary algorithm parameters
number_of_generations = 40
crossover_probability = 0.5
mutation_probability = 0.1

best_fitness = None

# Implement early stopping
not_improved = 0
patience = round(number_of_generations * 0.2)
gen = 0

metrics_train = []
metrics_test = []

# Evolution loop with early stopping
while gen < number_of_generations and not_improved < patience:
    # Evaluate the entire population
    offspring = algorithms.varAnd(pop, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)

    # Select individuals with invalid fitness and evaluate them.
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the population with newly selected individuals.
    pop[:] = toolbox.select(offspring, len(pop))
    # Update the Hall of Fame with top individuals.
    hof.update(pop)

    # Collect the statistics of the current generation and add them to the logbook.
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=gen, evals=len(invalid_ind), **record)

    # Save the best individual if it has improved
    if best_fitness is None or hof[0].fitness.values[0] > best_fitness:
        best_fitness = hof[0].fitness.values[0]
        cp = dict(population=pop, generation=gen, halloffame=hof, logbook=logbook, rndstate=random.getstate())
        # Save the checkpoint
        with open(f"{checkpoint_path}/best_individual.pkl", "wb") as file:
            pickle.dump(cp, file)

        # Save the best individual as a text file for human readability
        with open(f"{checkpoint_path}/best_individual.txt", "w") as file:
            file.write(str(hof[0]))
    else:
        not_improved += 1

    # Print the metrics of the current generation
    metrics_train.append(evaluate_model(hof[0], X_train, y_train, toolbox))
    print(f"Metriche training (gen:{gen}):", metrics_train[0])
    metrics_test.append(evaluate_model(hof[0], X_test, y_test, toolbox))
    print(f"Metriche test: (gen:{gen}):", metrics_test[0])
    gen += 1

# Accuracy evaluation on the test set of the best individual after the last generation.
evaluation_metrics = evaluate_model(hof[0], X_test, y_test, toolbox)
print("Metriche sul dataseset di test post evoluzione:", metrics_test)

# Save the logbook to a file
with open(f"{checkpoint_path}/evolution_log.txt", 'w') as logfile:
    for record in logbook:
        log_output = str(record)
        logfile.write(log_output + '\n')

    for i in range(len(metrics_train)):
        logfile.write(f"Metriche sul dataseset di training (gen:{i}): {metrics_train[i]}\n")
        logfile.write(f"Metriche sul dataseset di test (gen:{i}): {metrics_test[i]}\n")

    logfile.write(f"Metriche sul dataseset di test post evoluzione: {metrics_test}")

# Saving primitives
with open(f"{checkpoint_path}/primitive_set.pkl", "wb") as pset_file:
    pickle.dump(pset, pset_file)

# Save the primitives as a text file for human readability
with open(f"{checkpoint_path}/primitives.txt", "w") as file:
    for primitive in pset.primitives.values():
        for p in primitive:
            file.write(f"Name: {p.name}, Arity: {p.arity}\n")

# View the tree of the best individual
visualize_tree(hof[0], False, True, checkpoint_path)
