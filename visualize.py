import pickle
import sys
import matplotlib.pyplot as plt
import networkx as nx
from deap import gp
from networkx.drawing.nx_agraph import graphviz_layout
from utils import protectedDiv, ephemeral


def visualize_tree(individual, show=True, save=False, path=None):
    """
    Visualize the expression tree of an individual.
    :param individual: The individual to visualize.
    :param show: Whether to show the plot.
    :param save: Whether to save the plot in png.
    :param path: The path where to save the plot.
    """
    # Transforms the expression tree into a NetworkX graph.
    nodes, edges, labels = gp.graph(individual)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)

    if show:
        plt.show()

    if save and path is not None:
        plt.savefig(path + "/tree.png")


if __name__ == "__main__":
    """
    Visualize the best individual.
    Usage: python visualize.py <path_to_model>
    """
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <path_to_model>")
        exit()

    model_path = sys.argv[1]

    # Load the checkpoint from the specified file
    with open(model_path + "/best_individual.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)

    # Retrieve the best individual from the Hall of Fame in the checkpoint.
    best_individual = cp['halloffame'][0]

    print("Best individual:", best_individual)
    # View the expression tree of the best individual
    visualize_tree(best_individual, show=True, save=False)


