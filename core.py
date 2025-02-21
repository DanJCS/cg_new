# core.py
import networkx as nx
import numpy as np
import random


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def beta_extended(n_agents, initial_state_params):
    """
    Generates initial state vectors for agents, drawn from a Beta
    distribution and transformed to [-1, 1].

    Args:
        n_agents (int): Number of agents.
        initial_state_params (dict): Parameters for the Beta distribution
                                     (alpha and beta).

    Returns:
        np.ndarray:  A 2D NumPy array of shape (n_agents, m_dimensions)
                     where each row is an agent's initial state vector.
    """
    alpha = initial_state_params['alpha']
    beta = initial_state_params['beta']
    return 2 * np.random.beta(alpha, beta, size=(n_agents, initial_state_params['m_dimensions'])) - 1

def initialize_state_vectors(n_agents, m_dimensions, initial_state_params):
    """Initializes agent state vectors, handles different initialization methods."""
    initial_state_params['m_dimensions'] = m_dimensions
    return beta_extended(n_agents, initial_state_params)

def calculate_p_send(state_vector, beta):
    """Calculates the probability of sending each piece of information (Eq. 1)."""
    exp_values = np.exp(beta * state_vector)
    return exp_values / np.sum(exp_values)

def calculate_p_accept(state_vector, yi, beta):
    """Calculates the probability of accepting a piece of information (Eq. 2)."""
    return 1 / (1 + np.exp(-beta * state_vector[yi]))

def calculate_feedback(z, epsilon, gamma):
    """Calculates the feedback signal (Eq. 3)."""
    if random.random() < epsilon:
        return gamma
    else:
        return z

def update_sender_state(state_vector, yi, zij, alpha):
    """Updates the sender's state vector (Eq. 4a)."""
    updated_state = state_vector.copy()  # Crucial: work on a copy
    updated_state[yi] = (1 - alpha) * updated_state[yi] + alpha * zij
    return updated_state

def update_receiver_state(state_vector, yi, z, alpha, zeta, eta):
    """Updates the receiver's state vector (Eq. 4b and the f(z, x) function)."""
    updated_state = state_vector.copy() #Crucial: work on a copy
    if z == 1:
        updated_state[yi] = (1 - alpha) * updated_state[yi] + alpha * 1
    else:
        if zeta == 0:
            updated_state[yi] = (1 - alpha) * updated_state[yi] + alpha * updated_state[yi]
        else:
            updated_state[yi] = (1 - alpha) * updated_state[yi] + alpha * eta * (zeta**2) * (zeta- updated_state[yi])
    return updated_state

def apply_decay(state_vector, yi, sigma, alpha):
    """Applies decay to the receiver's state vector (Eq. 7)."""
    updated_state = state_vector.copy()  # Work on a copy
    for l in range(len(updated_state)):
        if l != yi:
            updated_state[l] = (1 - alpha * sigma) * updated_state[l] - sigma * alpha
    return updated_state

def generate_graph(graph_params):
    """Generates a NetworkX graph using the Holme-Kim algorithm."""
    n = graph_params['n_agents']
    m = graph_params['m_edges']
    p = graph_params['p_triad']
    # Use the provided seed for graph generation as well
    seed = graph_params['seed']
    set_seed(seed)  # Make sure this is called *before* graph generation

    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def get_neighbors(graph):
    """Precomputes neighbors for efficiency."""
    return {node: list(graph.neighbors(node)) for node in graph.nodes}

class Config:
    """Stores simulation parameters."""
    def __init__(self, params):
        self.n_agents = params['n_agents']
        self.m_dimensions = params['m_dimensions']
        self.timesteps = params['timesteps']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.sigma = params['sigma']
        self.zeta = params['zeta']
        self.eta = params['eta']
        self.graph_params = params['graph_params']
        self.save_history = params.get('save_history', False)  # Default to False
        self.record_interval = params.get('record_interval', 500) # Default to 500
        self.random_sample_percentage = params.get('random_sample_percentage', 0.01)  # Default to 0.01
        self.initial_state_params = params['initial_state_params']
        self.seed = params['seed']

        # Add the graph parameters to the general graph_params dictionary
        self.graph_params['n_agents'] = self.n_agents
        self.graph_params['seed'] = self.seed


    def __str__(self):
        return str(self.__dict__)