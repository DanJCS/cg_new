# config.py

SIM_PARAMS = {
    'n_agents': 500,               # Number of agents
    'm_dimensions': 2,              # Number of information dimensions
    'timesteps': 2000000,           # Total simulation timesteps (n_agents * 5000)
    'alpha': 0.2,                   # Influence factor
    'beta': 5.0,                    # Rationality parameter
    'epsilon': 0.1,                 # Probability of no feedback
    'gamma': -1.0,                  # Interpretation of silence
    'sigma': 0.05,            # Memory decay parameter (1 / (10 * m_dimensions))
    'zeta': 0,                      # Parameter controlling receiver's update on rejection.
    'eta': 0.5,                     # Parameter controlling receiver adaptation upon rejection.
    'graph_params': {
        'm_edges': 10,              # m parameter for Holme-Kim algorithm
        'p_triad': 0.9,            # p parameter for Holme-Kim algorithm
    },
    'save_history': False,          # Whether to save state vector history
    'record_interval': 500,        # Timestep interval for regular state vector sampling
    'random_sample_percentage': 0.01, # Percentage of agents to randomly sample at each timestep
    'initial_state_params': {       # Parameters for the initialization of agents' state vectors
        'alpha': 5.0,
        'beta': 5.0,
    },
    'seed' : 42,                  # Random Seed
}