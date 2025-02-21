# run_single.py
import json
import os
import numpy as np
import random
import core  # Import the core functions

def run_single_simulation(params, rep_num):
    """Runs a single simulation with the given parameters.

    Args:
        params (dict): A dictionary of simulation parameters.
        rep_num (int): The repetition number.

    Returns:
        dict: A dictionary containing the simulation results.
    """

    config = core.Config(params) # Create a Config object.
    core.set_seed(config.seed)

    # Generate the graph and get neighbors
    graph = core.generate_graph(config.graph_params)
    neighbors = core.get_neighbors(graph)

    # Initialize state vectors
    state_vectors = core.initialize_state_vectors(
        config.n_agents, config.m_dimensions, config.initial_state_params
    )

    # --- Main Simulation Loop ---
    sampled_states = {'regular': [], 'random': []}  # Initialize history
    for t in range(config.timesteps):
        # Select sender and receiver
        sender = random.randrange(config.n_agents)
        if not neighbors[sender]:  # Skip if the sender has no neighbors
            continue
        receiver = random.choice(neighbors[sender])

        # Get current state vectors
        sender_state = state_vectors[sender]
        receiver_state = state_vectors[receiver]

        # Calculate probabilities
        p_send = core.calculate_p_send(sender_state, config.beta)
        yi = np.random.choice(config.m_dimensions, p=p_send)  # Select information
        p_accept = core.calculate_p_accept(receiver_state, yi, config.beta)
        z = 1 if random.random() < p_accept else -1  # Acceptance decision
        zij = core.calculate_feedback(z, config.epsilon, config.gamma)

        # Update states
        new_sender_state = core.update_sender_state(sender_state, yi, zij, config.alpha)
        new_receiver_state = core.update_receiver_state(
            receiver_state, yi, z, config.alpha, config.zeta, config.eta
        )
        new_receiver_state = core.apply_decay(new_receiver_state, yi, config.sigma, config.alpha)

        # Update state_vectors array (NumPy efficient update)
        state_vectors[sender] = new_sender_state
        state_vectors[receiver] = new_receiver_state

        # --- History Recording (Conditional) ---
        if config.save_history:
            if t % config.record_interval == 0:
                sampled_states['regular'].append((t, state_vectors.copy()))  # Store the ENTIRE state
            if random.random() < config.random_sample_percentage:
                sampled_states['random'].append((t, sender, state_vectors[sender].copy()))

    # --- Prepare results dictionary ---
    results = {
        'params': params,
        'final_state_vectors': state_vectors.tolist(),  # Convert to list for JSON
        'graph_params': config.graph_params, # Save graph parameters
    }
    if config.save_history:
        results['sampled_states'] = sampled_states
        # Convert NumPy arrays to lists for JSON serialization
        results['sampled_states']['regular'] = [(t, s.tolist()) for t, s in sampled_states['regular']]
        results['sampled_states']['random'] = [(t, i, s.tolist()) for t, i, s in sampled_states['random']]

    return results

def save_results_to_json(results, output_dir, rep_num, param_string, save_history):
    """Saves the results to a JSON file, handling NumPy arrays correctly.
    Adds the underscore for the parameter string.
    """

    # Construct the base filename
    if param_string:
        base_filename = f"{param_string}_rep_{rep_num}"
    else:
        base_filename = f"rep_{rep_num}" #If no parameters are varied


    # Create the full path for saving
    json_filename = os.path.join(output_dir, f"{base_filename}.json")


    # Convert NumPy arrays to lists for JSON serialization
    results_serializable = results.copy()
    results_serializable['final_state_vectors'] = results['final_state_vectors']

    if save_history:

        # Convert NumPy arrays within 'regular' to lists
        regular_list = []
        for t, state_vectors in results['sampled_states']['regular']:
            regular_list.append((t, state_vectors))
        results_serializable['sampled_states']['regular'] = regular_list

        # Convert NumPy arrays within 'random' to lists
        random_list = []
        for t, agent_index, state_vector in results['sampled_states']['random']:
            random_list.append((t, agent_index, state_vector))
        results_serializable['sampled_states']['random'] = random_list

    # Save to JSON
    with open(json_filename, 'w') as f:
        json.dump(results_serializable, f, indent=4)


if __name__ == '__main__':
    import config
    # Example usage (single run)
    output_directory = "results_single"  # You can change this
    os.makedirs(output_directory, exist_ok=True)

    results = run_single_simulation(config.SIM_PARAMS, 0) # Repetition number 0
    param_string = ""
    save_results_to_json(results, output_directory, 0, param_string, config.SIM_PARAMS['save_history'] )
    print(f"Simulation completed. Results saved to {output_directory}")