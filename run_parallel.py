# run_parallel.py (REVISED)
import json
import os
import itertools
import multiprocessing
from run_single import run_single_simulation, save_results_to_json  # Import the single-run function
import config  # Import default parameters
from tqdm import tqdm

def run_parameter_sweep(parameter_grid, output_dir, n_reps=1):
    """Runs simulations for all parameter combinations and repetitions."""
    os.makedirs(output_dir, exist_ok=True)

    all_tasks = []  # List to hold all simulation tasks (param_set, rep_num, swept_params)

    # Create a list of all parameter combinations
    for param_combination in parameter_grid:
        for rep_num in range(n_reps):
            # Create a copy of the default parameters
            params = config.SIM_PARAMS.copy()
            # Update with the specific parameter combination
            params.update(param_combination)
            # Create a dictionary of ONLY the swept parameters
            swept_params = {k: v for k, v in param_combination.items()}
            all_tasks.append((params, rep_num, swept_params)) # Add swept_params

    # Run simulations in parallel with progress bar
    print(f"Running {len(all_tasks)} total simulations...")
    with multiprocessing.Pool() as pool:
        # Use imap_unordered for better progress bar updates
        results = list(tqdm(pool.imap_unordered(run_and_save_single_wrapper, all_tasks),
                            total=len(all_tasks),
                            desc="Simulations Progress"))

    print(f"All simulations completed. Results saved to {output_dir}")
    return results  #Return results, for potential analysis.

def run_and_save_single_wrapper(args):
    """Wrapper function to unpack args for imap."""
    return run_and_save_single(*args)

def run_and_save_single(params, rep_num, swept_params): # Add swept_params
    """Runs a single simulation and saves the results."""
    results = run_single_simulation(params, rep_num)

    # Construct the parameter string for the filename (only swept parameters)
    param_string = ""
    if swept_params:  # Use the passed swept_params
        param_string = "_".join(f"{k}_{v}" for k, v in sorted(swept_params.items()))

    save_results_to_json(
        results,
        output_dir,
        rep_num,
        param_string,
        params['save_history']
    )
    return results

if __name__ == '__main__':
    # Example usage with a parameter grid (replace with your actual grid)

    # Define output directory
    output_dir = "User/danieljung/Downloads/results_parallel"

    PARAMETER_GRID = [
        {'alpha': 0.2, 'gamma': -1.0, 'epsilon': 0.1},
        {'alpha': 0.2, 'gamma': -1.0, 'epsilon': 0.2},
        {'alpha': 0.2, 'gamma':  0.0, 'epsilon': 0.1},
        {'alpha': 0.2, 'gamma':  0.0, 'epsilon': 0.2},
        {'alpha': 0.4, 'gamma': -1.0, 'epsilon': 0.1},
        {'alpha': 0.4, 'gamma': -1.0, 'epsilon': 0.2},
        {'alpha': 0.4, 'gamma':  0.0, 'epsilon': 0.1},
        {'alpha': 0.4, 'gamma':  0.0, 'epsilon': 0.2},
    ]
    #The above grid demonstrates two varied parameters and 2 static parameters. It also demonstrates what occurs when different number of values are present for
    #parameters.

    #PARAMETER_GRID = [
    #{'alpha':0.1,'beta':4.0,'gamma':-1.0, 'epsilon':0.2,'sigma': 0.025,'zeta':-1,'eta':0.4},
    #{'alpha':0.2,'beta':5.0,'gamma':0.0, 'epsilon':0.4,'sigma': 0.0125,'zeta':0,'eta':0.5},
    #{'alpha':0.3,'beta':6.0,'gamma':1.0, 'epsilon':0.6,'sigma': 0.008333,'zeta':1,'eta':0.6},
    #] #The above grid demonstrates an example of when all parameters are varied (except n_agents, m_dimensions, timesteps, those related to history and seed)
    #PARAMETER_GRID = [] #Demonstration for when no parameters are varied.

    # Example: run with 2 repetitions, saving to 'results_parallel'
    run_parameter_sweep(PARAMETER_GRID, output_dir, n_reps=2)