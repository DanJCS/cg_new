# run_parallel.py
import json
import os
import itertools
import multiprocessing
from run_single import run_single_simulation, save_results_to_json
import config
from tqdm import tqdm

def run_parameter_sweep(parameter_grid, output_dir, n_reps=1):
    """Runs simulations for all parameter combinations and repetitions.

    Args:
        parameter_grid (list): A list of dictionaries, each representing
            a combination of parameters to vary.
        output_dir (str): The directory to save results.
        n_reps (int): Number of repetitions for each parameter combination.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_tasks = []  # List to hold all simulation tasks (param_set, rep_num, output_dir)

    # Create a list of all parameter combinations
    for param_combination in parameter_grid:
        for rep_num in range(n_reps):
            params = config.SIM_PARAMS.copy()
            params.update(param_combination)
            all_tasks.append((params, rep_num, output_dir, list(param_combination.keys())))  # Pass swept parameter keys

    print(f"Running {len(all_tasks)} total simulations...")
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(run_and_save_single_wrapper, all_tasks),
                           total=len(all_tasks),
                           desc="Simulations Progress"))

    print(f"All simulations completed. Results saved to {output_dir}")
    return results

def run_and_save_single_wrapper(args):
    """Wrapper function to unpack args for imap"""
    return run_and_save_single(*args)

def run_and_save_single(params, rep_num, output_dir, swept_keys):
    """
    Runs a single simulation and saves the results, taking into account
    whether parameters are being swept or not.
    
    Args:
        params: Simulation parameters
        rep_num: Repetition number
        output_dir: Output directory
        swept_keys: List of parameter keys being swept
    """
    results = run_single_simulation(params, rep_num)

    param_string = ""
    # Use the swept_keys passed in instead of referencing PARAMETER_GRID
    swept_params = {k: v for k, v in params.items() if k in swept_keys}
    if swept_params:
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
    output_dir = "/Users/dj/Downloads/New CG Results/results_parallel"

    PARAMETER_GRID = [
        {"gamma": -1.0, "epsilon": 0.0},
        {"gamma": -1.0, "epsilon": 0.1},
        {"gamma": -1.0, "epsilon": 0.2},
        {"gamma": -1.0, "epsilon": 0.3},
        {"gamma": -1.0, "epsilon": 0.4},
        {"gamma": -1.0, "epsilon": 0.5},
        {"gamma": -0.5, "epsilon": 0.0},
        {"gamma": -0.5, "epsilon": 0.1},
        {"gamma": -0.5, "epsilon": 0.2},
        {"gamma": -0.5, "epsilon": 0.3},
        {"gamma": -0.5, "epsilon": 0.4},
        {"gamma": -0.5, "epsilon": 0.5},
        {"gamma": 0.0, "epsilon": 0.0},
        {"gamma": 0.0, "epsilon": 0.1},
        {"gamma": 0.0, "epsilon": 0.2},
        {"gamma": 0.0, "epsilon": 0.3},
        {"gamma": 0.0, "epsilon": 0.4},
        {"gamma": 0.0, "epsilon": 0.5},
        {"gamma": 0.5, "epsilon": 0.0},
        {"gamma": 0.5, "epsilon": 0.1},
        {"gamma": 0.5, "epsilon": 0.2},
        {"gamma": 0.5, "epsilon": 0.3},
        {"gamma": 0.5, "epsilon": 0.4},
        {"gamma": 0.5, "epsilon": 0.5},
        {"gamma": 1.0, "epsilon": 0.0},
        {"gamma": 1.0, "epsilon": 0.1},
        {"gamma": 1.0, "epsilon": 0.2},
        {"gamma": 1.0, "epsilon": 0.3},
        {"gamma": 1.0, "epsilon": 0.4},
        {"gamma": 1.0, "epsilon": 0.5}
    ]
    
    run_parameter_sweep(PARAMETER_GRID, output_dir, n_reps=2)