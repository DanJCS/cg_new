# generate_grid.py
import json
import numpy as np
import itertools
import argparse
import os

from core import Config  # Import Config to validate parameter names.
from config import SIM_PARAMS

def generate_parameter_grid(grid_config_path, output_path = None):
    """Generates a parameter grid based on a JSON configuration file."""

    with open(grid_config_path, 'r') as f:
        grid_config = json.load(f)

    # Validate config
    allowed_types = ["range", "list", "logspace"]
    valid_parameters = set(SIM_PARAMS.keys())


    for param_name, param_config in grid_config["parameters"].items():
        if param_name not in valid_parameters:
            raise ValueError(
                f"Invalid parameter name: {param_name}.  Must be one of {valid_parameters}"
            )
        param_type = param_config["type"]
        if param_type not in allowed_types:
            raise ValueError(
                f"Invalid parameter type: {param_type}. Must be one of {allowed_types}"
            )
        # Check for necessary keys
        if param_type == "range":
            if not all(k in param_config for k in ["start", "stop", "step"]):
                raise ValueError(f"Range type requires 'start', 'stop', and 'step' keys: {param_name}")
        elif param_type == "list":
            if "values" not in param_config:
                raise ValueError(f"List type requires 'values' key: {param_name}")
        elif param_type == "logspace":
            if not all(k in param_config for k in ["start","stop","num"]):
                raise ValueError(f"Logspace type requires 'start', 'stop', and 'num' keys: {param_name}")


    # Generate parameter values
    generated_params = {}
    for param_name, param_config in grid_config["parameters"].items():
        param_type = param_config["type"]
        if param_type == "range":
            start = param_config["start"]
            stop = param_config["stop"]
            step = param_config["step"]
            # Use np.arange and include the stop value.  Round to 6 s.f.
            generated_params[param_name] = np.round(np.arange(start, stop + step, step), 6).tolist()
        elif param_type == "list":
            generated_params[param_name] = param_config["values"]
        elif param_type == "logspace":
            start = param_config["start"]
            stop = param_config["stop"]
            num = param_config["num"]
            base = param_config.get("base", 10) # Default to base 10
            generated_params[param_name] = np.round(np.logspace(start, stop, num, base=base),6).tolist()

    # Create all combinations
    keys = list(generated_params.keys())
    value_combinations = list(itertools.product(*[generated_params[key] for key in keys]))

    # Convert combinations to list of dictionaries
    parameter_grid = [dict(zip(keys, combination)) for combination in value_combinations]

    # Output the result (formatted for easy copy/paste into run_parallel.py)
    if output_path is None:
        print("PARAMETER_GRID = [")
        print(json.dumps(parameter_grid, indent=4))
        print("]")
    else:
        with open(output_path, "w") as f:
            json.dump(parameter_grid,f, indent=4)
        print(f"Parameter grid saved to {output_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generates parameter grids for ABM simulations")
    parser.add_argument("--config", type=str, default = "grid_config.json", help="Path to JSON configuration for generating parameter grid (default: grid_config.json)")
    parser.add_argument("--output", type = str, default = None, help="Path to output file")
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")


    generate_parameter_grid(args.config, args.output)