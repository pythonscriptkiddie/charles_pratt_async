# pareto_ggg/main.py
import numpy as np
import pandas as pd
import time
from pareto_ggg import utils, mcmc


def main():
    """
    Main entry point for running a demonstration of the Pareto/GGG model.

    - Simulates synthetic customer transaction data.
    - Runs MCMC sampling to estimate model parameters.
    - Prints summary statistics of population-level and individual-level draws.
    """
    total_start = time.perf_counter()

    # Define parameters for simulation
    params = dict(t=4.5, gamma=1.5, r=5, alpha=10, s=0.8, beta=12)

    # Simulate data
    sim_start = time.perf_counter()
    data = utils.generate_data(n=100, T_cal=32, T_star=32, params=params, seed=42)
    sim_end = time.perf_counter()
    print(f"Data generation took {sim_end - sim_start:.2f} seconds")

    cbs = data["cbs"]

    # Run MCMC
    mcmc_start = time.perf_counter()
    result = mcmc.run_chain(
        data={
            "x": cbs["x"].values,
            "t.x": cbs["t_x"].values,
            "T.cal": cbs["T.cal"].values,
            "litt": cbs["litt"].values
        },
        mcmc_steps=200, burnin=50, thin=10, trace=50
    )
    mcmc_end = time.perf_counter()
    print(f"MCMC sampling took {mcmc_end - mcmc_start:.2f} seconds")

    # Clamp gamma if needed
    final_level2 = result["level_2"][-1].copy()
    final_level2[1] = max(final_level2[1], 1e-2)  # Ensure gamma >= 1e-2

    # Print example output
    print("\nSampled Level 2 Parameters (last step):")
    for name, val in zip(["t", "gamma", "r", "alpha", "s", "beta"], final_level2):
        print(f"  {name:>6}: {val:10.4f}")

    print("\nSampled Level 1 Parameters for first customer:")
    param_labels = ["k", "lambda", "mu", "tau", "z"]
    df = pd.DataFrame(result["level_1"][:, :, 0], columns=param_labels)
    print(df.round(4).to_string(index=False))

    total_end = time.perf_counter()
    print(f"\nTotal runtime: {total_end - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
