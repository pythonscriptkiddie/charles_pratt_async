# pareto_ggg/mcmc.py
import numpy as np
from pareto_ggg.model import draw_k, draw_lambda, draw_mu, draw_tau, draw_gamma_params


def run_chain(data, mcmc_steps=1000, burnin=200, thin=10, trace=100, param_init=None, hyper_prior=None):
    """
    Run a Markov Chain Monte Carlo (MCMC) chain for hierarchical Bayesian inference
    using the Pareto/GGG (Generalized Gamma-Gamma) model.

    Parameters:
    -----------
    data : dict
        A dictionary containing the observed data with keys:
        - "x" : array-like, observed counts
        - "t.x" : array-like, observed time of last event
        - "T.cal" : array-like, calibration period
    
    mcmc_steps : int, default=1000
        Number of MCMC iterations to collect (after burn-in).

    burnin : int, default=200
        Number of burn-in iterations before collecting samples.

    thin : int, default=10
        Thinning interval to reduce autocorrelation in the chain.

    trace : int or None, default=100
        If not None, prints progress every `trace` steps.

    param_init : dict or None
        Initial values for level-2 (hyper)parameters. If None, defaults are used.

    hyper_prior : dict or None
        Hyperparameters for the priors of the gamma distributions. If None, vague priors are used.

    Returns:
    --------
    dict
        A dictionary with two keys:
        - "level_1": ndarray of shape (draws, 5, N), samples for individual-level parameters:
            ['k', 'lambda', 'mu', 'tau', 'z']
        - "level_2": ndarray of shape (draws, 6), samples for group-level hyperparameters:
            ['t', 'gamma', 'r', 'alpha', 's', 'beta']
    """
    N = len(data["x"])
    draws_count = (mcmc_steps - 1) // thin + 1

    level_1_draws = np.full((draws_count, 5, N), np.nan)
    level_2_draws = np.full((draws_count, 6), np.nan)

    if hyper_prior is None:
        hyper_prior = dict(r_1=1e-3, r_2=1e-3, alpha_1=1e-3, alpha_2=1e-3,
                           s_1=1e-3, s_2=1e-3, beta_1=1e-3, beta_2=1e-3,
                           t_1=1e-3, t_2=1e-3, gamma_1=1e-3, gamma_2=1e-3)

    if param_init is None:
        param_init = dict(t=1, gamma=1, r=1, alpha=1, s=1, beta=1)

    level_2 = param_init.copy()
    level_1 = {
        "k": np.ones(N),
        "lambda": np.maximum(1e-3, np.mean(data["x"]) / np.maximum(data["t.x"], 1)),
        "tau": data["t.x"] + 0.5,
        "mu": 1 / (data["t.x"] + 0.5),
        "z": np.ones(N)
    }

    for step in range(burnin + mcmc_steps):
        if trace and step % trace == 0:
            print(f"Step {step}/{burnin + mcmc_steps}")

        if step >= burnin and ((step - burnin) % thin == 0):
            idx = (step - burnin) // thin
            for j, key in enumerate(["k", "lambda", "mu", "tau", "z"]):
                level_1_draws[idx, j, :] = level_1[key]
            level_2_draws[idx, :] = [level_2[k] for k in ["t", "gamma", "r", "alpha", "s", "beta"]]

        level_1["k"] = draw_k(data, level_1, level_2)
        level_1["lambda"] = draw_lambda(data, level_1, level_2)
        level_1["mu"] = draw_mu(data, level_1, level_2)
        level_1["tau"] = draw_tau(data, level_1, level_2)
        level_1["z"] = (level_1["tau"] > data["T.cal"]).astype(float)

        level_2.update(zip(["t", "gamma"], draw_gamma_params("k", level_1, level_2, hyper_prior)))
        level_2.update(zip(["r", "alpha"], draw_gamma_params("lambda", level_1, level_2, hyper_prior)))
        level_2.update(zip(["s", "beta"], draw_gamma_params("mu", level_1, level_2, hyper_prior)))

    return {"level_1": level_1_draws, "level_2": level_2_draws}
