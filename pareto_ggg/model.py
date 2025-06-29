# pareto_ggg/model.py
import numpy as np
from typing import Dict, List
from pareto_ggg.slice_sampling import slice_sample_gamma_parameters, pggg_slice_sample, pggg_palive


def draw_gamma_params(param_type: str, level_1: Dict[str, np.ndarray], level_2: Dict[str, float], hyper_prior: Dict[str, float]) -> np.ndarray:
    """
    Draw hyperparameters for the specified parameter type using slice sampling.

    Parameters
    ----------
    param_type : str
        The name of the parameter to update ('lambda', 'mu', or 'k').

    level_1 : dict
        Dictionary of individual-level parameters.

    level_2 : dict
        Dictionary of current group-level parameters.

    hyper_prior : dict
        Dictionary of hyperparameters for the gamma priors.

    Returns
    -------
    np.ndarray
        Sampled values for the hyperparameters corresponding to the param_type.
    """
    if param_type == "lambda":
        x = level_1["lambda"]
        cur_params = [level_2["r"], level_2["alpha"]]
        hyper = [hyper_prior[k] for k in ["r_1", "r_2", "alpha_1", "alpha_2"]]
    elif param_type == "mu":
        x = level_1["mu"]
        cur_params = [level_2["s"], level_2["beta"]]
        hyper = [hyper_prior[k] for k in ["s_1", "s_2", "beta_1", "beta_2"]]
    elif param_type == "k":
        x = level_1["k"]
        cur_params = [level_2["t"], level_2["gamma"]]
        hyper = [hyper_prior[k] for k in ["t_1", "t_2", "gamma_1", "gamma_2"]]
    else:
        raise ValueError("Invalid param_type")

    return slice_sample_gamma_parameters(x, cur_params, hyper, steps=200, w=0.1)


def draw_k(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Draw new samples for individual-level shape parameter 'k' using slice sampling.

    Parameters
    ----------
    data : dict
        Observed data including counts, last transaction time, calibration period, and log-likelihood components.

    level_1 : dict
        Current samples of individual-level parameters.

    level_2 : dict
        Current samples of group-level parameters.

    Returns
    -------
    np.ndarray
        Updated values for 'k' for each individual.
    """
    return pggg_slice_sample(
        "k", data["x"], data["t.x"], data["T.cal"], data["litt"],
        level_1["k"], level_1["lambda"], level_1["mu"], level_1["tau"],
        level_2["t"], level_2["gamma"], level_2["r"], level_2["alpha"], level_2["s"], level_2["beta"]
    )


def draw_lambda(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Draw new samples for individual-level rate parameter 'lambda' using slice sampling.

    Parameters
    ----------
    data : dict
        Observed data.

    level_1 : dict
        Current samples of individual-level parameters.

    level_2 : dict
        Current samples of group-level parameters.

    Returns
    -------
    np.ndarray
        Updated values for 'lambda' for each individual.
    """
    return pggg_slice_sample(
        "lambda", data["x"], data["t.x"], data["T.cal"], data["litt"],
        level_1["k"], level_1["lambda"], level_1["mu"], level_1["tau"],
        level_2["t"], level_2["gamma"], level_2["r"], level_2["alpha"], level_2["s"], level_2["beta"]
    )


def draw_mu(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Draw new samples for individual-level dropout rate 'mu' from a Gamma posterior.

    Parameters
    ----------
    data : dict
        Observed data (not used directly in this function).

    level_1 : dict
        Dictionary containing current individual-level parameters including 'tau'.

    level_2 : dict
        Dictionary containing current group-level parameters including 's' and 'beta'.

    Returns
    -------
    np.ndarray
        Sampled dropout rates 'mu' for each individual.
    """
    tau = level_1["tau"]
    s = level_2["s"]
    beta = level_2["beta"]
    mu = np.random.gamma(shape=s + 1, scale=1 / (beta + tau))
    mu = np.clip(mu, np.exp(-30), None)
    return mu


def draw_tau(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Draw new samples for individual-level latent time-to-dropout 'tau'.

    Parameters
    ----------
    data : dict
        Observed data including:
        - "x": transaction counts
        - "t.x": last transaction times
        - "T.cal": end of observation window
        - "litt": log-likelihood terms for dead individuals

    level_1 : dict
        Current samples of individual-level parameters.

    level_2 : dict
        Current samples of group-level parameters.

    Returns
    -------
    np.ndarray
        Updated values for 'tau' for each individual.
    """
    N = len(data["x"])
    x = data["x"]
    tx = data["t.x"]
    Tcal = data["T.cal"]
    lambda_ = level_1["lambda"]
    k = level_1["k"]
    mu = level_1["mu"]

    p_alive = pggg_palive(x, tx, Tcal, k, lambda_, mu)
    alive = p_alive > np.random.uniform(size=N)

    tau = np.zeros(N)
    tau[alive] = Tcal[alive] + np.random.exponential(scale=1 / mu[alive])
    not_alive = ~alive
    if np.any(not_alive):
        tau[not_alive] = pggg_slice_sample(
            "tau", x[not_alive], tx[not_alive], Tcal[not_alive], data["litt"][not_alive],
            k[not_alive], lambda_[not_alive], mu[not_alive], level_1["tau"][not_alive],
            level_2["t"], level_2["gamma"], level_2["r"], level_2["alpha"], level_2["s"], level_2["beta"]
        )

    return tau
