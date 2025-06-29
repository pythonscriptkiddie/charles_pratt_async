# pareto_ggg/model.py
import numpy as np
from typing import Dict
from slice_sampling import slice_sample_gamma_parameters, pggg_slice_sample
import scipy.stats


def pggg_palive(x: np.ndarray, tx: np.ndarray, Tcal: np.ndarray,
                 k: np.ndarray, lambda_: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Compute the probability that a customer is still alive under the Pareto/GGG model.

    This function numerically evaluates the probability a customer is still active
    at the end of the observation period using a combination of gamma distribution
    survival functions and numerical integration via Simpson's 3/8 rule.

    Parameters
    ----------
    x : np.ndarray
        Number of repeat transactions observed for each individual.

    tx : np.ndarray
        Time of the last transaction for each individual.

    Tcal : np.ndarray
        End of the observation window for each individual.

    k : np.ndarray
        Shape parameter of the Gamma distribution for each individual.

    lambda_ : np.ndarray
        Rate parameter of the Gamma distribution for each individual.

    mu : np.ndarray
        Dropout rate for each individual.

    Returns
    -------
    np.ndarray
        Estimated probability that each customer is still alive at `Tcal`.
    """
    N = len(x)
    p_alive = np.zeros(N)
    for i in range(N):
        delta = Tcal[i] - tx[i]
        one_minus_F = 1 - scipy.stats.gamma.cdf(delta, a=k[i], scale=1 / (k[i] * lambda_[i]))
        numer = one_minus_F * np.exp(-mu[i] * Tcal[i])

        def integrand(t):
            inner = 1 - scipy.stats.gamma.cdf(t - tx[i], a=k[i], scale=1 / (k[i] * lambda_[i]))
            return np.exp(-mu[i] * t) * inner

        ts = np.linspace(tx[i], Tcal[i], 13)
        ys = integrand(ts)
        integral = (3 * (Tcal[i] - tx[i]) / 8 / 12) * (ys[0] + 3 * np.sum(ys[1:-1:3]) + 3 * np.sum(ys[2:-1:3]) + 2 * np.sum(ys[3:-1:3]) + ys[-1])

        denom = numer + mu[i] * integral
        p_alive[i] = numer / denom
    return p_alive


def draw_gamma_params(param_type: str, level_1: Dict[str, np.ndarray], level_2: Dict[str, float], hyper_prior: Dict[str, float]) -> np.ndarray:
    """
    Sample hyperparameters from the posterior distribution using slice sampling.

    Parameters
    ----------
    param_type : str
        The name of the parameter to update ('lambda', 'mu', or 'k').

    level_1 : dict
        Dictionary of individual-level parameter samples.

    level_2 : dict
        Dictionary of current group-level parameters.

    hyper_prior : dict
        Dictionary of hyperparameters for the gamma priors.

    Returns
    -------
    np.ndarray
        Sampled values for the hyperparameters corresponding to the specified `param_type`.
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
    Draw samples for the individual-level shape parameter `k` using slice sampling.

    Parameters
    ----------
    data : dict
        Observed dataset with transaction and calibration information.

    level_1 : dict
        Dictionary of current individual-level parameter samples.

    level_2 : dict
        Dictionary of group-level hyperparameter values.

    Returns
    -------
    np.ndarray
        Updated samples of `k` for each individual.
    """
    return pggg_slice_sample(
        "k", data["x"], data["t.x"], data["T.cal"], data["litt"],
        level_1["k"], level_1["lambda"], level_1["mu"], level_1["tau"],
        level_2["t"], level_2["gamma"], level_2["r"], level_2["alpha"], level_2["s"], level_2["beta"]
    )


def draw_lambda(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Draw samples for the individual-level rate parameter `lambda` using slice sampling.

    Parameters
    ----------
    data : dict
        Observed dataset with transaction and calibration information.

    level_1 : dict
        Dictionary of current individual-level parameter samples.

    level_2 : dict
        Dictionary of group-level hyperparameter values.

    Returns
    -------
    np.ndarray
        Updated samples of `lambda` for each individual.
    """
    return pggg_slice_sample(
        "lambda", data["x"], data["t.x"], data["T.cal"], data["litt"],
        level_1["k"], level_1["lambda"], level_1["mu"], level_1["tau"],
        level_2["t"], level_2["gamma"], level_2["r"], level_2["alpha"], level_2["s"], level_2["beta"]
    )


def draw_mu(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Sample dropout rate `mu` for each individual from the Gamma posterior.

    Parameters
    ----------
    data : dict
        Observed dataset (not used directly here).

    level_1 : dict
        Current individual-level parameter samples, must include `tau`.

    level_2 : dict
        Group-level parameters, must include `s` and `beta`.

    Returns
    -------
    np.ndarray
        Sampled values of `mu`, clipped to avoid numerical underflow.
    """
    tau = level_1["tau"]
    s = level_2["s"]
    beta = level_2["beta"]
    mu = np.random.gamma(shape=s + 1, scale=1 / (beta + tau))
    mu = np.clip(mu, np.exp(-30), None)
    return mu


def draw_tau(data: Dict[str, np.ndarray], level_1: Dict[str, np.ndarray], level_2: Dict[str, float]) -> np.ndarray:
    """
    Sample latent dropout time `tau` for each individual.

    The draw depends on whether a customer is considered "alive" or "not alive"
    at the end of the calibration period. If alive, `tau` is drawn from an
    exponential distribution; otherwise, it is sampled using slice sampling.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'x', 't.x', 'T.cal', and 'litt'.

    level_1 : dict
        Current individual-level parameter values including `k`, `lambda`, `mu`, and `tau`.

    level_2 : dict
        Group-level hyperparameters.

    Returns
    -------
    np.ndarray
        Updated values for `tau` for each individual.
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
