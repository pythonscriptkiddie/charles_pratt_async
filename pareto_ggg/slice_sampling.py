# pareto_ggg/slice_sampling.py
from typing import Tuple, Callable
import numpy as np
import scipy.stats
from scipy.special import gammaln
from concurrent.futures import ThreadPoolExecutor


def slice_sample(log_fn: Callable, x0: float, width=1.0, steps_out=10, lower=-np.inf,
                 upper=np.inf, max_iter=1000):
    """
    Perform univariate slice sampling from a log-probability function.

    Parameters
    ----------
    log_fn : callable
        Function returning the log-probability at a given point.

    x0 : float
        Initial point to start the sampling.

    width : float, default=1.0
        Initial width of the sampling interval.

    steps_out : int, default=10
        Number of steps to expand the interval before sampling.

    lower : float, default=-np.inf
        Lower bound for the parameter.

    upper : float, default=np.inf
        Upper bound for the parameter.

    max_iter : int, default=1000
        Maximum number of iterations to sample a valid point.

    Returns
    -------
    float
        A sample from the target distribution.

    Raises
    ------
    RuntimeError
        If no acceptable sample is found within max_iter iterations.
    """
    x = x0
    log_y = log_fn(x) - np.random.exponential()
    u = np.random.uniform(0, width)
    left = x - u
    right = x + (width - u)

    for _ in range(steps_out):
        if left <= lower or log_fn(left) < log_y:
            break
        left -= width
    for _ in range(steps_out):
        if right >= upper or log_fn(right) < log_y:
            break
        right += width

    for _ in range(max_iter):
        x_new = np.random.uniform(left, right)
        log_prob = log_fn(x_new)
        if log_prob >= log_y:
            return x_new
        elif x_new < x:
            left = x_new
        else:
            right = x_new

    raise RuntimeError("Slice sampling failed to converge")


def slice_sample_gamma_parameters(x, cur_params, hyper, steps=20, w=1.0):
    """
    Perform slice sampling to estimate shape and rate parameters of a Gamma distribution.

    Parameters
    ----------
    x : array-like
        Data points assumed to follow a Gamma distribution.

    cur_params : array-like of shape (2,)
        Current estimates for shape and rate parameters.

    hyper : array-like of shape (4,)
        Hyperparameters for the prior on shape and rate:
        [shape_prior_alpha, shape_prior_beta, rate_prior_alpha, rate_prior_beta].

    steps : int, default=20
        Number of slice samples (1 per parameter per iteration).

    w : float, default=1.0
        Step size for slice sampling.

    Returns
    -------
    np.ndarray
        Updated estimates for shape and rate parameters.
    """
    def log_posterior(log_params):
        shape = np.exp(log_params[0])
        rate = np.exp(log_params[1])
        len_x = len(x)
        sum_x = np.sum(x)
        sum_log_x = np.sum(np.log(x))
        h1, h2, h3, h4 = hyper
        return (
            len_x * (shape * np.log(rate) - gammaln(shape))
            + (shape - 1) * sum_log_x - rate * sum_x
            + (h1 - 1) * np.log(shape) - shape * h2
            + (h3 - 1) * np.log(rate) - rate * h4
        )

    log_params = np.log(cur_params)
    for i in range(len(log_params)):
        def lp_i(val):
            temp = log_params.copy()
            temp[i] = val
            return log_posterior(temp)
        log_params[i] = slice_sample(lp_i, log_params[i], width=w)

    return np.exp(log_params)


def _log_posterior_k(k, params):
    """
    Log-posterior for the shape parameter `k` in the Pareto/GGG model.

    Parameters
    ----------
    k : float
        Proposed value for the shape parameter.

    params : tuple
        Model components required for evaluation:
        (x, tx, Tcal, litt, lambda_, tau, t, gamma)

    Returns
    -------
    float
        Log-posterior value for `k`.
    """
    x, tx, Tcal, litt, lambda_, tau, t, gamma = params
    if k <= 0:
        return -np.inf
    logF = np.log(np.clip(1 - scipy.stats.gamma.cdf(min(Tcal, tau) - tx, a=k, scale=1 / (k * lambda_)), 1e-300, None))
    return (
        (t - 1) * np.log(k) - k * gamma
        + k * x * np.log(k * lambda_) - x * gammaln(k)
        - k * lambda_ * tx + (k - 1) * litt
        + logF
    )


def _log_posterior_lambda(lambda_: float, params: Tuple):
    """
    Log-posterior for the rate parameter `lambda` in the Pareto/GGG model.

    Parameters
    ----------
    lambda_ : float
        Proposed value for the rate parameter.

    params : tuple
        Model components required for evaluation:
        (x, tx, Tcal, k, tau, r, alpha)

    Returns
    -------
    float
        Log-posterior value for `lambda`.
    """
    x, tx, Tcal, k, tau, r, alpha = params
    if lambda_ <= 0:
        return -np.inf
    logF = np.log(np.clip(1 - scipy.stats.gamma.cdf(min(Tcal, tau) - tx, a=k, scale=1 / (k * lambda_)), 1e-300, None))
    return (
        (r - 1 + k * x) * np.log(lambda_) - lambda_ * (alpha + k * tx) + logF
    )


def _log_posterior_tau(tau_rel, params):
    """
    Log-posterior for the latent dropout time `tau`.

    Parameters
    ----------
    tau_rel : float
        Time since last transaction (tau - tx).

    params : tuple
        Model components required for evaluation:
        (k, lambda_, mu)

    Returns
    -------
    float
        Log-posterior value for `tau`.
    """
    k, lambda_, mu = params
    if tau_rel <= 0:
        return -np.inf
    return -mu * tau_rel + np.log(np.clip(scipy.stats.gamma.cdf(tau_rel, a=k, scale=1 / (k * lambda_)), 1e-300, None))


def pggg_slice_sample(what, x, tx, Tcal, litt, k, lambda_, mu, tau, t, gamma, r, alpha, s, beta):
    """
    Log-posterior for the latent dropout time `tau`.

    Parameters
    ----------
    tau_rel : float
        Time since last transaction (tau - tx).

    params : tuple
        Model components required for evaluation:
        (k, lambda_, mu)

    Returns
    -------
    float
        Log-posterior value for `tau`.
    """
    N = len(x)
    out = np.empty(N)

    def sample_one(i):
        if what == "k":
            params = (x[i], tx[i], Tcal[i], litt[i], lambda_[i], tau[i], t, gamma)
            log_fn = lambda val: _log_posterior_k(val, params)
            return i, slice_sample(log_fn, k[i], width=3 * np.sqrt(t) / gamma, lower=0.1, upper=50.0)

        elif what == "lambda":
            params = (x[i], tx[i], Tcal[i], k[i], tau[i], r, alpha)
            log_fn = lambda val: _log_posterior_lambda(val, params)
            return i, slice_sample(log_fn, lambda_[i], width=3 * np.sqrt(r) / alpha, lower=1e-6, upper=1e3)

        elif what == "tau":
            tau_init = min(Tcal[i] - tx[i], np.random.gamma(k[i], 1 / (k[i] * lambda_[i]))) / 2
            params = (k[i], lambda_[i], mu[i])
            log_fn = lambda val: _log_posterior_tau(val, params)
            return i, tx[i] + slice_sample(log_fn, tau_init, width=(Tcal[i] - tx[i]) / 2, lower=0, upper=Tcal[i] - tx[i])

        else:
            raise ValueError("Invalid sampling target")

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(sample_one, range(N)))
        for i, val in results:
            out[i] = val

    return out


def pggg_palive(x, tx, Tcal, k, lambda_, mu):
    '''Placeholder so we do not have to call the palive function.'''
    return np.ones_like(x) * 0.5  # Placeholder
