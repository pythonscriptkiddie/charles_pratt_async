# pareto_ggg/slice_sampling.py
import numpy as np
import scipy.stats
from scipy.special import gammaln
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Tuple, List

def slice_sample(log_fn: Callable[[float], float], x0: float, width: float = 1.0, steps_out: int = 10,
                 lower: float = -np.inf, upper: float = np.inf, max_iter: int = 1000) -> float:
    """
    Perform univariate slice sampling.

    Parameters:
        log_fn: Callable returning the log-posterior at a given x
        x0: Initial point
        width: Initial bracket width
        steps_out: How many times to expand the interval
        lower: Minimum boundary for x
        upper: Maximum boundary for x
        max_iter: Maximum number of rejection attempts

    Returns:
        A sample from the target distribution.
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

def slice_sample_gamma_parameters(x: np.ndarray, cur_params: List[float], hyper: List[float],
                                   steps: int = 20, w: float = 1.0) -> np.ndarray:
    """
    Perform slice sampling to estimate shape and rate parameters of a gamma distribution.

    Parameters:
        x: Observed data samples
        cur_params: Initial [shape, rate] parameter guess
        hyper: Hyperparameters for the shape and rate priors
        steps: Number of slice samples
        w: Step width

    Returns:
        Updated [shape, rate] parameters.
    """
    def log_posterior(log_params: np.ndarray) -> float:
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
        def lp_i(val: float) -> float:
            temp = log_params.copy()
            temp[i] = val
            return log_posterior(temp)
        log_params[i] = slice_sample(lp_i, log_params[i], width=w)

    return np.exp(log_params)

def _log_posterior_k(k: float, params: Tuple) -> float:
    """Log-posterior for sampling k"""
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

def _log_posterior_lambda(lambda_: float, params: Tuple) -> float:
    """Log-posterior for sampling lambda"""
    x, tx, Tcal, k, tau, r, alpha = params
    if lambda_ <= 0:
        return -np.inf
    logF = np.log(np.clip(1 - scipy.stats.gamma.cdf(min(Tcal, tau) - tx, a=k, scale=1 / (k * lambda_)), 1e-300, None))
    return (
        (r - 1 + k * x) * np.log(lambda_) - lambda_ * (alpha + k * tx) + logF
    )

def _log_posterior_tau(tau_rel: float, params: Tuple) -> float:
    """Log-posterior for sampling tau"""
    k, lambda_, mu = params
    if tau_rel <= 0:
        return -np.inf
    return -mu * tau_rel + np.log(np.clip(scipy.stats.gamma.cdf(tau_rel, a=k, scale=1 / (k * lambda_)), 1e-300, None))

async def pggg_slice_sample_async(
    what: str, x: np.ndarray, tx: np.ndarray, Tcal: np.ndarray, litt: np.ndarray,
    k: np.ndarray, lambda_: np.ndarray, mu: np.ndarray, tau: np.ndarray,
    t: float, gamma: float, r: float, alpha: float, s: float, beta: float
) -> np.ndarray:
    """
    Asynchronous slice sampler for Pareto/GGG parameters.

    Parameters:
        what: which parameter to sample ('k', 'lambda', 'tau')
        x, tx, Tcal, litt, k, lambda_, mu, tau: vectors of data and params
        t, gamma, r, alpha, s, beta: hyperparameters

    Returns:
        Array of updated samples for the target parameter.
    """
    N = len(x)
    out = np.empty(N)

    def sample_one(i: int) -> Tuple[int, float]:
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

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        futures = [loop.run_in_executor(executor, sample_one, i) for i in range(N)]
        results = await asyncio.gather(*futures)
        for i, val in results:
            out[i] = val

    return out

def pggg_slice_sample(*args, **kwargs) -> np.ndarray:
    """
    Synchronous wrapper for async pggg_slice_sample_async.

    Returns:
        Sampled parameter values (numpy array).
    """
    return asyncio.run(pggg_slice_sample_async(*args, **kwargs))

def pggg_palive(x: np.ndarray, tx: np.ndarray, Tcal: np.ndarray,
                k: np.ndarray, lambda_: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Placeholder for computing P(alive) for each customer.

    Returns:
        Array of probabilities.
    """
    return np.ones_like(x) * 0.5  # Placeholder
