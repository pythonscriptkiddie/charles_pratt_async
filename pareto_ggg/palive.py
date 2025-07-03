# pareto_ggg/palive.py
import numpy as np
import scipy.stats

def pggg_palive_vectorized(
    x: np.ndarray,
    tx: np.ndarray,
    Tcal: np.ndarray,
    k: np.ndarray,
    lambda_: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    N = len(x)
    delta = Tcal - tx
    rate = k * lambda_
    scale = 1 / rate
    one_minus_F = 1 - scipy.stats.gamma.cdf(delta, a=k, scale=scale)
    numer = one_minus_F * np.exp(-mu * Tcal)

    p_alive = np.zeros(N)

    for i in range(N):
        ts = np.linspace(tx[i], Tcal[i], 13)
        t_diff = ts - tx[i]
        inner = 1 - scipy.stats.gamma.cdf(t_diff, a=k[i], scale=scale[i])
        ys = np.exp(-mu[i] * ts) * inner
        integral = (3 * (Tcal[i] - tx[i]) / 8 / 12) * (
            ys[0] + 3 * np.sum(ys[1:-1:3]) + 3 * np.sum(ys[2:-1:3]) + 2 * np.sum(ys[3:-1:3]) + ys[-1]
        )
        denom = numer[i] + mu[i] * integral
        p_alive[i] = numer[i] / denom

    return p_alive
