# pareto_ggg/utils.py
import numpy as np
import pandas as pd
from typing import Dict, Union


def generate_data(n: int, T_cal: Union[int, float], T_star: Union[int, float],
                  params: Dict[str, float], seed: int = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate customer transaction data under the Pareto/GGG model.

    This function generates synthetic data for `n` customers using a
    hierarchical Pareto/GGG model. It produces both:
    - A customer-by-summary (CBS) table with aggregated statistics, and
    - An event log (elog) of individual transactions.

    Parameters
    ----------
    n : int
        Number of customers to simulate.

    T_cal : int or float
        Length of the calibration period.

    T_star : int or float
        Length of the holdout period (not used directly, but affects total time window).

    params : dict
        Dictionary of global model parameters with keys:
        - "t" : shape hyperparameter for k (interpurchase rate)
        - "gamma" : rate hyperparameter for k
        - "r" : shape hyperparameter for lambda (transaction rate)
        - "alpha" : rate hyperparameter for lambda
        - "s" : shape hyperparameter for mu (dropout rate)
        - "beta" : rate hyperparameter for mu

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict of pd.DataFrame
        Dictionary with two keys:
        - "cbs" : DataFrame with one row per customer containing:
            - "x": number of transactions in calibration period
            - "t_x": time of last transaction in calibration period
            - "T.cal": calibration window length
            - "litt": log-interpurchase time for positive `x`
        - "elog" : DataFrame with raw event log, columns: ["cust", "t"]
    """
    if seed is not None:
        np.random.seed(seed)

    T_cal = np.full(n, T_cal)
    T_star = np.full(n, T_star)

    ks = np.random.gamma(params["t"], 1 / params["gamma"], size=n)
    lambdas = np.random.gamma(params["r"], 1 / params["alpha"], size=n)
    mus = np.random.gamma(params["s"], 1 / params["beta"], size=n)
    taus = np.random.exponential(1 / mus)

    elog = []
    for i in range(n):
        itt_draws = int(1.5 * (T_cal[i] + T_star[i]) * lambdas[i]) + 20
        itts = np.random.gamma(ks[i], 1 / (ks[i] * lambdas[i]), size=itt_draws)
        ts = np.cumsum(itts)
        ts = ts[ts <= taus[i]]
        elog.extend([(i, t) for t in ts if t <= T_cal[i] + T_star[i]])

    elog_df = pd.DataFrame(elog, columns=["cust", "t"])
    cbs = elog_df[elog_df.t <= T_cal.max()].groupby("cust").agg(
        x=("t", "count"),
        t_x=("t", "max")
    ).reindex(range(n), fill_value=0)
    cbs["T.cal"] = T_cal
    cbs["litt"] = 0.0

    cbs.loc[cbs["x"] > 0, "litt"] = np.log(
        cbs.loc[cbs["x"] > 0, "x"] / cbs.loc[cbs["x"] > 0, "t_x"]
    )
    cbs["litt"] = cbs["litt"].replace([np.inf, -np.inf], 0).fillna(0)

    return {"cbs": cbs.reset_index(drop=True), "elog": elog_df}