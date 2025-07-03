# pareto_ggg/utils.py
import numpy as np
import pandas as pd
from typing import Dict, Union

def generate_data(n: int, T_cal: Union[int, float], T_star: Union[int, float],
                  params: Dict[str, float], seed: int = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate customer-level transaction data under the Pareto/GGG model assumptions.

    Parameters:
        n: Number of customers to simulate
        T_cal: Calibration period length (can be scalar or array)
        T_star: Holdout period length (scalar)
        params: Dictionary with model parameters t, gamma, r, alpha, s, beta
        seed: Optional random seed for reproducibility

    Returns:
        Dictionary with two DataFrames:
            - 'cbs': customer-by-summary table (x, t_x, T.cal, litt)
            - 'elog': event log of all purchases (cust, t)
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
