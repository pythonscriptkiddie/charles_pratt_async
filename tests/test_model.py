# pareto_ggg/tests/test_model.py
import numpy as np
import pytest
from pareto_ggg import utils, mcmc


def test_generate_data_shapes():
    params = dict(t=4.5, gamma=1.5, r=5, alpha=10, s=0.8, beta=12)
    data = utils.generate_data(n=10, T_cal=32, T_star=32, params=params, seed=123)
    assert "cbs" in data and "elog" in data
    assert data["cbs"].shape[0] == 10
    assert set(["x", "t_x", "T.cal", "litt"]).issubset(data["cbs"].columns)


def test_mcmc_output_shapes():
    params = dict(t=4.5, gamma=1.5, r=5, alpha=10, s=0.8, beta=12)
    data = utils.generate_data(n=5, T_cal=32, T_star=32, params=params, seed=123)
    cbs = data["cbs"]

    result = mcmc.run_chain(
        data={
            "x": cbs["x"].values,
            "t.x": cbs["t_x"].values,
            "T.cal": cbs["T.cal"].values,
            "litt": cbs["litt"].values
        },
        mcmc_steps=20, burnin=5, thin=5, trace=10
    )

    assert "level_1" in result and "level_2" in result
    level_1, level_2 = result["level_1"], result["level_2"]

    assert level_1.shape[2] == 5  # 5 customers
    assert level_1.shape[1] == 5  # 5 individual-level params
    assert level_2.shape[1] == 6  # 6 hyperparams
    assert level_1.shape[0] == level_2.shape[0]  # same number of MCMC draws