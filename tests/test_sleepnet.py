import numpy as np
import pytest
import torch

from sleep_predictor import data as d
from sleep_predictor import sleepnet as sn

STAGE2 = d.HABIT_FEATURES + d.LAG_FEATURES


def test_forward_is_sum_of_branches():
    torch.manual_seed(0)
    net = sn.SleepNet(1, 5, n_history=3).eval()  # eval: dropout off, so calls are repeatable
    x_act, x_hab, x_hist = torch.randn(8, 1), torch.randn(8, 5), torch.randn(8, 7, 3)
    out = net(x_act, x_hab, x_hist)
    parts = net.branches(x_act, x_hab, x_hist)
    assert out.shape == (8,)
    assert set(parts) == {"activity", "habits", "history"}
    torch.testing.assert_close(out, net.bias + sum(parts.values()))


def test_decorrelation_penalty_is_squared_correlation():
    steps = torch.linspace(-1, 1, 50)
    assert float(sn.decorrelation_penalty(2 * steps + 1, steps)) == pytest.approx(1.0, abs=1e-5)
    assert float(sn.decorrelation_penalty(steps**2, steps)) == pytest.approx(0.0, abs=1e-5)


def test_history_windows_use_only_previous_days(raw):
    scaler = sn.Standardizer(np.zeros(len(sn.HISTORY_COLS)), np.ones(len(sn.HISTORY_COLS)))
    user = raw[raw.user_id == "U0000"]
    rows = user.iloc[[0, 5]]
    windows = sn.history_windows(raw, rows, scaler, days=7)
    steps_col = sn.HISTORY_COLS.index("steps")
    assert (windows[0] == 0).all()  # first day has no history
    np.testing.assert_allclose(windows[1, -5:, steps_col], user.steps.iloc[0:5].to_numpy(), rtol=1e-5)
    assert (windows[1, :2] == 0).all()
    assert user.steps.iloc[5] not in windows[1, :, steps_col]


def test_ensemble_contributions_sum_and_round_trip(dataset, raw, tmp_path):
    train, val, test = d.grouped_split(dataset, test_size=0.25, val_size=0.25)
    ens = sn.SleepNetEnsemble(d.ACTIVITY_FEATURES, STAGE2)
    ens.fit(train, val, raw, seeds=(0, 1))
    parts = ens.predict_contributions(test)
    raw_sum = parts.sum(axis=1).to_numpy()
    np.testing.assert_allclose(ens.predict(test), np.clip(raw_sum, *ens.target_range_), atol=1e-6)

    ens.save(tmp_path / "net.pt")
    loaded = sn.SleepNetEnsemble.load(tmp_path / "net.pt")
    np.testing.assert_allclose(loaded.predict(test), ens.predict(test), atol=1e-6)
