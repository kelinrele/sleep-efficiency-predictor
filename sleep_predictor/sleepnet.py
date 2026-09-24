"""SleepNet: a neural network whose prediction is a sum of interpretable branches.

prediction = bias + activity(steps) + habits(habits, lags) [+ history(previous 7 days)]

Inputs are standardised with training-user statistics only, and the target is multiplied by
100 so the loss is in percentage points.
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import data as d

TARGET_SCALE = 100.0
HISTORY_COLS = ["steps", "stress_score", d.TARGET, "alcohol_units", "caffeine_mg", "screen_time_min"]


def set_seed(seed):
    torch.set_num_threads(4)  # small networks slow down when threads oversubscribe the CPU
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(n_in, hidden, dropout=0.0):
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.SiLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class SleepNet(nn.Module):
    def __init__(self, n_activity, n_habits, n_history=0, hidden_history=16):
        super().__init__()
        self.activity = mlp(n_activity, [32, 16])
        self.habits = mlp(n_habits, [64, 32], dropout=0.1)
        self.use_history = n_history > 0
        if self.use_history:
            self.gru = nn.GRU(n_history, hidden_history, batch_first=True)
            self.history_head = nn.Linear(hidden_history, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def branches(self, x_act, x_hab, x_hist=None):
        out = {
            "activity": self.activity(x_act).squeeze(-1),
            "habits": self.habits(x_hab).squeeze(-1),
        }
        if self.use_history:
            _, h = self.gru(x_hist)
            out["history"] = self.history_head(h[-1]).squeeze(-1)
        return out

    def forward(self, x_act, x_hab, x_hist=None):
        return self.bias + sum(self.branches(x_act, x_hab, x_hist).values())


class Standardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.where(np.asarray(std) > 0, std, 1.0).astype(np.float32)

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=np.float32)
        return cls(values.mean(axis=0), values.std(axis=0))

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def history_windows(raw, rows, standardizer, days=7):
    """For each row, the user's previous `days` days of HISTORY_COLS (never day t itself).

    Built from the full calendar; missing days and days before the series starts are zero,
    which is the training mean after standardisation.
    """
    raw = raw.sort_values([d.GROUP, "date"])
    out = np.zeros((len(rows), days, len(HISTORY_COLS)), dtype=np.float32)
    per_user = {}
    for user, frame in raw.groupby(d.GROUP, sort=False):
        values = np.nan_to_num(standardizer.transform(frame[HISTORY_COLS].to_numpy()), nan=0.0)
        per_user[user] = (dict(zip(frame["date"], range(len(frame)))), values)
    for i, (user, date) in enumerate(zip(rows[d.GROUP], rows["date"])):
        position, values = per_user[user]
        t = position[date]
        window = values[max(0, t - days):t]
        if len(window):
            out[i, days - len(window):] = window
    return out


def decorrelation_penalty(habits_out, steps_std):
    """Squared Pearson correlation between the habits branch and standardised steps."""
    h = habits_out - habits_out.mean()
    s = steps_std - steps_std.mean()
    corr = (h * s).sum() / (torch.sqrt((h**2).sum() * (s**2).sum()) + 1e-8)
    return corr**2


def train_one(
    tensors_train,
    tensors_val,
    y_train,
    y_val,
    n_history,
    seed,
    penalty_weight=0.0,
    epochs=200,
    patience=20,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
):
    """AdamW + Huber loss, early stopping on validation users; returns best model and curve."""
    set_seed(seed)
    model = SleepNet(tensors_train[0].shape[1], tensors_train[1].shape[1], n_history)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=2.0)
    n = len(y_train)
    generator = torch.Generator().manual_seed(seed)

    best_state, best_val, stale, curve = None, np.inf, 0, []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = [t[idx] for t in tensors_train]
            branches = model.branches(*batch)
            pred = model.bias + sum(branches.values())
            loss = loss_fn(pred, y_train[idx])
            if penalty_weight:
                loss = loss + penalty_weight * decorrelation_penalty(branches["habits"], batch[0][:, 0])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_mse = float(((model(*tensors_val) - y_val) ** 2).mean())
        curve.append(val_mse)
        if val_mse < best_val - 1e-4:
            best_val, best_state, stale = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, curve


class SleepNetEnsemble:
    """Several seeds of SleepNet averaged. Averaging keeps the branch sums additive."""

    def __init__(self, activity_features, habit_features, use_history=False, penalty_weight=0.0):
        self.activity_features = list(activity_features)
        self.habit_features = list(habit_features)
        self.use_history = use_history
        self.penalty_weight = penalty_weight
        self.models, self.curves = [], []

    def _tensors(self, frame, raw=None):
        tensors = [
            torch.tensor(self.act_scaler.transform(frame[self.activity_features].to_numpy())),
            torch.tensor(self.hab_scaler.transform(frame[self.habit_features].to_numpy())),
        ]
        if self.use_history:
            tensors.append(torch.tensor(history_windows(raw, frame, self.hist_scaler)))
        return tensors

    def fit(self, train, val, raw=None, seeds=(0, 1, 2, 3, 4)):
        self.act_scaler = Standardizer.fit(train[self.activity_features].to_numpy())
        self.hab_scaler = Standardizer.fit(train[self.habit_features].to_numpy())
        if self.use_history:
            train_raw = raw[raw[d.GROUP].isin(set(train[d.GROUP]))]
            self.hist_scaler = Standardizer.fit(train_raw[HISTORY_COLS].dropna().to_numpy())
        y_tr = torch.tensor(train[d.TARGET].to_numpy() * TARGET_SCALE, dtype=torch.float32)
        y_va = torch.tensor(val[d.TARGET].to_numpy() * TARGET_SCALE, dtype=torch.float32)
        t_tr, t_va = self._tensors(train, raw), self._tensors(val, raw)
        n_history = len(HISTORY_COLS) if self.use_history else 0
        self.target_range_ = (float(train[d.TARGET].min()), float(train[d.TARGET].max()))
        for seed in seeds:
            model, curve = train_one(
                t_tr, t_va, y_tr, y_va, n_history, seed, penalty_weight=self.penalty_weight
            )
            self.models.append(model)
            self.curves.append(curve)
        # A constant can sit in any branch, so the raw split between branches is arbitrary.
        # Centre every non-activity branch on its training mean (moving the constant into the
        # activity term), so "habits" means the shift from a typical training day.
        self.offsets_ = {}
        raw_parts = self._branches(train, raw)
        self.offsets_ = {k: float(raw_parts[k].mean()) for k in raw_parts if k != "activity"}
        return self

    def _branches(self, frame, raw=None, model=None):
        tensors = self._tensors(frame, raw)
        models = [model] if model is not None else self.models
        parts = []
        with torch.no_grad():
            for mdl in models:
                b = {k: v.numpy() / TARGET_SCALE for k, v in mdl.branches(*tensors).items()}
                # Fold the bias into the activity term so contributions sum to the prediction.
                b["activity"] = b["activity"] + float(mdl.bias) / TARGET_SCALE
                parts.append(b)
        out = pd.DataFrame({k: np.mean([p[k] for p in parts], axis=0) for k in parts[0]}, index=frame.index)
        for k, offset in getattr(self, "offsets_", {}).items():
            out[k] -= offset
            out["activity"] += offset
        return out

    def predict_contributions(self, frame, raw=None):
        return self._branches(frame, raw)

    def predict(self, frame, raw=None):
        return np.clip(self._branches(frame, raw).sum(axis=1).to_numpy(), *self.target_range_)

    def predict_each_seed(self, frame, raw=None):
        return [
            np.clip(self._branches(frame, raw, model=mdl).sum(axis=1).to_numpy(), *self.target_range_)
            for mdl in self.models
        ]

    def save(self, path):
        torch.save(
            {
                "activity_features": self.activity_features,
                "habit_features": self.habit_features,
                "use_history": self.use_history,
                "penalty_weight": self.penalty_weight,
                "target_range": list(self.target_range_),
                "act_scaler": self.act_scaler.state(),
                "hab_scaler": self.hab_scaler.state(),
                "hist_scaler": self.hist_scaler.state() if self.use_history else None,
                "offsets": self.offsets_,
                "states": [m.state_dict() for m in self.models],
            },
            path,
        )

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, weights_only=True)
        ens = cls(ckpt["activity_features"], ckpt["habit_features"], ckpt["use_history"], ckpt["penalty_weight"])
        ens.target_range_ = tuple(ckpt["target_range"])
        ens.act_scaler = Standardizer(**ckpt["act_scaler"])
        ens.hab_scaler = Standardizer(**ckpt["hab_scaler"])
        if ens.use_history:
            ens.hist_scaler = Standardizer(**ckpt["hist_scaler"])
        n_history = len(HISTORY_COLS) if ens.use_history else 0
        for state in ckpt["states"]:
            model = SleepNet(len(ens.activity_features), len(ens.habit_features), n_history)
            model.load_state_dict(state)
            model.eval()
            ens.models.append(model)
        ens.offsets_ = dict(ckpt["offsets"])
        return ens
