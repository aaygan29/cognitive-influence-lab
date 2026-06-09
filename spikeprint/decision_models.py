"""Decision models for Study 2 (NARPS spiking study). Requires torch + snntorch.

Lean by design: a rate MLP control and a spiking LIF net (current/direct input encoding, rate
readout), matched in width/training. Energy is counted honestly — rate MACs vs the SNN's analog
input MACs (current injection over T steps) plus measured hidden SynOps (spikes/decision). The
economic baselines (EV / prospect-theory logistic) and the GBT ceiling live in the runner via
sklearn. See PREREGISTRATION.md "Study 2".
"""
from __future__ import annotations

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


class RateMLP(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def macs_per_decision(self) -> int:
        return 2 * self.hidden + self.hidden * 1  # input->hidden + hidden->out (dense)


class SpikingMLP(nn.Module):
    """Current/direct input encoding -> LIF hidden (surrogate grad) -> rate readout. Counts spikes."""

    def __init__(self, hidden: int = 16, t_steps: int = 20, beta: float = 0.9):
        super().__init__()
        self.hidden = hidden
        self.t_steps = t_steps
        self.fc1 = nn.Linear(2, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        spk_sum = torch.zeros(x.shape[0], self.hidden, device=x.device)
        spike_count = torch.zeros((), device=x.device)
        for _ in range(self.t_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)  # analog current each step
            spk_sum = spk_sum + spk1
            spike_count = spike_count + spk1.sum()
        logit = self.fc2(spk_sum / self.t_steps).squeeze(-1)  # rate readout
        return logit, spike_count

    def input_macs_per_decision(self) -> int:
        return 2 * self.hidden * self.t_steps  # current injection every timestep


def train_model(model, x, y, epochs: int = 60, lr: float = 0.01, seed: int = 0, spiking: bool = False):
    torch.manual_seed(seed)
    xt = torch.tensor(np.asarray(x), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logit = model(xt)[0] if spiking else model(xt)
        lossf(logit, yt).backward()
        opt.step()
    return model


@torch.no_grad()
def predict_proba(model, x, spiking: bool = False):
    """Return (held-out probabilities, mean spikes/decision or None)."""
    model.eval()
    xt = torch.tensor(np.asarray(x), dtype=torch.float32)
    if spiking:
        logit, spikes = model(xt)
        return torch.sigmoid(logit).numpy(), float(spikes) / xt.shape[0]
    return torch.sigmoid(model(xt)).numpy(), None
