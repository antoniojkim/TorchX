# -*- coding: utf-8 -*-
import torch

class BatchAccumulator:
    """Tool to help accumulate gradients before optimizer steps.

    Args:
        optimizer: Wrapped optimizer.
        interval: The number of times the gradient should be accumulated before the
            optimizer takes a step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, interval: int):
        self.optimizer = optimizer
        self.interval = interval

    def __enter__(self):
        self.count = 0
        self.optimizer.zero_grad()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.count > 0:
            self.step()

    def __next__(self):
        self.count += 1

        if self.count >= self.interval:
            self.count = 0
            self.step()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
