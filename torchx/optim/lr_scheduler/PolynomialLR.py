# -*- coding: utf-8 -*-

import torch


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay scheduler.

    Args:
        optimizer: Wrapped optimizer.
        power: The polynomial factor at which the learning rate will decay
        starting_step: The step that the scheduler will start at.
        max_decay_steps: The maximum number of steps the learning rate will decay for.
        final_learning_rate: The learning rate that will be used after the maximum number
            of decay steps
        last_epoch: The index of last epoch. Default: -1.

    Example:
        >>> for epoch, learning_rate in enumerate(PolynomialLR(optimizer, power=4)):
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        power: float = 1,
        starting_step: int = 0,
        max_decay_steps: int = 100,
        final_learning_rate: float = 0.001,
        max_steps: int = 100,
        last_epoch: int = -1,
    ):
        assert max_decay_steps > 0

        self.optimizer = optimizer
        self.power = power
        self.current_step = starting_step
        self.starting_step = starting_step
        self.max_decay_steps = max_decay_steps
        self.final_learning_rate = final_learning_rate
        self.max_steps = max_steps
        self.last_epoch = last_epoch

        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        super().step(epoch)
        self.current_step += 1

    def get_lr(self):
        if self.current_step > self.max_decay_steps:
            return [self.final_learning_rate for lr in self.base_lrs]

        return [
            (base_lr - self.final_learning_rate)
            * (1 - self.current_step / self.max_decay_steps) ** self.power  # noqa: W503
            + self.final_learning_rate  # noqa: W503
            for base_lr in self.base_lrs
        ]

    @property
    def learning_rate(self):
        return self._last_lr[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step <= self.max_steps:
            if self.current_step > self.starting_step:
                self.step()

            return self.learning_rate
        else:
            raise StopIteration

    def __len__(self):
        return self.max_steps - self.starting_step
