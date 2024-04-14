from torch.optim.lr_scheduler import _LRScheduler
import torch
from typing import List
import math


class PolyLR(_LRScheduler):
    """Polynomial scheduler used in deeplab training"""

    def __init__(
        self, optimizer, max_iters=1, end_learning_rate=0.0001, power=0.9, last_epoch=-1
    ):
        self.power = power
        self.max_iters = max_iters
        self.end_learning_rate = end_learning_rate
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def set_max_iters(self, max_iters, warmup_iters=1000):
        """sets the maximum number of iterations

        Args:
            max_iters (int): maximum number of iterations
        """
        self.max_iters = max_iters
        super(PolyLR, self).__init__(self.optimizer, -1)

    def get_lr(self):
        """returns learning rate at current step

        Returns:
            float: updated learning rate at this current step
        """
        if self.last_epoch > self.max_iters:
            print(
                "########Possible error in number of iters current {} and total {}".format(
                    self.last_epoch, self.max_iters
                )
            )
            return [self.end_learning_rate for _ in self.base_lrs]
        return [
            base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
            for base_lr in self.base_lrs
        ]


class WarmupPoly(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int = 1,
        warmup_factor: float = 0.001,
        warmup_iters_percentage: float = 0.1,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters_percentage = warmup_iters_percentage
        self.warmup_iters = max_iters * warmup_iters_percentage
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def set_max_iters(self, max_iters):
        """sets the maximum number of iterations

        Args:
            max_iters (int): maximum number of iterations
        """
        self.max_iters = max_iters
        self.warmup_iters = max_iters * self.warmup_iters_percentage
        super(WarmupPoly, self).__init__(self.optimizer, -1)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr
            * warmup_factor
            * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
