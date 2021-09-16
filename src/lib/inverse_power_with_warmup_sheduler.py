import torch


class InversePowerWithWarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup learning rate until `warmup_steps` then apply inverse power function.
    It is more flexible version of inverse sqrt function.
    base formula after warmup: 1 / (i + shift) ** power

    Parameters
    ----------
    optimizer : torch.optim
        PyTorch optimizer for model weights updates.
    peak_lr : float
        maximum learning rate at peak.
    warmup_steps : int
        Number of warmup steps.
    power : float
        power for LR function. Set to 1/2 for inverse sqrt function.
    shift : int
        shift helps control the duration of decay.
    last_epoch : int
        last_epoch is treated as last_step in this scheduler.
    """
    def __init__(self, optimizer, peak_lr, warmup_steps, power=0.5, shift=0, last_epoch=-1):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.power = power
        self.shift = shift
        self.warmup_rate = self.peak_lr / (self.warmup_steps)
        self.decay_factor = self.peak_lr * (self.warmup_steps + self.shift) ** self.power
        super(InversePowerWithWarmupLRScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        i = self.last_epoch
        if i < self.warmup_steps:
            lr = self.warmup_rate * (i+1)
        else:
            lr = self.decay_factor / (i + self.shift) ** self.power

        lrs = [lr for _ in self.optimizer.param_groups]
        return lrs