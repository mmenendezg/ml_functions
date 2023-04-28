def exponential_decay(lr: float = 1e-4, s: int = 5):
    """Implements exponential decay of learning rate

    n(t) = lr * 0.1 ** (epoch / s)

    The function returns a function that will be passed to the callback.

    Example:

        exponential_decay_fn = exponential_decay(lr0=0.1, s=20)
        lr_scheduler = callbacks.LearningRateScgedyler(exponential_decay_fn)

    Args:
        lr (float): Initial value of learning rate.
        s (int): Number of steps in which the learning rate will decay.
    """

    def exponential_decay_fn(epoch):
        return lr * 0.1 ** (epoch / s)

    return exponential_decay_fn


def exponential_decay_with_warmup(
    lr_start: float = 1e-4,
    lr_max: float = 1e-3,
    lr_min: float = 1e-5,
    lr_rampup_epochs: int = 4,
    lr_sustain_epochs: int = 1,
    lr_exp_decay: float = 0.8,
):
    """Implements exponential decay learning rate with warm up.
    
    example:
        lr_function = exponential_decay_wtih_warmup()
        lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_function)

    Args:
        lr_start (float, optional): Initial value of the learning rate. Defaults to 0.0001.
        lr_max (float, optional): Maximum value of the learning rate. Defaults to 0.0001.
        lr_min (float, optional): Minimum value of the learning rate. Defaults to 0.00001.
        lr_rampup_epochs (int, optional): Number of epochs that the learning rate will increase up to lr_max. Defaults to 4.
        lr_sustain_epochs (int, optional): Number of epochs the learning rate will be equal to lr_max. Defaults to 1.
        lr_exp_decay (float, optional): Factor in which the learning rate will decay. Defaults to 0.8.
    """

    def exponential_decay_fn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < (lr_rampup_epochs + lr_sustain_epochs):
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min

        return lr

    return exponential_decay_fn


def piecewise_constant_fn(epoch: int):
    """Implements piecewise constant decay of learning rate.

    Args:
        epoch (int): Number of the current epoch.
    """

    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
