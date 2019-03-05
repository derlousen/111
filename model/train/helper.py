class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        reset all parameters
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        update parameters
        """
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.PReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.BatchNorm2d,
        # nn.Sequential,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))