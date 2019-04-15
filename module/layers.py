import functools
import torch
import torch.nn as nn


class LabelSmoothLoss(nn.Module):

    def __init__(self, class_num, smooth=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.class_num = class_num
        self.smooth = smooth
        self.loss = nn.KLDivLoss()
        self.true_dist = None

    def forward(self, X, target):
        assert X.size(1) == self.class_num
        true_dist = X.clone()
        true_dist.fill_(self.smooth / (self.class_num - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smooth)
        true_dist = true_dist.detach()
        self.true_dist = true_dist

        return self.loss(X, true_dist)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
