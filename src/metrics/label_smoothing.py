import torch
from torch.autograd import Variable


class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, padding_idx=None, label_smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.true_dist = None
        
    def forward(self, x, target):
        size = x.size(1)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        if self.padding_idx is not None:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
