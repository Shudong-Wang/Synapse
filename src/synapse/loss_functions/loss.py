import torch

class CrossEntropyLossWeighted(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weight):
        ce_loss=self.ce_loss(pred, target)
        return (ce_loss*weight).sum() / weight.sum()
