import torch

class CrossEntropyLossWeighted(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weight):
        ce_loss=self.ce_loss(pred, target)
        return (ce_loss*weight).sum() / weight.sum()

class SigmoidFocalBCELossWeighted(torch.nn.Module):
    def __init__(self, central=0.9, sigma=0.05, **kwargs) -> None:
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        self.sigmoid_func = torch.nn.Sigmoid()
        self.central = central
        self.sigma = sigma

    def forward(self, pred, target, weight):
        score = torch.softmax(pred, dim=1)
        prob = score[:,1]
        bce_loss = self.bce_loss(prob, target.float())
        with torch.no_grad():
            focal_weight = self.sigmoid_func( (prob - self.central) / self.sigma ) * 2
        return torch.mean(bce_loss * focal_weight * weight)