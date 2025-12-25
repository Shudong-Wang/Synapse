import torch

class CrossEntropyLossWeighted(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weight, **kwargs):
        ce_loss=self.ce_loss(pred, target)
        scores = torch.softmax(logits, dim=1)
        return (ce_loss*weight).sum() / weight.sum(), scores

class MyCELoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weight, extra_event_info=None, extra_obj_info=None):
        pred_main = pred[:, 0:2]
        weight_main = extra_event_info[:,0]
        # two tasks with different weights
        pred_high_njet = pred[:,2:4]
        weight_high_njet = extra_event_info[:,1]

        pred_low_njet = pred[:,4:6]
        weight_low_njet = extra_event_info[:,2]

        # ce_loss = (self.ce_loss(pred_main, target) * weight_main).sum() / weight_main.sum() + \
        #         5 * (self.ce_loss(pred_high_njet, target) * weight_high_njet).sum() / weight_high_njet.sum() + \
        #         (self.ce_loss(pred_low_njet, target) * weight_low_njet).sum() / weight_low_njet.sum()
        ce_loss = self.ce_loss(pred_main, target) * weight_main + \
                self.ce_loss(pred_high_njet, target) * weight_high_njet + \
                self.ce_loss(pred_low_njet, target) * weight_low_njet
        loss = ce_loss.sum() / (weight_main.sum() + weight_high_njet.sum() + weight_low_njet.sum())

        scores = torch.cat([
                            torch.softmax(pred_main, dim=1),
                            torch.softmax(pred_high_njet, dim=1),
                            torch.softmax(pred_low_njet, dim=1),
                            ], dim=1)

        return loss, scores


# class SigmoidFocalBCELossWeighted(torch.nn.Module):
#     def __init__(self, central=0.9, sigma=0.05, **kwargs) -> None:
#         super().__init__()
#         self.bce_loss = torch.nn.BCELoss(reduction='none')
#         self.sigmoid_func = torch.nn.Sigmoid()
#         self.central = central
#         self.sigma = sigma

#     def forward(self, pred, target, weight):
#         score = torch.softmax(pred, dim=1)
#         prob = score[:,1]
#         bce_loss = self.bce_loss(prob, target.float())
#         with torch.no_grad():
#             focal_weight = self.sigmoid_func( (prob - self.central) / self.sigma ) * 2
#         return torch.mean(bce_loss * focal_weight * weight)