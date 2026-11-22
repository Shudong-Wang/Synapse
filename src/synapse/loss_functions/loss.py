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
        pred = pred[0]
        label_signal = extra_event_info[:, -1].long()
        list_num_classes = [4, 2, 2]
        pred_main = pred[:, 0:list_num_classes[0]]
        weight_main = extra_event_info[:,0]
        # two tasks with different weights
        pred_high_njet = pred[:,list_num_classes[0]:sum(list_num_classes[:2])]
        weight_high_njet = extra_event_info[:,1]

        pred_low_njet = pred[:,sum(list_num_classes[:2]):sum(list_num_classes[:3])]
        weight_low_njet = extra_event_info[:,2]

        ce_loss = self.ce_loss(pred_main, target) * weight_main + \
                self.ce_loss(pred_high_njet, label_signal) * weight_high_njet + \
                self.ce_loss(pred_low_njet, label_signal) * weight_low_njet
        loss = ce_loss.sum() / (weight_main.sum() + weight_high_njet.sum() + weight_low_njet.sum())

        scores = torch.cat([
                            torch.softmax(pred_main, dim=1),
                            torch.softmax(pred_high_njet, dim=1),
                            torch.softmax(pred_low_njet, dim=1),
                            ], dim=1)

        return loss, scores


class MyCELossSimple(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weight, extra_event_info=None, extra_obj_info=None):
        object_pred = pred[2]
        pred = pred[0]
        obj_target = extra_obj_info[1].long()
        obj_mask = extra_obj_info[2].bool()
        obj_target = obj_target.view(-1)
        obj_mask = obj_mask.view(-1)

        label_signal = extra_event_info[:, -1].long()
        pred_main = pred
        weight_main = extra_event_info[:,0]

        ce_loss = self.ce_loss(pred_main, target) * weight_main
        loss = ce_loss.sum() / (weight_main.sum())

        obj_loss = self.ce_loss(object_pred[obj_mask], obj_target[obj_mask]).mean()
        loss = loss + obj_loss * 0.01

        scores = torch.softmax(pred_main, dim=1)

        return loss, scores


class MyCELossZLoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weight, extra_event_info=None, extra_obj_info=None):
        object_pred = pred[2]
        pred = pred[0]
        obj_target = extra_obj_info[1].long()
        obj_mask = extra_obj_info[2].bool()
        obj_target = obj_target.view(-1)
        obj_mask = obj_mask.view(-1)

        label_signal = extra_event_info[:, -1].long()
        pred_main = pred
        weight_main = extra_event_info[:,0]

        ce_loss = self.ce_loss(pred_main, target) * weight_main
        loss = ce_loss.sum() / (weight_main.sum())

        obj_loss = self.ce_loss(object_pred[obj_mask], obj_target[obj_mask]).mean()
        loss = loss + obj_loss * 0.1

        scores = torch.softmax(pred_main, dim=1)

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