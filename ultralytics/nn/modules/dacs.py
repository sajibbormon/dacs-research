import torch
import torch.nn as nn


def box_iou(box1, box2):
    area1 = box1[:, 2] * box1[:, 3]
    area2 = box2[:, 2] * box2[:, 3]

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 0] + box1[:, None, 2], box2[:, 0] + box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 1] + box1[:, None, 3], box2[:, 1] + box2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)


class DACS(nn.Module):
    def __init__(self, topk=100):
        super().__init__()
        self.topk = topk

        self.suppressor = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

        self.lambda_net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, boxes, scores, classes):

        # Top-K
        scores, idx = scores.topk(min(self.topk, len(scores)))
        boxes = boxes[idx]
        classes = classes[idx]

        N = boxes.size(0)

        iou = box_iou(boxes, boxes)

        # Density
        D = iou.mean(dim=1)

        # Pairwise features
        xi, yi, wi, hi = boxes.T
        xi, yi, wi, hi = xi.unsqueeze(1), yi.unsqueeze(1), wi.unsqueeze(1), hi.unsqueeze(1)

        features = torch.stack(
            [
                iou,
                torch.abs(xi - xi.T),
                torch.abs(yi - yi.T),
                torch.abs(wi - wi.T),
                torch.abs(hi - hi.T),
                scores.unsqueeze(1).expand(N, N),
                scores.unsqueeze(0).expand(N, N),
            ],
            dim=-1,
        ).view(-1, 7)

        s_ij = self.suppressor(features).view(N, N)

        # Class-aware mask
        class_mask = (classes.unsqueeze(1) == classes.unsqueeze(0)).float()
        s_ij = s_ij * class_mask

        # Lambda
        lambda_i = self.lambda_net(torch.cat([boxes, scores.unsqueeze(1)], dim=1)).squeeze()

        # Energy
        S = torch.sum(s_ij * iou, dim=1)
        E = lambda_i * S * D

        new_scores = scores * torch.exp(-E)

        return boxes, new_scores, classes
