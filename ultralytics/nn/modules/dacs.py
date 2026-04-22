import torch
import torch.nn as nn


def box_iou(box1, box2):
    # box format: x1, y1, x2, y2

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


class DACS(nn.Module):
    def __init__(self, topk=100):
        super().__init__()
        self.topk = topk

        # 🔹 pairwise suppressor
        self.suppressor = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 🔹 adaptive lambda
        self.lambda_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, boxes, scores, classes):

        if len(boxes) == 0:
            return boxes, scores, classes

        # ---------------------------
        # 🔹 Step 1: Top-K prefilter
        # ---------------------------
        k = min(self.topk, len(scores))
        scores, idx = scores.topk(k)
        boxes = boxes[idx]
        classes = classes[idx]

        N = boxes.size(0)

        # ---------------------------
        # 🔹 Step 2: IoU matrix
        # ---------------------------
        iou = box_iou(boxes, boxes)
        iou.fill_diagonal_(0)

        # ---------------------------
        # 🔹 Step 3: Density
        # ---------------------------
        D = iou.mean(dim=1)

        # ---------------------------
        # 🔹 Step 4: Pairwise features
        # ---------------------------
        xi, yi, x2i, y2i = boxes.T
        xi, yi, x2i, y2i = xi.unsqueeze(1), yi.unsqueeze(1), x2i.unsqueeze(1), y2i.unsqueeze(1)

        features = torch.stack([
            iou,
            torch.abs(xi - xi.T),
            torch.abs(yi - yi.T),
            torch.abs(x2i - x2i.T),
            torch.abs(y2i - y2i.T),
            scores.unsqueeze(1).expand(N, N),
            scores.unsqueeze(0).expand(N, N)
        ], dim=-1).view(-1, 7)

        s_ij = self.suppressor(features).view(N, N)

        # ---------------------------
        # 🔹 Class-aware masking
        # ---------------------------
        class_mask = (classes.unsqueeze(1) == classes.unsqueeze(0)).float()
        s_ij = s_ij * class_mask

        # ---------------------------
        # 🔹 Lambda
        # ---------------------------
        lambda_i = self.lambda_net(
            torch.cat([boxes, scores.unsqueeze(1)], dim=1)
        ).squeeze()

        # ---------------------------
        # 🔹 Energy-based suppression
        # ---------------------------
        # 🔥 only consider stronger neighbors
        score_j = scores.unsqueeze(0).expand(N, N)
        score_i = scores.unsqueeze(1).expand(N, N)

        mask = (score_j > score_i).float()

        S = torch.sum(s_ij * iou * mask, dim=1)
        E = lambda_i * S * D

        # 🔥 smooth score update (core idea)
        new_scores = scores * torch.exp(-E)

        # =========================================================
        # 🔥 FINAL STEP: GLOBAL RANKING ONLY (NO HARD RULES)
        # =========================================================
        k_final = min(50, len(new_scores))
        idx = torch.topk(new_scores, k_final).indices

        return boxes[idx], new_scores[idx], classes[idx]