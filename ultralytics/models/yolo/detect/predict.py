# Ultralytics 🚀 AGPL-3.0 License

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops
from ultralytics.nn.modules.dacs import DACS

import torch


class DetectionPredictor(BasePredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 🔥 IMPORTANT: initialize mode
        self.dacs_mode = "nms"   # default

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions with DACS integration."""

        print("🔥 CUSTOM PREDICTOR ACTIVE")  # debug

        save_feats = getattr(self, "_feats", None) is not None

        # 🔹 Mode control
        mode = self.dacs_mode
        print("MODE:", mode)

        # -------------------------------
        # 🔹 NMS (baseline or weak)
        # -------------------------------
        if mode == "nms":
            iou_thr = self.args.iou

        elif mode in ["weak_nms", "dacs"]:
            iou_thr = 0.95  # weaken NMS

        else:
            raise ValueError(f"Unknown mode: {mode}")

        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            iou_thr,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=300,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        # -------------------------------
        # Convert images
        # -------------------------------
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        # -------------------------------
        # Feature extraction
        # -------------------------------
        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        # -------------------------------
        # 🔥 Apply DACS
        # -------------------------------
        if mode == "dacs":

            print("🔥 DACS EXECUTING")  # debug

            device = preds[0].device if len(preds) else "cpu"
            dacs = DACS(topk=100).to(device)

            new_preds = []

            for pred in preds:
                if pred is None or len(pred) == 0:
                    new_preds.append(pred)
                    continue

                boxes = pred[:, :4]
                scores = pred[:, 4]
                classes = pred[:, 5]

                # DEBUG check
                print("Before scores:", scores[:3])

                boxes, scores, classes = dacs(boxes, scores, classes)

                print("After scores:", scores[:3])

                keep = scores > self.args.conf

                pred = torch.cat([
                    boxes[keep],
                    scores[keep].unsqueeze(1),
                    classes[keep].unsqueeze(1)
                ], dim=1)

                new_preds.append(pred)

            preds = new_preds

        # -------------------------------
        # Build results
        # -------------------------------
        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        import torch

        s = min(x.shape[1] for x in feat_maps)
        obj_feats = torch.cat(
            [
                x.permute(0, 2, 3, 1)
                .reshape(x.shape[0], -1, s, x.shape[1] // s)
                .mean(dim=-1)
                for x in feat_maps
            ],
            dim=1,
        )
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]

    def construct_results(self, preds, img, orig_imgs):
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])