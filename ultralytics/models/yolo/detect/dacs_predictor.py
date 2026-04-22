from ultralytics.models.yolo.detect.predict import DetectionPredictor

class DACSPredictor(DetectionPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dacs_mode = "dacs"  # force DACS ON

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        print("🔥 DACS PREDICTOR ACTIVE")

        # force mode
        self.dacs_mode = "dacs"

        return super().postprocess(preds, img, orig_imgs, **kwargs)