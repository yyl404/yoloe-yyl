from ultralytics import YOLOE
import torch

det_model = YOLOE("yoloe-v8m.yaml")

YOLOE("yoloe-v8m-seg.pt")
state = torch.load("yoloe-v8m-seg.pt")

det_model.load(state["model"])
det_model.save("yoloe-v8m-seg-det.pt")

model = YOLOE("yoloe-v8m-seg-det.pt")
print(model.args)
