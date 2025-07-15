from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer, YOLOEPESegTrainer
from ultralytics.models.yolo.yoloe.train import YOLOETrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

os.environ["PYTHONHASHSEED"] = "0"

# data = "/root/data/voc_as_yolo/stage1/stage1.yaml"
# data = "/root/data/voc_as_yolo/stage2/stage2.yaml"
data = "/root/data/voc_as_yolo/voc.yaml"

model_path = "yoloe-v8m.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

# model = YOLOE(model_path)
# model = YOLOE("yoloe-v8m-seg-det.pt")
# model = YOLOE("trained_on_old_best.pt")
model = YOLOE("trained_on_new_best.pt")
# model = YOLOE("trained_on_all_best.pt")

# Ensure pe is set for classes
names = list(yaml_load(data)['names'].values())
tpe = model.get_text_pe(names)
pe_path = "voc-pe.pt"
torch.save({"names": names, "pe": tpe}, pe_path)


# Test model before training
# LOGGER.info("Testing model before training...")
# model.val(data=data, batch=1, device="0")
# print(model.is_fused())
LOGGER.info("Start training...")

# model.train(data=data, epochs=100, close_mosaic=10, batch=32, 
#             optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, \
#             weight_decay=0.025, momentum=0.9, workers=4, \
#             device="0", **extends, \
#             trainer=YOLOEPETrainer, train_pe_path=pe_path,
#             val_interval=1)
model.val(data=data, batch=1, device="0")
LOGGER.info("Training completed successfully!")