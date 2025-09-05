from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save
from tools.incremental_utils import transfer_weights, merge_task_classes, create_id_converted_dataset
import os

data_task1 = "/root/datasets/VOC_inc_15_1_1_1_1_1_full-labels_only-train/task_5_cls_1/dataset.yaml"
data_task2 = "/root/datasets/VOC_inc_15_1_1_1_1_1_full-labels_only-train/task_6_cls_1/dataset.yaml"
model_path = "/root/datasets/yolov8m_voc_inc_15_1_1_1_1_1_full-labels-only-train_fromscratch_naive_checkpoints/train-task4/weights/best.pt"
save_dir = "/hy-tmp/yolov8m_voc_inc_15_1_1_1_1_1_full-labels-only-train_fromscratch_naive_checkpoints"

encountered_classes = [yaml_load(data_task1)['names'][k] for k in sorted(yaml_load(data_task1)['names'].keys())]
task_classes = [yaml_load(data_task2)['names'][k] for k in sorted(yaml_load(data_task2)['names'].keys())]
merged_classes, encountered_classes_id_to_new_id, task_classes_id_to_new_id = merge_task_classes(encountered_classes, task_classes)

transfer_weights(model_path, "yolov8m.yaml", encountered_classes_id_to_new_id, merged_classes, save_dir, "yolov8m-task4-transferred.pt")

model = YOLO(os.path.join(save_dir, "yolov8m-task4-transferred.pt"))
model.train(data=data_task2, epochs=30, batch=16, workers=2, device="cuda:0", project=save_dir, name="train-task5", val_interval=1, save_period=3)
