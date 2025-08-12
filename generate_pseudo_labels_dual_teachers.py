import argparse
import os
import json

from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save

from tools.incremental_utils import process_pseudo_labels_dual_teachers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--old_teacher_model_path", type=str)
    parser.add_argument("--new_teacher_model_path", type=str)
    parser.add_argument("--pseudo_labels_conf", type=float)
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--encountered_classes_id_to_new_id_json", type=str)
    parser.add_argument("--task_classes_id_to_new_id_json", type=str)
    parser.add_argument("--merged_classes_json", type=str)
    args = parser.parse_args()

    encountered_classes_id_to_new_id = {int(k): v for k, v in json.loads(args.encountered_classes_id_to_new_id_json).items()} # 将json中的字符串key转换为int
    task_classes_id_to_new_id = {int(k): v for k, v in json.loads(args.task_classes_id_to_new_id_json).items()} # 将json中的字符串key转换为int
    merged_classes = json.loads(args.merged_classes_json)

    old_teacher_model = YOLO(args.old_teacher_model_path)
    new_teacher_model = YOLO(args.new_teacher_model_path)
    process_pseudo_labels_dual_teachers(old_teacher_model, new_teacher_model, yaml_load(args.data), args.data, args.save_dir, args.task_id,
                                        encountered_classes_id_to_new_id, task_classes_id_to_new_id, args.pseudo_labels_conf)
    # 更新数据配置
    yaml_data = yaml_load(args.data)
    yaml_data['names'] = {k: v for k, v in enumerate(merged_classes)}
    yaml_save(data=yaml_data, file=os.path.join(args.save_dir, f"task_{args.task_id}_pseudo_labels/dataconfig.yaml"))
    os.path.abspath(os.path.join(args.save_dir, f"task_{args.task_id}_pseudo_labels/dataconfig.yaml"))