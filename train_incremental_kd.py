import os
import shutil

from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save

from tools.incremental_utils import merge_task_classes, process_pseudo_labels, transfer_weights


def main():
    # 配置参数
    task_data_paths = [
        "/root/data/voc_inc_10_10/task_0/voc.yaml",
        "/root/data/voc_inc_10_10/task_1/voc.yaml",
    ]
    init_model_path = "yolov8m.pt"
    model_cfg = "yolov8m.yaml"
    epochs = 300
    batch = 16
    device = "0"
    save_dir = f"runs/yolov8m_voc_inc_10_10_pseudo_labels_fromscratch"
    os.makedirs(save_dir, exist_ok=True)

    encountered_classes = []

    for i, data_path in enumerate(task_data_paths):
        if i > 0:
            if os.path.exists(os.path.join(save_dir, f"task_{i}_pseudo_labels")):
                shutil.rmtree(os.path.join(save_dir, f"task_{i}_pseudo_labels"))

            yaml_data = yaml_load(data_path)
            classes_current_task = list(yaml_data['names'].values())
            merged_classes, encountered_classes_id_to_new_id, task_classes_id_to_new_id = merge_task_classes(encountered_classes, classes_current_task)
            
            model_path = os.path.join(save_dir, f"yolov8m-task{i-1}.pt")

            # 处理伪标注：生成、转换、合并标注文件
            teacher_model = YOLO(model_path)
            process_pseudo_labels(teacher_model, yaml_data, data_path, save_dir, i,
                                  encountered_classes_id_to_new_id, task_classes_id_to_new_id)
            # 更新数据配置
            yaml_data['names'] = {k: v for k, v in enumerate(merged_classes)}
            yaml_data['path'] = os.path.abspath(os.path.join(save_dir, f"task_{i}_pseudo_labels"))
            yaml_save(data=yaml_data, file=os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml"))
            data_path = os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml")
            model = transfer_weights(model_path, model_cfg, encountered_classes_id_to_new_id, encountered_classes, merged_classes, save_dir)
            encountered_classes = merged_classes
        else:
            yaml_data = yaml_load(data_path)
            encountered_classes = [yaml_data['names'][k] for k in sorted(yaml_data['names'].keys())]
            model = YOLO(init_model_path)
        
        model.train(data=data_path,
                    epochs=epochs, close_mosaic=10, batch=batch, 
                    optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, 
                    weight_decay=0.025, momentum=0.9, workers=4, 
                    device=device, val_interval=1, project=save_dir)
        
        # 保存模型
        model.save(os.path.join(save_dir, f"yolov8m-task{i}.pt"))

if __name__ == "__main__":
    main()
