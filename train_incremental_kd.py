import os
import shutil
import argparse
import yaml

from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save

from tools.incremental_utils import merge_task_classes, process_pseudo_labels, create_id_converted_dataset, transfer_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_name", type=str, default="yolov8m")
    parser.add_argument("--init_model_path", type=str)
    parser.add_argument("--model_cfg", type=str, default="yolov8m.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save_dir", type=str, default="runs/incremental_learning_kd")
    parser.add_argument("--checkpoint_task", type=int, default=0)
    parser.add_argument("--pseudo_labels", action="store_true", default=False)
    parser.add_argument("--pseudo_conf", type=float, default=0.25, help="Confidence threshold for pseudo labels")
    args = parser.parse_args()
    
    task_data_paths = []
    with open(args.data, "r") as f:
        incremental_config = yaml.load(f, Loader=yaml.FullLoader)
        task_names = incremental_config["tasks"]
        task_data_paths = [os.path.join(os.path.dirname(args.data), task_names[k]) for k in sorted(task_names.keys())]

    encountered_classes = []
    for i, data_path in enumerate(task_data_paths):
        if i > 0:
            if os.path.exists(os.path.join(args.save_dir, f"task_{i}_pseudo_labels")):
                shutil.rmtree(os.path.join(args.save_dir, f"task_{i}_pseudo_labels"))

            yaml_data = yaml_load(data_path)
            classes_current_task = list(yaml_data['names'].values())
            merged_classes, encountered_classes_id_to_new_id, task_classes_id_to_new_id = merge_task_classes(encountered_classes, classes_current_task)
            
            model_path = os.path.join(args.save_dir, f"{args.model_name}-task{i-1}.pt")

            if i >= args.checkpoint_task:
                # 处理伪标注：生成、转换、合并标注文件
                if args.pseudo_labels:
                    teacher_model = YOLO(model_path)
                    process_pseudo_labels(teacher_model, yaml_data, data_path, args.save_dir, i,
                                          encountered_classes_id_to_new_id, task_classes_id_to_new_id, args.pseudo_conf)
                    # 更新数据配置
                    yaml_data['names'] = {k: v for k, v in enumerate(merged_classes)}
                    yaml_save(data=yaml_data, file=os.path.join(args.save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml"))
                    data_path = os.path.abspath(os.path.join(args.save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml"))
                else:
                    # 不生成伪标签，但是要按照类别id的映射来生成临时数据集
                    dataset_name = f"task_{i}_pseudo_labels"
                    data_path = create_id_converted_dataset(data_path, task_classes_id_to_new_id, args.save_dir, dataset_name, merged_classes)
                
                model = transfer_weights(model_path, args.model_cfg, encountered_classes_id_to_new_id, merged_classes, args.save_dir)
                # 输出迁移信息
                print("=" * 60)
                print("Classification Head Channel Mapping")
                print("=" * 60)
                print(f"Old classes ({len(encountered_classes)}):")
                for old_idx, cls in enumerate(encountered_classes):
                    print(f"   {old_idx:2d}: {cls}")
                print(f"New classes ({len(merged_classes)}):")
                for new_idx, cls in enumerate(merged_classes):
                    # 检查这个类别是否在旧类别中
                    is_kept = cls in encountered_classes
                    if is_kept:
                        print(f"   {new_idx:2d}: {cls} (kept)")
                    else:
                        print(f"   {new_idx:2d}: {cls} (new)")
                print("=" * 60)
            encountered_classes = merged_classes
        else:
            yaml_data = yaml_load(data_path)
            encountered_classes = [yaml_data['names'][k] for k in sorted(yaml_data['names'].keys())]
            model = YOLO(args.init_model_path) if args.init_model_path else YOLO(args.model_cfg)
        
        if i >= args.checkpoint_task: # 允许中断后从中断的任务继续训练
            print(f"Training model for task {i}")
            model.train(data=data_path,
                        epochs=args.epochs, close_mosaic=10, batch=args.batch, 
                        optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, 
                        weight_decay=0.025, momentum=0.9, workers=4, 
                        device=args.device, val_interval=1, project=args.save_dir)
        
            # 保存模型
            print(f"Saving model to {os.path.join(args.save_dir, f'{args.model_name}-task{i}.pt')}")
            model.save(os.path.join(args.save_dir, f"{args.model_name}-task{i}.pt"))

if __name__ == "__main__":
    main()
