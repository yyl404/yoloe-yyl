import os
import argparse
import yaml
import numpy as np
import pandas as pd
import shutil
from ultralytics import YOLO
from ultralytics.utils import yaml_load

from tools.incremental_utils import create_id_converted_dataset


def evaluate_model_on_task(model, task_data_path, task_classes, save_dir, model_task_idx, eval_task_idx, device="0"):
    """
    在指定任务上评估模型性能
    """
    # 获取模型类别
    model_classes = list(model.model.names.values())
    
    # 获取类别映射：任务ID -> 模型ID
    class_mapping = {}
    for task_idx, task_class in enumerate(task_classes):
        if task_class in model_classes:
            model_idx = model_classes.index(task_class)
            class_mapping[task_idx] = model_idx
    
    # 创建评估数据集
    dataset_name = f"task_{eval_task_idx}_eval_for_model_{model_task_idx}"
    data_path = create_id_converted_dataset(
        task_data_path, class_mapping, save_dir, dataset_name, model_classes
    )
    
    try:
        # 运行验证
        results = model.val(data=data_path, device=device, verbose=False, project=save_dir)
        
        # 提取mAP指标
        map50 = results.box.map50  # mAP@0.5
        map50_95 = results.box.map  # mAP@0.5:0.95
        
        return map50, map50_95, class_mapping
        
    finally:
        # 清理评估数据集
        if os.path.exists(os.path.dirname(data_path)):
            shutil.rmtree(os.path.dirname(data_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="增量学习配置文件路径")
    parser.add_argument("--save_dir", type=str, required=True, help="模型保存目录")
    parser.add_argument("--model_name", type=str, default="yolov8m", help="模型名称")
    parser.add_argument("--device", type=str, default="0", help="设备")
    parser.add_argument("--output", type=str, default="incremental_evaluation_results.csv", help="结果输出文件")
    args = parser.parse_args()
    
    # 读取任务配置
    with open(args.data, "r") as f:
        incremental_config = yaml.load(f, Loader=yaml.FullLoader)
        task_names = incremental_config["tasks"]
        task_data_paths = [os.path.join(os.path.dirname(args.data), task_names[k]) for k in sorted(task_names.keys())]
    
    num_tasks = len(task_data_paths)
    print(f"Found {num_tasks} tasks")
    
    # 读取每个任务的类别信息
    task_classes_list = []
    for data_path in task_data_paths:
        yaml_data = yaml_load(data_path)
        task_classes = [yaml_data['names'][k] for k in sorted(yaml_data['names'].keys())]
        task_classes_list.append(task_classes)
        print(f"Task {len(task_classes_list)-1}: {len(task_classes)} classes - {task_classes}")
    
    # 初始化结果矩阵
    map50_matrix = np.zeros((num_tasks, num_tasks))
    map50_95_matrix = np.zeros((num_tasks, num_tasks))
    

    
    # 对每个模型进行评估
    for model_task_idx in range(num_tasks):
        model_path = os.path.join(args.save_dir, f"{args.model_name}-task{model_task_idx}.pt")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Evaluating model from task {model_task_idx}")
        print(f"{'='*60}")
        
        # 加载模型
        model = YOLO(model_path)
        model_classes = list(model.model.names.values())
        print(f"Model classes ({len(model_classes)}): {model_classes}")
        
        # 在之前的所有任务上评估
        for eval_task_idx in range(model_task_idx + 1):
             print(f"\nEvaluating on task {eval_task_idx}...")
             
             task_classes = task_classes_list[eval_task_idx]
             print(f"Task {eval_task_idx} classes: {task_classes}")
             print(f"Creating evaluation dataset for Task {eval_task_idx} -> Model {model_task_idx}...")
             
             # 评估模型
             map50, map50_95, mapping = evaluate_model_on_task(
                 model, task_data_paths[eval_task_idx], task_classes, args.save_dir, model_task_idx, eval_task_idx, args.device
             )
             
             # 记录结果
             map50_matrix[eval_task_idx, model_task_idx] = map50
             map50_95_matrix[eval_task_idx, model_task_idx] = map50_95
             

             
             print(f"Task {eval_task_idx} -> Model {model_task_idx}: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")
             print(f"Class mapping: {mapping}")
             
             # 显示详细的类别映射信息
             if mapping:
                 print("Detailed class mapping:")
                 for task_idx, model_idx in mapping.items():
                     task_class = task_classes[task_idx]
                     model_class = model_classes[model_idx]
                     print(f"  Task[{task_idx}]: {task_class} -> Model[{model_idx}]: {model_class}")
             else:
                 print("No class mapping found - no common classes between model and task")
    
    # 创建结果DataFrame
    map50_df = pd.DataFrame(map50_matrix, 
                           index=[f"Task_{i}" for i in range(num_tasks)],
                           columns=[f"Model_{i}" for i in range(num_tasks)])
    
    map50_95_df = pd.DataFrame(map50_95_matrix,
                              index=[f"Task_{i}" for i in range(num_tasks)],
                              columns=[f"Model_{i}" for i in range(num_tasks)])
    

    
    # 保存到CSV
    map50_df.to_csv(os.path.join(args.save_dir, args.output.replace('.csv', '_mAP50.csv')))
    map50_95_df.to_csv(os.path.join(args.save_dir, args.output.replace('.csv', '_mAP50-95.csv')))
    
    # 打印结果摘要
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print("\nmAP@0.5 Matrix:")
    print(map50_df.round(4))
    
    print("\nmAP@0.5:0.95 Matrix:")
    print(map50_95_df.round(4))
    
    # 计算平均性能
    print(f"\nAverage mAP@0.5 per model:")
    for i in range(num_tasks):
        valid_values = map50_matrix[:i+1, i]
        if len(valid_values) > 0:
            avg_map50 = np.mean(valid_values)
            print(f"Model {i}: {avg_map50:.4f}")
    
    print(f"\nAverage mAP@0.5:0.95 per model:")
    for i in range(num_tasks):
        valid_values = map50_95_matrix[:i+1, i]
        if len(valid_values) > 0:
            avg_map50_95 = np.mean(valid_values)
            print(f"Model {i}: {avg_map50_95:.4f}")
    
    print(f"\nResults saved to:")
    print(f"  {args.output.replace('.csv', '_mAP50.csv')}")
    print(f"  {args.output.replace('.csv', '_mAP50-95.csv')}")


if __name__ == "__main__":
    main()
