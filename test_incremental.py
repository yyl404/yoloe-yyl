import os
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
import subprocess
from ultralytics import YOLO
from ultralytics.utils import yaml_load


def evaluate_model_on_task_via_subprocess(model_path, task_data_path, task_classes, save_dir, model_task_idx, eval_task_idx, device="0"):
    """在子进程中评估模型，返回与原函数相同的数据结构。"""
    output_json_path = os.path.join(save_dir, "eval_tmp", f"task{eval_task_idx}_by_model{model_task_idx}.json")
    cmd = [
        "python", os.path.join(os.path.dirname(__file__), "eval_one_task.py"),
        "--task_data_path", task_data_path,
        "--task_classes_json", json.dumps(task_classes),
        "--save_dir", save_dir,
        "--model_path", model_path,
        "--model_task_idx", str(model_task_idx),
        "--eval_task_idx", str(eval_task_idx),
        "--device", device,
        "--output_json_path", output_json_path,
    ]
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    subprocess.run(cmd, check=True)

    with open(output_json_path, "r") as f:
        data = json.load(f)

    map50 = data["map50"]
    map50_95 = data["map50_95"]
    class_mapping = {int(k): int(v) for k, v in data.get("class_mapping_pairs", [])}
    class_ap50 = {int(k): float(v) for k, v in data.get("class_ap50_pairs", [])}
    class_ap50_95 = {int(k): float(v) for k, v in data.get("class_ap50_95_pairs", [])}

    return map50, map50_95, class_mapping, class_ap50, class_ap50_95


def create_detailed_results_dataframe(all_class_results, task_classes_list, num_tasks):
    """
    创建详细的类别级别结果DataFrame
    """
    # 收集所有列名
    all_columns = []
    task_column_mapping = {}  # 记录每个任务对应的列范围
    
    for task_idx, task_classes in enumerate(task_classes_list):
        task_start_col = len(all_columns)
        for class_idx, class_name in enumerate(task_classes):
            col_name = f"Task{task_idx}_{class_name}"
            all_columns.append(col_name)
        task_end_col = len(all_columns)
        task_column_mapping[task_idx] = (task_start_col, task_end_col)
    
    # 创建结果矩阵
    num_models = num_tasks
    results_matrix_ap50 = np.full((num_models, len(all_columns)), np.nan)
    results_matrix_ap50_95 = np.full((num_models, len(all_columns)), np.nan)
    
    # 填充结果
    for model_task_idx in range(num_models):
        for eval_task_idx in range(num_models):
            if eval_task_idx <= model_task_idx:  # 只评估已学习的任务
                key = (eval_task_idx, model_task_idx)
                if key in all_class_results:
                    class_ap50, class_ap50_95 = all_class_results[key]
                    
                    # 获取该任务对应的列范围
                    task_start_col, task_end_col = task_column_mapping[eval_task_idx]
                    
                    # 填充该任务中每个类别的结果
                    for class_idx in range(len(task_classes_list[eval_task_idx])):
                        col_idx = task_start_col + class_idx
                        if class_idx in class_ap50:
                            results_matrix_ap50[model_task_idx, col_idx] = class_ap50[class_idx]
                            results_matrix_ap50_95[model_task_idx, col_idx] = class_ap50_95[class_idx]
    
    # 创建DataFrame
    df_ap50 = pd.DataFrame(results_matrix_ap50, 
                     index=[f"Model_{i}" for i in range(num_models)],
                     columns=all_columns)
    df_ap50_95 = pd.DataFrame(results_matrix_ap50_95, 
                     index=[f"Model_{i}" for i in range(num_models)],
                     columns=all_columns)
    
    return df_ap50, df_ap50_95, task_column_mapping


def create_task_results_dataframe(all_task_results, num_tasks):
    """
    创建任务级别（以任务为列关键字）的结果DataFrame
    """
    # 初始化结果矩阵（模型 x 任务）
    results_matrix_ap50 = np.full((num_tasks, num_tasks), np.nan)
    results_matrix_ap50_95 = np.full((num_tasks, num_tasks), np.nan)

    for model_task_idx in range(num_tasks):
        for eval_task_idx in range(num_tasks):
            key = (eval_task_idx, model_task_idx)
            if key in all_task_results:
                map50, map50_95 = all_task_results[key]
                results_matrix_ap50[model_task_idx, eval_task_idx] = map50
                results_matrix_ap50_95[model_task_idx, eval_task_idx] = map50_95

    # 创建DataFrame
    columns = [f"Task{j}" for j in range(num_tasks)]
    index = [f"Model_{i}" for i in range(num_tasks)]

    df_task_ap50 = pd.DataFrame(results_matrix_ap50, index=index, columns=columns)
    df_task_ap50_95 = pd.DataFrame(results_matrix_ap50_95, index=index, columns=columns)

    return df_task_ap50, df_task_ap50_95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="增量学习配置文件路径")
    parser.add_argument("--save_dir", type=str, required=True, help="模型保存目录")
    parser.add_argument("--model_name", type=str, default="yolov8m", help="模型名称")
    parser.add_argument("--device", type=str, default="0", help="设备")
    parser.add_argument("--output", type=str, default="incremental_evaluation_results.csv", help="结果输出文件")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint路径")
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
    
    # 初始化结果存储
    all_class_results = {}  # (eval_task_idx, model_task_idx) -> (class_ap50, class_ap50_95)
    all_task_results = {}   # (eval_task_idx, model_task_idx) -> (map50, map50_95)
    start_model_task_idx = 0
    start_evaluation_task_idx = 0
    
    if args.checkpoint is not None:
        checkpoint_data = torch.load(args.checkpoint)
        all_class_results = checkpoint_data.get('all_class_results', {})
        all_task_results = checkpoint_data.get('all_task_results', {})
        start_model_task_idx = checkpoint_data.get('start_model_task_idx', 0)
        start_evaluation_task_idx = checkpoint_data.get('start_evaluation_task_idx', 0)
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"Start model task idx: {start_model_task_idx}")
        print(f"Start evaluation task idx: {start_evaluation_task_idx}")
    
    # 对每个模型进行评估，从start_model_task_idx开始
    for model_task_idx in range(start_model_task_idx, num_tasks):
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
        for eval_task_idx in range(start_evaluation_task_idx, model_task_idx + 1):
            print(f"\nEvaluating on task {eval_task_idx}...")
             
            task_classes = task_classes_list[eval_task_idx]
            print(f"Task {eval_task_idx} classes: {task_classes}")
            print(f"Creating evaluation dataset for Task {eval_task_idx} -> Model {model_task_idx}...")
             
            # 评估模型
            map50, map50_95, mapping, class_ap50, class_ap50_95 = evaluate_model_on_task_via_subprocess(
                model_path, task_data_paths[eval_task_idx], task_classes, args.save_dir, model_task_idx, eval_task_idx, args.device
            )

            # 显示类别映射信息
            if mapping:
                print("Detailed class mapping:")
                for task_idx, model_idx in mapping.items():
                    task_class = task_classes[task_idx]
                    model_class = model_classes[model_idx]
                    print(f"  Task[{task_idx}]: {task_class} -> Model[{model_idx}]: {model_class}")
            else:
                print("No class mapping found - no common classes between model and task")
             
            # 记录结果
            all_class_results[(eval_task_idx, model_task_idx)] = (class_ap50, class_ap50_95)
            all_task_results[(eval_task_idx, model_task_idx)] = (map50, map50_95)

            # 打印结果
            print(f"Results for Task {eval_task_idx} evaluated by Model {model_task_idx}:")
            print(f"  mAP@0.5: {map50:.4f}")
            print(f"  mAP@0.5:0.95: {map50_95:.4f}")
            print("  Per-class AP@0.5:")
            for idx, ap in class_ap50.items():
                class_name = task_classes[idx]
                print(f"    {class_name}: {ap:.4f}")
            print("  Per-class AP@0.5:0.95:")
            for idx, ap in class_ap50_95.items():
                class_name = task_classes[idx]
                print(f"    {class_name}: {ap:.4f}")

            # 保存checkpoint
            torch.save({
                'all_class_results': all_class_results,
                'all_task_results': all_task_results,
                'start_model_task_idx': model_task_idx,
                'start_evaluation_task_idx': eval_task_idx+1
            }, os.path.join(args.save_dir, f"evaluation_checkpoint.pt"))

        # 重置start_evaluation_task_idx
        start_evaluation_task_idx = 0 # start_evaluation_task_idx只会在读取checkpoint后用于从指定的验证任务开始继续验证，一旦被使用过后就重置为0
    
    # 创建详细（按类别）与普通（按任务）的结果DataFrame
    detailed_map50_df, detailed_map50_95_df, task_column_mapping = create_detailed_results_dataframe(
        all_class_results, task_classes_list, num_tasks
    )
    task_map50_df, task_map50_95_df = create_task_results_dataframe(all_task_results, num_tasks)
    
    # 保存到CSV
    # 普通版本（按任务列）：保持原有文件名约定
    task_map50_df.to_csv(os.path.join(args.save_dir, args.output.replace('.csv', '_mAP50.csv')))
    task_map50_95_df.to_csv(os.path.join(args.save_dir, args.output.replace('.csv', '_mAP50-95.csv')))

    # 详细版本（按类别列）：新增 detailed 文件名
    detailed_map50_df.to_csv(os.path.join(args.save_dir, args.output.replace('.csv', '_detailed_mAP50.csv')))
    detailed_map50_95_df.to_csv(os.path.join(args.save_dir, args.output.replace('.csv', '_detailed_mAP50-95.csv')))
    
    # 打印结果摘要
    print(f"\n{'='*80}")
    print("DETAILED EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print("\nmAP@0.5 (per task, NORMAL):")
    print(task_map50_df.round(4).fillna('-'))

    print("\nmAP@0.5:0.95 (per task, NORMAL):")
    print(task_map50_95_df.round(4).fillna('-'))

    print("\nmAP@0.5 Detailed Results (per class):")
    print(detailed_map50_df.round(4).fillna('-'))
    
    print("\nmAP@0.5:0.95 Detailed Results (per class):")
    print(detailed_map50_95_df.round(4).fillna('-'))
    
    # 按任务分组显示结果
    print(f"\n{'='*80}")
    print("RESULTS BY TASK")
    print(f"{'='*80}")
    
    for task_idx in range(num_tasks):
        task_start_col, task_end_col = task_column_mapping[task_idx]
        task_columns = detailed_map50_df.columns[task_start_col:task_end_col]
        
        print(f"\nTask {task_idx} Results (mAP@0.5):")
        task_df = detailed_map50_df[task_columns]
        print(task_df.round(4).fillna('-'))
        
        print(f"\nTask {task_idx} Results (mAP@0.5:0.95):")
        task_df_95 = detailed_map50_95_df[task_columns]
        print(task_df_95.round(4).fillna('-'))
    
    print(f"\nResults saved to:")
    print(f"  NORMAL (per task): {args.output.replace('.csv', '_mAP50.csv')}")
    print(f"  NORMAL (per task): {args.output.replace('.csv', '_mAP50-95.csv')}")
    print(f"  DETAILED (per class): {args.output.replace('.csv', '_detailed_mAP50.csv')}")
    print(f"  DETAILED (per class): {args.output.replace('.csv', '_detailed_mAP50-95.csv')}")


if __name__ == "__main__":
    main()
