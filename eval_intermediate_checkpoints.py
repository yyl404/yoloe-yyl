# 评估中间检查点
import os
import subprocess
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import shutil
from tools.incremental_utils import create_id_converted_dataset
from ultralytics.utils import yaml_load
from ultralytics import YOLO

def run(cmd, env=None):
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    model_root = "/root/datasets/yolov8l_voc_inc_15_5_fromscratch_pseudo_labels/task1/weights"
    data = ["/root/datasets/VOC_inc_15_5/task_1_cls_15/dataset.yaml",
            "/root/datasets/VOC_inc_15_5/task_2_cls_5/dataset.yaml"]
    data_names = ["task_1_cls_15", "task_2_cls_5"]
    save_dir = "/hy-tmp/yolov8l_voc_inc_15_5_fromscratch_pseudo_labels/val-task1"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["WANDB_DISABLED"] = "true"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # if os.path.exists(args.save_dir):
    #     shutil.rmtree(args.save_dir)
    # os.makedirs(args.save_dir)

    models = []
    for model_path in os.listdir(model_root):
        if model_path.endswith(".pt") and model_path.startswith("epoch"):
            models.append(os.path.join(model_root, model_path))
    
    models.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('epoch')[1]))

    # 按数据集顺序进行评估，先完成一个数据集的所有模型，再处理下一个数据集
    for dataset_idx, (data_path, data_name) in enumerate(zip(data, data_names)):
        print(f"\n开始评估数据集: {data_name} ({data_path})")
        
        # 读取数据集配置获取类别信息
        dataset_config = yaml_load(data_path)
        dataset_classes = list(dataset_config['names'].values())
        
        # 检查是否需要创建转换后的数据集
        need_converted_dataset = False
        converted_data_path = None
        
        # 检查第一个模型的类别是否与数据集一致
        first_model = YOLO(models[0])
        first_model_classes = list(first_model.model.names.values())
        
        if first_model_classes != dataset_classes:
            need_converted_dataset = True
            print(f"Dataset {data_name} needs ID conversion:")
            print(f"  Model classes ({len(first_model_classes)}): {first_model_classes}")
            print(f"  Dataset classes ({len(dataset_classes)}): {dataset_classes}")
            
            # 构建类别映射：数据集类别ID -> 模型类别ID
            class_mapping = {}
            for dataset_idx_inner, dataset_class in enumerate(dataset_classes):
                if dataset_class in first_model_classes:
                    model_idx = first_model_classes.index(dataset_class)
                    class_mapping[dataset_idx_inner] = model_idx
            
            # 创建转换后的数据集（只创建一次）
            dataset_name = f"converted_dataset_{data_name}"
            converted_dataset_dir = os.path.join(save_dir, dataset_name)
            
            # 检查是否已经存在转换后的数据集
            if os.path.exists(converted_dataset_dir):
                print(f"Using existing converted dataset: {converted_dataset_dir}")
                converted_data_path = os.path.join(converted_dataset_dir, 'dataconfig.yaml')
            else:
                print("Creating new converted dataset...")
                converted_data_path = create_id_converted_dataset(
                    data_path, class_mapping, save_dir, dataset_name, first_model_classes
                )
        else:
            print(f"Dataset {data_name} classes match model classes, no conversion needed.")
        
        # 对当前数据集评估所有模型
        for model_path in models:
            print(f"\n评估模型 {model_path.split('/')[-1]} 在数据集 {data_name} 上...")
            
            # 加载模型获取类别信息
            model = YOLO(model_path)
            model_classes = list(model.model.names.values())
            
            # 使用数据集路径
            if need_converted_dataset:
                data_path_to_use = converted_data_path
            else:
                data_path_to_use = data_path

            # 构建保存名称，使用用户指定的数据集名称
            save_name = f"results_{model_path.split('/')[-1].split('.')[0]}_{data_name}.pt"

            cmd = [
                sys.executable, f"{script_dir}/eval.py",
                "--model_path", model_path,
                "--data", data_path_to_use,
                "--project", save_dir,
                "--save_name", save_name,
                "--device", "0"
            ]
            run(cmd, env=env)
        
        # 清理转换后的数据集（如果存在且不再需要）
        if need_converted_dataset and converted_dataset_dir and os.path.exists(converted_dataset_dir):
            shutil.rmtree(converted_dataset_dir)
            print(f"Cleaned up converted dataset: {converted_dataset_dir}")
    
    # 读取所有评测结果，画出mAP50的折线图
    # 为每个数据集创建结果字典
    dataset_results = {}
    for data_name in data_names:
        dataset_results[data_name] = {'epochs': [], 'map50_values': []}
    
    # 遍历保存的结果文件
    for model_path in models:
        epoch_name = model_path.split('/')[-1].split('.')[0]
        epoch_num = int(epoch_name.split('epoch')[1])+1
        
        # 对每个数据集读取结果
        for data_name in data_names:
            result_file = os.path.join(save_dir, f"results_{epoch_name}_{data_name}.pt")
            
            if os.path.exists(result_file):
                try:
                    # 加载评测结果
                    results = torch.load(result_file, map_location='cpu')
                    
                    # 提取mAP50值
                    if isinstance(results, dict) and 'map50' in results:
                        map50 = float(results['map50'])
                    else:
                        raise ValueError(f"No mAP50 found in results: {results}")
                    
                    dataset_results[data_name]['epochs'].append(epoch_num)
                    dataset_results[data_name]['map50_values'].append(map50)
                    print(f"Epoch {epoch_num} - {data_name}: mAP50 = {map50:.4f}")
                    
                except Exception as e:
                    print(f"Error loading results for epoch {epoch_num} - {data_name}: {e}")
                    continue
    
    # 绘制mAP50折线图
    valid_datasets = [name for name, data in dataset_results.items() if data['epochs'] and data['map50_values']]
    
    if valid_datasets:
        plt.figure(figsize=(14, 10))
        
        # 定义颜色和标记
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # 为每个数据集绘制折线
        for i, data_name in enumerate(valid_datasets):
            data = dataset_results[data_name]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.plot(data['epochs'], data['map50_values'], 
                    color=color, marker=marker, linewidth=2, markersize=6, 
                    label=f'{data_name} (mAP50)', alpha=0.8)
            
            # 添加数值标签
            for epoch, map50 in zip(data['epochs'], data['map50_values']):
                plt.annotate(f'{map50:.4f}', 
                            (epoch, map50), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=8,
                            color=color)
        
        # 设置图表属性
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('mAP50', fontsize=14)
        plt.title('mAP50 vs Epoch', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # 设置x轴刻度
        all_epochs = set()
        for data in dataset_results.values():
            all_epochs.update(data['epochs'])
        all_epochs = sorted(list(all_epochs))
        plt.xticks(all_epochs, rotation=45)
        
        # 保存图表
        plot_path = os.path.join(save_dir, 'map50_vs_epoch.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.show()
        
        print(f"多数据集mAP50折线图已保存到: {plot_path}")
        
        # 打印每个数据集的最佳结果
        print("\n各数据集最佳结果:")
        for data_name in valid_datasets:
            data = dataset_results[data_name]
            best_epoch = data['epochs'][np.argmax(data['map50_values'])]
            best_map50 = max(data['map50_values'])
            print(f"  {data_name}: Epoch {best_epoch}, mAP50 = {best_map50:.4f}")
        
    else:
        print("没有找到有效的评测结果数据")
    