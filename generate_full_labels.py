import os
import shutil
import argparse
import warnings
import copy
from tqdm import tqdm

from ultralytics.utils import yaml_save, yaml_load
from tools.incremental_utils import merge_task_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inc_data", type=str, default="/hy-tmp/VOC_inc_15_1_1_1_1_1/incremental_config.yaml")
    parser.add_argument("--full_data", type=str, default="/hy-tmp/VOC/VOC.yaml")
    parser.add_argument("--output_dir", type=str, default="/hy-tmp/VOC_inc_15_1_1_1_1_1_full-labels")
    args = parser.parse_args()

    print("🚀 开始生成完整标签数据集...")
    print(f"📁 增量数据配置: {args.inc_data}")
    print(f"📁 完整数据配置: {args.full_data}")
    print(f"📁 输出目录: {args.output_dir}")

    splits = ["train", "val", "test"]

    print("\n📖 加载配置文件...")
    inc_data_config = yaml_load(args.inc_data)

    full_data_config = yaml_load(args.full_data)
    full_data_names = full_data_config["names"]

    print(f"✅ 配置加载完成")
    print(f"📊 完整数据集类别数量: {len(full_data_names)}")
    print(f"📋 任务数量: {len(inc_data_config['tasks'])}")

    if os.path.exists(args.output_dir):
        print(f"🗑️  清理已存在的输出目录: {args.output_dir}")
        shutil.rmtree(args.output_dir, ignore_errors=True)
    
    output_data_config = copy.deepcopy(inc_data_config)
    yaml_save(os.path.join(args.output_dir, "incremental_config.yaml"), output_data_config)
    print(f"💾 保存增量配置文件到: {os.path.join(args.output_dir, 'incremental_config.yaml')}")

    encountered_categories = []

    # 计算总任务数用于进度显示
    total_tasks = len(inc_data_config["tasks"])
    processed_images = 0
    
    print(f"\n🔄 开始处理 {total_tasks} 个任务...")
    
    for task_idx, (task_id, task_yaml) in enumerate(inc_data_config["tasks"].items(), 1):
        print(f"\n📋 处理任务 {task_idx}/{total_tasks}: {task_id}")
        print(f"   📄 任务配置文件: {task_yaml}")
        
        task_data_config = yaml_load(os.path.join(os.path.dirname(args.inc_data), task_yaml))
        
        output_task_data_config = copy.deepcopy(task_data_config)
        categories_current_task = [task_data_config["names"][k] for k in sorted(task_data_config["names"].keys())]
        
        print(f"   🏷️  当前任务类别: {categories_current_task}")
        
        merged_categories, encountered_cat_id_to_new_cat_id, task_cat_id_to_new_cat_id = merge_task_classes(encountered_categories, categories_current_task)
        encountered_categories = merged_categories
        
        print(f"   🔄 合并后总类别数: {len(encountered_categories)}")
        
        output_task_data_config["names"] = {k: v for k, v in enumerate(encountered_categories)}
        yaml_save(os.path.join(args.output_dir, task_yaml), output_task_data_config)
        print(f"   💾 保存任务配置到: {os.path.join(args.output_dir, task_yaml)}")

        # 计算当前任务的总图像数
        total_images_in_task = 0
        for split in splits:
            if split in task_data_config and split in full_data_config:
                image_dir = task_data_config[split]
                image_path = os.path.join(os.path.dirname(args.inc_data), os.path.dirname(task_yaml), image_dir)
                if os.path.exists(image_path):
                    total_images_in_task += len([f for f in os.listdir(image_path) if f.endswith('.jpg')])

        print(f"   📸 当前任务总图像数: {total_images_in_task}")
        
        for split in splits:
            if split not in task_data_config or split not in full_data_config:
                continue
            
            print(f"   📂 处理 {split} 分割...")
            full_data_label_dir = full_data_config[split].replace("images", "labels")

            image_dir = task_data_config[split]
            label_dir = image_dir.replace("images", "labels")
            image_path = os.path.join(os.path.dirname(args.inc_data), os.path.dirname(task_yaml), image_dir)
            
            if not os.path.exists(image_path):
                print(f"      ⚠️  图像目录不存在: {image_path}")
                continue
            
            image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
            
            processed_images_in_task = 0

            # 为每个图像目录创建进度条
            with tqdm(image_files, desc=f"      📸 {split}/{os.path.basename(image_dir)}", 
                        unit="img", leave=False) as pbar:
                
                for image_name in pbar:
                    label_name = image_name.replace(".jpg", ".txt")
                    output_label_path = os.path.join(args.output_dir, os.path.dirname(task_yaml), label_dir, label_name)
                    output_image_path = os.path.join(args.output_dir, os.path.dirname(task_yaml), image_dir, image_name)
                    full_data_label_path = os.path.join(os.path.dirname(args.full_data), full_data_label_dir, label_name)
                    
                    if not os.path.exists(full_data_label_path):
                        warnings.warn(f"Label file {full_data_label_path} not found")
                        continue
                    
                    # 处理标签文件
                    with open(full_data_label_path, "r") as f:
                        full_data_lines = f.readlines()
                        for _line in full_data_lines:
                            _line = _line.strip()
                            _line = _line.split(" ")
                            cat_id_in_full_data = int(_line[0])
                            cat_name_in_full_data = full_data_names[cat_id_in_full_data]
                            if split == "train" and cat_name_in_full_data in encountered_categories:
                                cat_id_in_output_label = encountered_categories.index(cat_name_in_full_data)
                                _line[0] = str(cat_id_in_output_label)
                                os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
                                with open(output_label_path, "a") as f:
                                    f.write(" ".join(_line) + "\n")
                            elif split != "train" and cat_name_in_full_data in categories_current_task:
                                # 验证集和测试集只保留当前任务的类别，其他类别的精度无关
                                cat_id_in_output_label = categories_current_task.index(cat_name_in_full_data)
                                _line[0] = str(cat_id_in_output_label)
                                os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
                                with open(output_label_path, "a") as f:
                                    f.write(" ".join(_line) + "\n")
                    
                    # 复制图像文件
                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                    shutil.copy(os.path.join(os.path.dirname(args.inc_data), os.path.dirname(task_yaml), image_dir, image_name), output_image_path)
                    
                    processed_images += 1
                    processed_images_in_task += 1
                    pbar.set_postfix({"processed": f"{processed_images_in_task}/{total_images_in_task}"})

    print(f"\n✅ 所有任务处理完成!")
    print(f"📊 最终统计:")
    print(f"   📋 处理的任务数: {total_tasks}")
    print(f"   🏷️  总类别数: {len(encountered_categories)}")
    print(f"   📸 总处理图像数: {processed_images}")
    print(f"   📁 输出目录: {args.output_dir}")
    print("🎉 完整标签生成完成!")