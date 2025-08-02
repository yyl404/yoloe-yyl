import os
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import yaml_model_load, yaml_save



# ------------------知识蒸馏相关函数------------------
def merge_task_classes(encountered_classes, classes_current_task):
    encountered_classes_id_to_new_id = {} # encountered_classes_id_to_new_id[encountered_class_id] = new_class_id
    task_classes_id_to_new_id = {} # task_classes_id_to_new_id[task_class_id] = new_class_id
    merged_classes = list(set(encountered_classes).union(classes_current_task))
    for new_cat_id, class_name in enumerate(merged_classes):
        if class_name in encountered_classes:
            encountered_classes_id_to_new_id[encountered_classes.index(class_name)] = new_cat_id
        if class_name in classes_current_task:
            task_classes_id_to_new_id[classes_current_task.index(class_name)] = new_cat_id
    
    return merged_classes, encountered_classes_id_to_new_id, task_classes_id_to_new_id


def convert_label_ids(label_lines, id_mapping):
    """转换标注文件中的类别ID"""
    converted_lines = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_cat_id = int(parts[0])
            if old_cat_id in id_mapping:
                new_cat_id = id_mapping[old_cat_id]
                parts[0] = str(new_cat_id)
                converted_lines.append(' '.join(parts) + '\n')
    return converted_lines


def read_and_convert_labels(labels_dir, id_mapping):
    """读取标注文件并转换类别ID"""
    labels = {}
    if os.path.exists(labels_dir):
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                labels[label_file] = convert_label_ids(lines, id_mapping)
    return labels


def merge_and_save_labels(original_labels, pseudo_labels, output_dir):
    """合并标注文件并保存"""
    os.makedirs(output_dir, exist_ok=True)
    all_files = set(original_labels.keys()) | set(pseudo_labels.keys())
    
    for label_file in all_files:
        merged_lines = []
        
        # 添加原始标注
        if label_file in original_labels:
            merged_lines.extend(original_labels[label_file])
        
        # 添加伪标注
        if label_file in pseudo_labels:
            merged_lines.extend(pseudo_labels[label_file])
        
        # 输出合并后的标注文件
        output_path = os.path.join(output_dir, label_file)
        with open(output_path, 'w') as f:
            f.writelines(merged_lines)


def process_pseudo_labels(teacher_model, yaml_data, data_path, save_dir, task_id, 
                          encountered_classes_id_to_new_id):
    """处理伪标注：生成、转换、合并标注文件"""
    splits = ['train', 'val']
    
    for split in splits:
        # 生成伪标注
        source = os.path.join(yaml_data['path'], yaml_data[split]) if 'path' in yaml_data.keys() else \
            os.path.join(os.path.dirname(data_path), yaml_data[split])
        
        # 使用teacher_model生成伪标注
        results = teacher_model.predict(source, conf=0.25, save_txt=True, save_conf=False, stream=True,
                                        project=save_dir, name=f"task_{task_id}_pseudo_labels/{split}", verbose=False)
        for result in tqdm(results, desc=f"Generating pseudo labels for {split}"):
            pass # 遍历结果生成器的同时会自动保存结果文件
        
        images_output_dir = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/images/{split}")
        # 直接复制图像文件而不是创建软链接，避免YOLO路径推断错误
        if os.path.exists(images_output_dir):
            shutil.rmtree(images_output_dir)
        shutil.copytree(source, images_output_dir)
        print(f"Copied images from {source} to {images_output_dir}")
        
        # 设置路径
        pseudo_labels_dir = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/{split}/labels")
        output_dir = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/labels/{split}")
        
        # 读取并转换原始标注
        original_labels_dir = os.path.join(yaml_data['path'], yaml_data[split].replace('images', 'labels')) if 'path' in yaml_data.keys() else \
            os.path.join(os.path.dirname(data_path), yaml_data[split].replace('images', 'labels'))
        # original_labels = read_and_convert_labels(original_labels_dir, task_classes_id_to_new_id)
        original_labels = {}
        
        # 读取并转换伪标注
        pseudo_labels = read_and_convert_labels(pseudo_labels_dir, encountered_classes_id_to_new_id)
        
        # 合并并保存标注
        merge_and_save_labels(original_labels, pseudo_labels, output_dir)
        
        # 删除临时生成的伪标注文件
        if os.path.exists(os.path.join(save_dir, f"task_{task_id}_pseudo_labels/{split}")):
            shutil.rmtree(os.path.join(save_dir, f"task_{task_id}_pseudo_labels/{split}"))


# ------------------增量学习相关函数------------------
def transfer_weights(ckpt_path, model_cfg, weight_transfer_map, encountered_classes, merged_classes, save_dir):
    weight = YOLO(ckpt_path).model.state_dict()
    model_cfg = yaml_model_load(model_cfg)
    model_cfg['nc'] = len(merged_classes)
    yaml_save(data=model_cfg, file=os.path.join(save_dir, "modelconfig_temp.yaml"))
    model = YOLO(os.path.join(save_dir, "modelconfig_temp.yaml"))
    new_weight = model.model.state_dict()

    # 权重迁移：分类层按映射迁移，其他层直接复制
    for key in new_weight.keys():
        if key in weight:
            # 处理cv3中最后的分类层权重（Conv2d层，即.2.weight）
            if 'cv3' in key and key.endswith('.2.weight'):
                # 根据transfer_map迁移权重
                for old_idx, new_idx in weight_transfer_map.items():
                    if old_idx < weight[key].shape[0] and new_idx < new_weight[key].shape[0]:
                        new_weight[key][new_idx] = weight[key][old_idx].clone()
            
            # 处理cv3中最后的分类层偏置（Conv2d层，即.2.bias）
            elif 'cv3' in key and key.endswith('.2.bias'):
                # 根据transfer_map迁移偏置
                for old_idx, new_idx in weight_transfer_map.items():
                    if old_idx < weight[key].shape[0] and new_idx < new_weight[key].shape[0]:
                        new_weight[key][new_idx] = weight[key][old_idx].clone()
            
            # 其他层直接复制（形状相同）
            else:
                new_weight[key] = weight[key].clone()
    
    # 输出迁移信息
    print("  分类头通道映射: {} -> {}".format(
        {k: v for k, v in enumerate(encountered_classes)},
        {k: v for k, v in enumerate(merged_classes)}
        )
    )

    model.model.load_state_dict(new_weight)
    model.save(os.path.join(save_dir, "model_temp.pt"))
    model = YOLO(os.path.join(save_dir, "model_temp.pt"))
    return model