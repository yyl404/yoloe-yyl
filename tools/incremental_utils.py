import os
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import yaml_save, yaml_load
from ultralytics.nn.tasks import yaml_model_load


# ------------------知识蒸馏相关函数------------------
def merge_task_classes(encountered_classes, classes_current_task):
    encountered_classes_id_to_new_id = {} # encountered_classes_id_to_new_id[encountered_class_id] = new_class_id
    task_classes_id_to_new_id = {} # task_classes_id_to_new_id[task_class_id] = new_class_id
    merged_classes = sorted(list(set(encountered_classes).union(classes_current_task))) # 排序后可以保证可复现性
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


def merge_labels(original_labels, pseudo_labels):
    """合并标注文件"""
    merged_labels = {}
    all_files = set(original_labels.keys()) | set(pseudo_labels.keys())
    
    for label_file in all_files:
        merged_lines = []
        
        # 添加原始标注
        if label_file in original_labels:
            merged_lines.extend(original_labels[label_file])
        
        # 添加伪标注
        if label_file in pseudo_labels:
            merged_lines.extend(pseudo_labels[label_file])
        
        merged_labels[label_file] = merged_lines
    
    return merged_labels


def save_labels(labels, output_dir):
    """保存标注文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for label_file, lines in labels.items():
        output_path = os.path.join(output_dir, label_file)
        with open(output_path, 'w') as f:
            f.writelines(lines)


def create_id_converted_dataset(task_data_path, class_mapping, save_dir, dataset_name, new_classes_names=None):
    """
    创建临时数据集，转换标签ID以匹配模型输出
    通用的数据集创建函数，支持评估和训练场景
    
    Args:
        task_data_path: 任务数据配置文件路径
        class_mapping: 类别ID映射 {old_id: new_id}
        save_dir: 保存目录
        dataset_name: 数据集目录名称
        model_classes: 模型类别列表（用于评估场景，训练场景可为None）
    
    Returns:
        tuple: (config_path, dataset_dir) 或 (config_path, class_mapping, dataset_dir) 用于评估场景
    """
    # 读取原始数据配置
    yaml_data = yaml_load(task_data_path)
    
    # 创建数据集目录
    dataset_dir = os.path.abspath(os.path.join(save_dir, dataset_name))
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 复制图像文件并转换标签
    for split in ['train', 'val']:
        if split in yaml_data:
            # 源图像目录
            source_images = os.path.join(yaml_data['path'], yaml_data[split]) if 'path' in yaml_data.keys() else \
                os.path.join(os.path.dirname(task_data_path), yaml_data[split])
            
            # 目标图像目录
            target_images = os.path.join(dataset_dir, f"images/{split}")
            os.makedirs(target_images, exist_ok=True)
            shutil.copytree(source_images, target_images, dirs_exist_ok=True)
            
            # 源标签目录
            source_labels = os.path.join(yaml_data['path'], yaml_data[split].replace('images', 'labels')) if 'path' in yaml_data.keys() else \
                os.path.join(os.path.dirname(task_data_path), yaml_data[split].replace('images', 'labels'))
            
            # 目标标签目录
            target_labels = os.path.join(dataset_dir, f"labels/{split}")
            os.makedirs(target_labels, exist_ok=True)
            
            # 转换标签ID并保存
            if os.path.exists(source_labels):
                converted_labels = read_and_convert_labels(source_labels, class_mapping)
                save_labels(converted_labels, target_labels)
    
    # 创建配置文件
    config = {
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: cls for i, cls in enumerate(new_classes_names)}
    }
    config_path = os.path.join(dataset_dir, 'dataconfig.yaml')
    yaml_save(data=config, file=config_path)
    return config_path


def generate_pseudo_labels(teacher_model, source, save_dir, task_id, split, conf_threshold=0.25, name=None):
    """生成伪标注
    
    Args:
        teacher_model: 教师模型
        source: 图像源目录
        save_dir: 保存目录
        task_id: 任务ID
        split: 数据集分割（train/val）
        conf_threshold: 置信度阈值，默认0.25
    """
    if name is None:
        name = f"task_{task_id}_pseudo_labels/{split}"
    results = teacher_model.predict(source, conf=conf_threshold, save_txt=True, save_conf=False, stream=True,
                                    project=save_dir, name=name, verbose=False)
    for result in tqdm(results, desc=f"Generating pseudo labels for {split}", total=len(os.listdir(source)),
                       position=0, leave=True, ncols=80):
        pass # 遍历结果生成器的同时会自动保存结果文件


def copy_images_and_labels(target_dir, yaml_data, data_path, split):
    """复制图像文件"""
    # 复制图像
    source_images = os.path.join(yaml_data['path'], yaml_data[split]) if 'path' in yaml_data.keys() else \
        os.path.join(os.path.dirname(data_path), yaml_data[split])
    
    target_images = os.path.join(target_dir, f"images/{split}")
    if os.path.exists(target_images):
        shutil.rmtree(target_images)
    shutil.copytree(source_images, target_images)
    print(f"Copied images from {source_images} to {target_images}")


def process_pseudo_labels(teacher_model, yaml_data, data_path, save_dir, task_id, 
                          encountered_classes_id_to_new_id, task_classes_id_to_new_id, conf_threshold=0.25):
    """处理伪标注：生成、转换、合并标注文件
    
    Args:
        teacher_model: 教师模型
        yaml_data: 数据配置
        data_path: 数据路径
        save_dir: 保存目录
        task_id: 任务ID
        encountered_classes_id_to_new_id: 已遇到类别ID到新ID的映射
        task_classes_id_to_new_id: 任务类别ID到新ID的映射
        conf_threshold: 伪标签置信度阈值，默认0.25
    """
    splits = ['train', 'val']
    
    for split in splits:
        # 生成伪标注
        source = os.path.join(yaml_data['path'], yaml_data[split]) if 'path' in yaml_data.keys() else \
            os.path.join(os.path.dirname(data_path), yaml_data[split])
        
        # 1. 生成伪标注
        if split == 'train':
            generate_pseudo_labels(teacher_model, source, save_dir, task_id, split, conf_threshold)
        
        # 2. 复制图像文件
        images_output_dir = os.path.join(save_dir, f"task_{task_id}_pseudo_labels")
        copy_images_and_labels(images_output_dir, yaml_data, data_path, split)
        
        # 3. 设置路径
        if split == 'train':
            pseudo_labels_dir = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/{split}/labels")
        else:
            pseudo_labels_dir = None
        output_dir = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/labels/{split}")
        
        # 4. 转换原始标注ID
        original_labels_dir = os.path.join(yaml_data['path'], yaml_data[split].replace('images', 'labels')) if 'path' in yaml_data.keys() else \
            os.path.join(os.path.dirname(data_path), yaml_data[split].replace('images', 'labels'))
        original_labels = read_and_convert_labels(original_labels_dir, task_classes_id_to_new_id)
        
        # 5. 转换伪标注ID
        if split == 'train':
            pseudo_labels = read_and_convert_labels(pseudo_labels_dir, encountered_classes_id_to_new_id)
        else:
            pseudo_labels = {}
        # 6. 合并标签
        merged_labels = merge_labels(original_labels, pseudo_labels)
        
        # 7. 保存数据集
        save_labels(merged_labels, output_dir)
        
        # 删除临时生成的伪标注文件
        if os.path.exists(os.path.join(save_dir, f"task_{task_id}_pseudo_labels/{split}")):
            shutil.rmtree(os.path.join(save_dir, f"task_{task_id}_pseudo_labels/{split}"))


def process_pseudo_labels_dual_teachers(teacher_model_old, teacher_model_new, yaml_data, data_path, save_dir, task_id, 
                                        encountered_classes_id_to_new_id, task_classes_id_to_new_id, conf_threshold=0.25):
    """处理伪标注：生成、转换、合并标注文件，使用两个教师模型同时生成训练集的伪标签
    
    Args:
        teacher_model_old: 旧类别教师模型
        teacher_model_new: 新类别教师模型
        yaml_data: 数据配置
        data_path: 数据路径
        save_dir: 保存目录
        task_id: 任务ID
        encountered_classes_id_to_new_id: 已遇到类别ID到新ID的映射
        task_classes_id_to_new_id: 任务类别ID到新ID的映射
        conf_threshold: 伪标签置信度阈值，默认0.25

    teacher_model_old是在encountered_classes上进行训练过的模型，输出分类的通道是按照encountered_classes_id的顺序排列的，
    teacher_model_new是在task_classes上进行训练过的模型，输出分类的通道是按照task_classes_id的顺序排列的，
    需要分别使用两个模型对数据集的训练集进行推理，并将结果的cat id分别映射到new_id，然后合并两个模型的伪标签，作为新的数据集的训练标签。
    新数据集的验证集标签使用原本的ground truth标签，只需要将其cat id（task_classes_id）映射到new_id，然后保存为新的数据集的验证标签。
    需要将图像从原数据集的目录复制到新数据集对应目录。
    """
    # 目录准备
    dataset_root = os.path.join(save_dir, f"task_{task_id}_pseudo_labels")
    os.makedirs(dataset_root, exist_ok=True)

    # 1) 训练集：分别用两个教师模型生成伪标签 -> 做ID映射 -> 合并 -> 保存
    train_source = os.path.join(yaml_data['path'], yaml_data['train']) if 'path' in yaml_data.keys() else \
        os.path.join(os.path.dirname(data_path), yaml_data['train'])

    # 1.1 运行推理（流式），分别保存到 train_old 与 train_new 目录
    results_old = teacher_model_old.predict(
        train_source, conf=conf_threshold, save_txt=True, save_conf=False, stream=True,
        project=save_dir, name=f"task_{task_id}_pseudo_labels/train_old", verbose=False
    )
    for _ in tqdm(results_old, desc="Generating pseudo labels (old teacher) for train",
                  total=len(os.listdir(train_source)), position=0, leave=True, ncols=80):
        pass

    results_new = teacher_model_new.predict(
        train_source, conf=conf_threshold, save_txt=True, save_conf=False, stream=True,
        project=save_dir, name=f"task_{task_id}_pseudo_labels/train_new", verbose=False
    )
    for _ in tqdm(results_new, desc="Generating pseudo labels (new teacher) for train",
                  total=len(os.listdir(train_source)), position=0, leave=True, ncols=80):
        pass

    # 1.2 复制训练集图像
    copy_images_and_labels(dataset_root, yaml_data, data_path, 'train')

    # 1.3 读取伪标签并做ID映射
    pseudo_labels_dir_old = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/train_old/labels")
    pseudo_labels_dir_new = os.path.join(save_dir, f"task_{task_id}_pseudo_labels/train_new/labels")
    pseudo_labels_old = read_and_convert_labels(pseudo_labels_dir_old, encountered_classes_id_to_new_id)
    pseudo_labels_new = read_and_convert_labels(pseudo_labels_dir_new, task_classes_id_to_new_id)

    # 1.4 合并两个教师的伪标签并保存为训练标签（不使用GT训练标签）
    merged_train_labels = merge_labels(pseudo_labels_old, pseudo_labels_new)
    train_labels_out = os.path.join(dataset_root, "labels/train")
    if os.path.exists(train_labels_out):
        shutil.rmtree(train_labels_out)
    os.makedirs(train_labels_out, exist_ok=True)
    save_labels(merged_train_labels, train_labels_out)

    # 1.5 清理临时伪标签目录
    if os.path.exists(os.path.join(dataset_root, "train_old")):
        shutil.rmtree(os.path.join(dataset_root, "train_old"))
    if os.path.exists(os.path.join(dataset_root, "train_new")):
        shutil.rmtree(os.path.join(dataset_root, "train_new"))

    # 2) 验证集：仅使用原始GT标签做ID映射 -> 保存
    # 2.1 复制验证集图像
    copy_images_and_labels(dataset_root, yaml_data, data_path, 'val')

    # 2.2 读取并转换原始验证集标签
    original_val_labels_dir = os.path.join(yaml_data['path'], yaml_data['val'].replace('images', 'labels')) if 'path' in yaml_data.keys() else \
        os.path.join(os.path.dirname(data_path), yaml_data['val'].replace('images', 'labels'))
    converted_val_labels = read_and_convert_labels(original_val_labels_dir, task_classes_id_to_new_id)

    val_labels_out = os.path.join(dataset_root, "labels/val")
    if os.path.exists(val_labels_out):
        shutil.rmtree(val_labels_out)
    os.makedirs(val_labels_out, exist_ok=True)
    save_labels(converted_val_labels, val_labels_out)


# ------------------增量学习相关函数------------------
def transfer_weights(ckpt_path, model_cfg, weight_transfer_map, classes_names, save_dir, output_name="model_transferred.pt"):
    weight = YOLO(ckpt_path).model.state_dict()
    model_cfg = yaml_model_load(model_cfg)
    model_cfg['nc'] = len(classes_names)
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

    model.model.load_state_dict(new_weight)
    model.model.names = {k: v for k, v in enumerate(classes_names)}
    model.save(os.path.join(save_dir, output_name))