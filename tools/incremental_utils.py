import os
import shutil
import random
import cv2
import numpy as np
import json
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

#------------------数据增强相关函数------------------
def calculate_iou(box1, box2):
    """计算两个边界框的IoU
    
    Args:
        box1: [x_center, y_center, width, height] (归一化坐标)
        box2: [x_center, y_center, width, height] (归一化坐标)
    
    Returns:
        float: IoU值
    """
    # 转换为左上角和右下角坐标
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def parse_paste_sample_filename(filename):
    """解析粘贴样本文件名，提取类别名和索引
    
    Args:
        filename: 文件名，格式为 {class_name}_best_sample_{index}.jpg
    
    Returns:
        tuple: (class_name, index)
    """
    if not filename.endswith('.jpg'):
        return None, None
    
    parts = filename.replace('.jpg', '').split('_best_sample_')
    if len(parts) != 2:
        return None, None
    
    class_name = parts[0]
    try:
        index = int(parts[1])
        return class_name, index
    except ValueError:
        return None, None


def get_paste_samples_by_class(paste_sample_dir, class_names):
    """获取按类别分组的粘贴样本
    
    Args:
        paste_sample_dir: 粘贴样本目录
        class_names: 类别名称列表
    
    Returns:
        dict: {class_name: [sample_files]}
    """
    paste_samples = {}
    for class_name in class_names:
        paste_samples[class_name] = []
    
    if not os.path.exists(paste_sample_dir):
        return paste_samples
    
    for filename in os.listdir(paste_sample_dir):
        class_name, index = parse_paste_sample_filename(filename)
        if class_name in class_names:
            paste_samples[class_name].append(filename)
    
    return paste_samples


def paste_sample_on_image(base_image, paste_image, paste_x, paste_y, paste_width, paste_height):
    """将粘贴样本粘贴到基础图像上
    
    Args:
        base_image: 基础图像 (numpy array)
        paste_image: 粘贴样本图像 (numpy array)
        paste_x, paste_y: 粘贴位置 (像素坐标)
        paste_width, paste_height: 粘贴尺寸 (像素)
    
    Returns:
        numpy array: 粘贴后的图像
    """
    # 调整粘贴样本尺寸
    paste_image_resized = cv2.resize(paste_image, (paste_width, paste_height))
    
    # 获取基础图像尺寸
    h, w = base_image.shape[:2]
    
    # 确保粘贴位置在图像范围内
    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(w, paste_x + paste_width)
    y2 = min(h, paste_y + paste_height)
    
    if x2 <= x1 or y2 <= y1:
        return base_image
    
    # 计算实际粘贴区域
    paste_x_offset = max(0, -paste_x)
    paste_y_offset = max(0, -paste_y)
    paste_w_actual = x2 - x1
    paste_h_actual = y2 - y1
    
    # 粘贴样本
    base_image[y1:y2, x1:x2] = paste_image_resized[paste_y_offset:paste_y_offset+paste_h_actual, 
                                                   paste_x_offset:paste_x_offset+paste_w_actual]
    
    return base_image


def copy_paste_augmentation(source_dataset, paste_sample_dir, save_dir, split):
    """复制粘贴增强
    
    Args:
        source_dataset: 源数据集（yaml配置文件）
        paste_sample_dir: 粘贴样本目录
        save_dir: 保存目录
        split: 数据集分割（train/val）

    source_dataset是任务数据集的yaml配置文件，paste_sample_dir是记忆库当中的样本目录，
    复制粘贴增强的实现方式是：
    1. 遍历source_dataset当中的指定split的样本，对于每个样本：
        1.1 随机生成一个[0, 3]的整数，作为粘贴样本的数量
        1.2 随机抽取相应数量的paste_sample_dir当中的样本
        1.3 将抽取的样本粘贴到数据集样本的随机位置，如果粘贴样本与数据集样本的实例有重叠（IoU>0.5），则删除数据集样本中对应实例的标注
        1.4 为增强后的样本生成新的标签，将粘贴样本的类别和边界框添加到数据集样本的标签当中
        1.5 将增强后的样本保存到f'{save_dir}/images/{split}'目录下，新标签保存到f'{save_dir}/labels/{split}'目录下
    2. 生成新的yaml配置文件，保存到f'{save_dir}/dataset.yaml'目录下，可以直接复制source_dataset当中的dataset.yaml文件内容

    声明：
    1. paste_sample_dir当中的样本文件命名是:{class_name}_best_sample_{index}.jpg
    2. 所有paste_sample_dir当中的样本类别都能够在source_dataset当中的names字段中找到
    """
    # 读取源数据集配置
    yaml_data = yaml_load(source_dataset)
    class_names = list(yaml_data['names'].values())
    class_name_to_id = {name: idx for idx, name in yaml_data['names'].items()}
    
    # 获取粘贴样本
    paste_samples = get_paste_samples_by_class(paste_sample_dir, class_names)
    
    # 创建保存目录
    images_save_dir = os.path.join(save_dir, 'images', split)
    labels_save_dir = os.path.join(save_dir, 'labels', split)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)
    
    # 获取源数据集图像和标签路径
    source_images_dir = os.path.join(os.path.dirname(source_dataset), yaml_data[split])
    source_labels_dir = os.path.join(os.path.dirname(source_dataset), 'labels', split)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(image_files)} images for {split} split...")
    
    for image_file in tqdm(image_files, desc=f"Copy-paste augmentation for {split}"):
        # 读取基础图像
        image_path = os.path.join(source_images_dir, image_file)
        base_image = cv2.imread(image_path)
        if base_image is None:
            continue
        
        h, w = base_image.shape[:2]
        
        # 读取基础标签
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(source_labels_dir, label_file)
        base_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                base_labels = [line.strip().split() for line in f.readlines()]
        
        # 随机决定粘贴样本数量 (0-3)
        num_paste_samples = random.randint(0, 3)
        
        # 复制基础图像和标签
        augmented_image = base_image.copy()
        augmented_labels = base_labels.copy()
        
        # 记录需要删除的原始标注索引
        labels_to_remove = set()
        
        # 记录已粘贴样本的边界框，用于检查粘贴样本之间的重叠
        pasted_boxes = []
        
        for _ in range(num_paste_samples):
            # 随机选择一个类别
            available_classes = [cls for cls in class_names if paste_samples[cls]]
            if not available_classes:
                break
            
            selected_class = random.choice(available_classes)
            selected_sample = random.choice(paste_samples[selected_class])
            
            # 读取粘贴样本图像
            paste_image_path = os.path.join(paste_sample_dir, selected_sample)
            paste_image = cv2.imread(paste_image_path)
            if paste_image is None:
                continue
            
            # 随机生成粘贴位置和尺寸
            paste_h, paste_w = paste_image.shape[:2]
            max_paste_w = min(w // 3, paste_w)
            max_paste_h = min(h // 3, paste_h)
            
            paste_width = random.randint(max_paste_w // 2, max_paste_w)
            paste_height = random.randint(max_paste_h // 2, max_paste_h)
            
            # 尝试找到不重叠的位置
            max_attempts = 50  # 最大尝试次数
            valid_position_found = False
            
            for attempt in range(max_attempts):
                paste_x = random.randint(0, w - paste_width)
                paste_y = random.randint(0, h - paste_height)
                
                # 计算粘贴样本的归一化边界框
                paste_center_x = (paste_x + paste_width / 2) / w
                paste_center_y = (paste_y + paste_height / 2) / h
                paste_norm_width = paste_width / w
                paste_norm_height = paste_height / h
                
                paste_box = [paste_center_x, paste_center_y, paste_norm_width, paste_norm_height]
                
                # 检查与现有标注的重叠
                overlap_with_existing = False
                for i, label in enumerate(augmented_labels):
                    if len(label) >= 5:
                        existing_box = [float(label[1]), float(label[2]), float(label[3]), float(label[4])]
                        iou = calculate_iou(paste_box, existing_box)
                        if iou > 0.5:
                            labels_to_remove.add(i)
                            overlap_with_existing = True
                
                # 检查与已粘贴样本的重叠
                overlap_with_pasted = False
                for pasted_box in pasted_boxes:
                    iou = calculate_iou(paste_box, pasted_box)
                    if iou > 1e-3:
                        overlap_with_pasted = True
                        break
                
                # 如果位置合适，跳出循环
                if not overlap_with_pasted:
                    valid_position_found = True
                    break
            
            # 如果没有找到合适的位置，跳过这个样本
            if not valid_position_found:
                continue
            
            # 粘贴样本到图像
            augmented_image = paste_sample_on_image(augmented_image, paste_image, 
                                                   paste_x, paste_y, paste_width, paste_height)
            
            # 添加粘贴样本的标注
            class_id = class_name_to_id[selected_class]
            augmented_labels.append([str(class_id), str(paste_center_x), str(paste_center_y), 
                                   str(paste_norm_width), str(paste_norm_height)])
            
            # 记录已粘贴的边界框
            pasted_boxes.append(paste_box)
        
        # 删除重叠的原始标注
        final_labels = []
        for i, label in enumerate(augmented_labels):
            if i not in labels_to_remove:
                final_labels.append(label)
        
        # 保存增强后的图像
        save_image_path = os.path.join(images_save_dir, image_file)
        cv2.imwrite(save_image_path, augmented_image)
        
        # 保存增强后的标签
        save_label_path = os.path.join(labels_save_dir, label_file)
        with open(save_label_path, 'w') as f:
            for label in final_labels:
                f.write(' '.join(label) + '\n')
    
    # 复制并修改yaml配置文件
    new_yaml_path = os.path.join(save_dir, 'dataset.yaml')
    yaml_save(new_yaml_path, yaml_data)
    
    print(f"Copy-paste augmentation completed. Results saved to {save_dir}")
    return new_yaml_path


def crop_instance_from_image(image, bbox, class_name, index):
    """从图像中裁剪出指定边界框的实例
    
    Args:
        image: 输入图像
        bbox: 边界框 [x_center, y_center, width, height] (归一化坐标)
        class_name: 类别名称
        index: 索引
    
    Returns:
        tuple: (cropped_image, filename)
    """
    h, w = image.shape[:2]
    
    # 转换为像素坐标
    x_center = int(bbox[0] * w)
    y_center = int(bbox[1] * h)
    bbox_width = int(bbox[2] * w)
    bbox_height = int(bbox[3] * h)
    
    # 计算边界框的左上角坐标
    x1 = max(0, x_center - bbox_width // 2)
    y1 = max(0, y_center - bbox_height // 2)
    x2 = min(w, x_center + bbox_width // 2)
    y2 = min(h, y_center + bbox_height // 2)
    
    # 裁剪图像
    cropped = image[y1:y2, x1:x2]
    
    # 生成文件名
    filename = f"{class_name}_sample_{index}.jpg"
    
    return cropped, filename


def mix_up_augmentation(source_dataset, cropped_sample_dir, save_dir, split, num_generations):
    """mix-up增强
    
    Args:
        source_dataset: 源数据集（yaml配置文件）
        cropped_sample_dir: 裁剪样本目录
        save_dir: 保存目录
        split: 数据集分割（train/val）
        num_generations: 生成的新样本数量

    source_dataset是任务数据集的yaml配置文件，cropped_sample_dir是记忆库当中的样本目录，
    mix-up增强的实现方式是：
    1. 读取source_dataset当中的指定split的图像，对于每个图像,
       裁剪下它当中包含的全部实例，按照命名格式：{class_name}_sample_{index}.jpg，
       保存至save_dir/cropped_samples_source目录下
    2. 遍历save_dir/cropped_samples_source当中的样本，对于每个样本：
        2.1 随机选择一个cropped_sample_dir当中的样本
        2.2 将两个样本变换到相同的图像尺度，并以\lambda和(1-\lambda)的权重进行像素相加，生成新的样本
        2.3 将新的样本保存到f'{save_dir}/mixed_up_samples目录下，命名格式为: {class_name_1}_{class_name_2}_mixed_up_sample_{index}.jpg
    3. 生成num_generations个新的样本，对于每个样本：
        3.1 首先以噪声图作为基础图像，图像尺度为(640, 640)
        3.2 生成[1, 4]之间的随机数，记为k
        3.3 从save_dir/mixed_up_samples目录下随机选择k个样本
        3.4 将这k个样本粘贴到基础图像上，确保粘贴的过程中不会互相遮挡（IoU>0.5）
        3.5 将生成的新图像保存至save_dir目录下，命名格式为: sample_{index}.jpg
    4. 删除中间裁剪得到的图像和混合得到的图像，只保留最终的样本

    声明：
    1. \lambda的取值服从beta分布，参数为(1, 1)
    2. 所有cropped_sample_dir当中的样本文件命名是:{class_name}_best_sample_{index}.jpg
    3. 不需要生成标签文件和yaml配置文件，只需要保存最终的样本到save_dir目录下
    """
    # 读取源数据集配置
    yaml_data = yaml_load(source_dataset)
    class_names = list(yaml_data['names'].values())
    class_name_to_id = {name: idx for idx, name in yaml_data['names'].items()}
    
    # 创建保存目录
    cropped_source_dir = os.path.join(save_dir, 'cropped_samples_source')
    mixed_up_dir = os.path.join(save_dir, 'mixed_up_samples')
    final_samples_dir = os.path.join(save_dir)
    
    os.makedirs(cropped_source_dir, exist_ok=True)
    os.makedirs(mixed_up_dir, exist_ok=True)
    os.makedirs(final_samples_dir, exist_ok=True)
    
    # 获取源数据集图像和标签路径
    source_images_dir = os.path.join(os.path.dirname(source_dataset), yaml_data[split])
    source_labels_dir = os.path.join(os.path.dirname(source_dataset), 'labels', split)
    
    print(f"Step 1: Cropping instances from source dataset...")
    
    # 步骤1: 裁剪源数据集中的实例
    cropped_count = 0
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in tqdm(image_files, desc="Cropping instances"):
        # 读取图像
        image_path = os.path.join(source_images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # 读取标签
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(source_labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]
            
            # 裁剪每个实例
            for i, label in enumerate(labels):
                if len(label) >= 5:
                    class_id = int(label[0])
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                        bbox = [float(label[1]), float(label[2]), float(label[3]), float(label[4])]
                        
                        cropped_image, filename = crop_instance_from_image(image, bbox, class_name, cropped_count)
                        
                        if cropped_image.size > 0:  # 确保裁剪的图像不为空
                            save_path = os.path.join(cropped_source_dir, filename)
                            cv2.imwrite(save_path, cropped_image)
                            cropped_count += 1
    
    print(f"Cropped {cropped_count} instances from source dataset")
    
    print(f"Step 2: Creating mixed-up samples...")
    
    # 步骤2: 创建mix-up样本
    mixed_up_count = 0
    cropped_source_files = [f for f in os.listdir(cropped_source_dir) if f.endswith('.jpg')]
    
    # 获取记忆库样本
    memory_samples = {}
    if os.path.exists(cropped_sample_dir):
        for filename in os.listdir(cropped_sample_dir):
            if filename.endswith('.jpg'):
                class_name, index = parse_paste_sample_filename(filename)
                if class_name in class_names:
                    if class_name not in memory_samples:
                        memory_samples[class_name] = []
                    memory_samples[class_name].append(filename)
    
    for source_file in tqdm(cropped_source_files, desc="Creating mixed-up samples"):
        # 解析源文件名
        parts = source_file.replace('.jpg', '').split('_sample_')
        if len(parts) != 2:
            continue
        
        source_class = parts[0]
        
        # 随机选择记忆库中的样本
        available_classes = list(memory_samples.keys())
        if not available_classes:
            continue
        
        memory_class = random.choice(available_classes)
        memory_file = random.choice(memory_samples[memory_class])
        
        # 读取源样本
        source_path = os.path.join(cropped_source_dir, source_file)
        source_image = cv2.imread(source_path)
        
        # 读取记忆库样本
        memory_path = os.path.join(cropped_sample_dir, memory_file)
        memory_image = cv2.imread(memory_path)
        
        if source_image is None or memory_image is None:
            continue
        
        # 调整到相同尺寸 (使用较小的尺寸)
        target_size = (min(source_image.shape[1], memory_image.shape[1]), 
                      min(source_image.shape[0], memory_image.shape[0]))
        
        source_resized = cv2.resize(source_image, target_size)
        memory_resized = cv2.resize(memory_image, target_size)
        
        # 生成lambda值 (beta分布，参数为(1,1))
        lambda_val = np.random.beta(1, 1)
        
        # 进行mix-up
        mixed_image = cv2.addWeighted(source_resized, lambda_val, memory_resized, 1 - lambda_val, 0)
        
        # 保存mix-up样本
        mixed_filename = f"{source_class}_{memory_class}_mixed_up_sample_{mixed_up_count}.jpg"
        mixed_path = os.path.join(mixed_up_dir, mixed_filename)
        cv2.imwrite(mixed_path, mixed_image)
        mixed_up_count += 1
    
    print(f"Created {mixed_up_count} mixed-up samples")
    
    print(f"Step 3: Generating final samples...")
    
    # 步骤3: 生成最终样本
    mixed_up_files = [f for f in os.listdir(mixed_up_dir) if f.endswith('.jpg')]
    generation_log = {}
    
    for i in tqdm(range(num_generations), desc="Generating final samples"):
        # 创建噪声基础图像 (640x640)
        base_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 随机选择k个mix-up样本 (1-4个)
        k = random.randint(1, 4)
        selected_samples = random.sample(mixed_up_files, min(k, len(mixed_up_files)))
        
        # 记录使用的样本
        generation_log[f"sample_{i}.jpg"] = selected_samples
        
        # 粘贴选中的样本到基础图像上
        pasted_boxes = []
        
        for sample_file in selected_samples:
            # 读取样本图像
            sample_path = os.path.join(mixed_up_dir, sample_file)
            sample_image = cv2.imread(sample_path)
            
            if sample_image is None:
                continue
            
            # 随机生成粘贴位置和尺寸
            sample_h, sample_w = sample_image.shape[:2]
            
            paste_width = random.randint(int(sample_w * 0.75), sample_w)
            paste_height = random.randint(int(sample_h * 0.75), sample_h)
            
            # 尝试找到不重叠的位置
            max_attempts = 50
            valid_position_found = False
            
            for attempt in range(max_attempts):
                paste_x = random.randint(0, 640 - paste_width)
                paste_y = random.randint(0, 640 - paste_height)
                
                # 计算粘贴样本的归一化边界框
                paste_center_x = (paste_x + paste_width / 2) / 640
                paste_center_y = (paste_y + paste_height / 2) / 640
                paste_norm_width = paste_width / 640
                paste_norm_height = paste_height / 640
                
                paste_box = [paste_center_x, paste_center_y, paste_norm_width, paste_norm_height]
                
                # 检查与已粘贴样本的重叠
                overlap = False
                for pasted_box in pasted_boxes:
                    iou = calculate_iou(paste_box, pasted_box)
                    if iou > 1e-3:
                        overlap = True
                        break
                
                if not overlap:
                    valid_position_found = True
                    break
            
            if valid_position_found:
                # 粘贴样本到图像
                sample_resized = cv2.resize(sample_image, (paste_width, paste_height))
                base_image[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = sample_resized
                pasted_boxes.append(paste_box)
        
        # 保存最终样本
        final_filename = f"sample_{i}.jpg"
        final_path = os.path.join(final_samples_dir, final_filename)
        cv2.imwrite(final_path, base_image)
    
    # 步骤4: 删除中间文件
    shutil.rmtree(cropped_source_dir)
    shutil.rmtree(mixed_up_dir)
    
    return save_dir