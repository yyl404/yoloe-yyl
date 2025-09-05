import os
import sys
import yaml
import subprocess
import argparse
import torch
import json
import shutil

from ultralytics.utils import yaml_load

from tools.incremental_utils import merge_task_classes, transfer_weights, create_id_converted_dataset, \
    copy_paste_augmentation, mix_up_augmentation


def run(cmd, env=None):
    subprocess.run(cmd, check=True, env=env)

def save_checkpoint(save_dir, encountered_classes, merged_classes, model_path,
    encountered_classes_id_to_new_id, task_classes_id_to_new_id, training_task_idx):
    checkpoint_data = { # 设置新的checkpoint数据
        "encountered_classes": encountered_classes,
        "merged_classes": merged_classes,
        "model_path": model_path,
        "encountered_classes_id_to_new_id": encountered_classes_id_to_new_id,
        "task_classes_id_to_new_id": task_classes_id_to_new_id,
        "training_task_idx": training_task_idx
    }
    torch.save(checkpoint_data, os.path.join(save_dir, f"checkpoint.pt")) # 保存checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_name", type=str, default="yolo11m")
    parser.add_argument("--init_model_path", type=str, default=None)
    parser.add_argument("--model_cfg", type=str, default="yolo11m.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--save_dir", type=str, default="runs/incremental_learning")
    parser.add_argument("--checkpoint", type=str, default=None) # 外层增量学习情景调度器保存的checkpoint
    parser.add_argument("--task_checkpoint", type=str, default=None) # 内层单一任务上训练时保存的checkpoint
    parser.add_argument("--method", type=str, default="naive", choices=["naive", "pseudo_labels", "dual_teachers", "OSR"])
    parser.add_argument("--pseudo_labels_conf", type=float, default=0.25)
    parser.add_argument("--save_period", type=int, default=-1)
    args = parser.parse_args()

    data_cfg = args.data
    save_dir = args.save_dir
    model_name = args.model_name
    init_model_path = args.init_model_path
    model_cfg = args.model_cfg
    epochs = args.epochs
    batch = args.batch
    workers = args.workers
    device = args.device
    checkpoint = args.checkpoint
    task_checkpoint = args.task_checkpoint
    method = args.method
    pseudo_labels_conf = args.pseudo_labels_conf
    save_period = args.save_period
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device)
    env["WANDB_DISABLED"] = "true"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    with open(data_cfg, "r") as f:
        inc = yaml.load(f, Loader=yaml.FullLoader)
    task_paths = [os.path.join(os.path.dirname(data_cfg), inc["tasks"][k]) for k in sorted(inc["tasks"].keys())]

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint)
        encountered_classes = checkpoint_data['encountered_classes'] # 旧类别
        merged_classes = checkpoint_data['merged_classes'] # 合并后的类别
        model_path = checkpoint_data['model_path'] # 模型路径
        encountered_classes_id_to_new_id = checkpoint_data['encountered_classes_id_to_new_id'] # 旧类别ID到新ID的映射
        task_classes_id_to_new_id = checkpoint_data['task_classes_id_to_new_id'] # 任务类别ID到新ID的映射
        resume_task_idx = checkpoint_data['training_task_idx'] # 当前训练任务的索引
    else:
        resume_task_idx = 0

    # for i, task_yaml in enumerate(task_paths, start=resume_task_idx):
    for i in range(resume_task_idx, len(task_paths)):
        task_yaml = task_paths[i]
        task_yaml_data = yaml_load(task_yaml)
        task_classes = [task_yaml_data['names'][k] for k in sorted(task_yaml_data['names'].keys())]

        if i == resume_task_idx and checkpoint is not None: # 如果checkpoint存在，且当前任务是resume_task_idx，则跳过类型合并和权重迁移
            pass
        elif i == 0:
            encountered_classes = task_classes
            merged_classes = encountered_classes
            model_path = init_model_path if init_model_path is not None else model_cfg
        else:
            # 始终计算 merged_classes，便于传参；若是 resume 的当前任务，则跳过权重迁移
            merged_classes, encountered_classes_id_to_new_id, task_classes_id_to_new_id = merge_task_classes(encountered_classes, task_classes)
            last_task_model_path = os.path.join(save_dir, f"{model_name}-task{i-1}.pt")
            # 进行分类层通道权重的迁移，以扩充通道，为新类别分配新通道
            transfer_weights(last_task_model_path, model_cfg, encountered_classes_id_to_new_id, merged_classes, save_dir, f"{model_name}-task{i}-transferred.pt")
            model_path = os.path.join(save_dir, f"{model_name}-task{i}-transferred.pt")
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

        if i == 0: # 如果当前任务是第一个任务，则直接训练
            # 更新checkpoint
            save_checkpoint(save_dir, encountered_classes, merged_classes, model_path,
                encountered_classes_id_to_new_id=None, task_classes_id_to_new_id=None, training_task_idx=i)
            cmd = [
                sys.executable,
                f"{script_dir}/train_incremental_naive.py",
                "--data", task_yaml,
                "--save_dir", save_dir,
                "--task_id", str(i),
                "--model_name", model_name,
                "--model_path", model_path,
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--workers", str(workers),
                "--save_period", str(save_period)
            ]
            if task_checkpoint is not None:
                cmd.extend(["--checkpoint", task_checkpoint])
            run(cmd, env)
        elif method == "naive": # 如果方法为naive，先将数据集类别id进行转换，然后训练
            # 更新checkpoint
            save_checkpoint(save_dir, encountered_classes, merged_classes, model_path,
                encountered_classes_id_to_new_id, task_classes_id_to_new_id, i)
            # 将数据集类别id进行转换
            create_id_converted_dataset(task_yaml, task_classes_id_to_new_id, save_dir, f"task_{i}_converted", merged_classes)
            cmd = [
                sys.executable,
                f"{script_dir}/train_incremental_naive.py",
                "--data", os.path.join(save_dir, f"task_{i}_converted/dataconfig.yaml"),
                "--save_dir", save_dir,
                "--task_id", str(i),
                "--model_name", model_name,
                "--model_path", model_path,
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--workers", str(workers),
                "--save_period", str(save_period)
            ]
            if task_checkpoint is not None:
                cmd.extend(["--checkpoint", task_checkpoint])
            run(cmd, env)
            # 删除转换后的数据集
            shutil.rmtree(os.path.join(save_dir, f"task_{i}_converted"))
            # 删除迁移后的模型
            os.remove(os.path.join(save_dir, f"{model_name}-task{i}-transferred.pt"))
        elif method == "pseudo_labels":
            # 更新checkpoint
            save_checkpoint(save_dir, encountered_classes, merged_classes, model_path,
                encountered_classes_id_to_new_id, task_classes_id_to_new_id, i)
            if os.path.exists(os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml")): # 如果伪标签存在，则跳过生成
                pass
            else:
                cmd = [
                    sys.executable,
                    f"{script_dir}/generate_pseudo_labels.py",
                    "--data", task_yaml,
                    "--save_dir", save_dir,
                    "--teacher_model_path", os.path.join(save_dir, f"{model_name}-task{i-1}.pt"),
                    "--pseudo_labels_conf", str(pseudo_labels_conf),
                    "--encountered_classes_id_to_new_id_json", json.dumps(encountered_classes_id_to_new_id),
                    "--task_classes_id_to_new_id_json", json.dumps(task_classes_id_to_new_id),
                    "--merged_classes_json", json.dumps(merged_classes),
                    "--task_id", str(i)
                ]
                run(cmd, env)

            cmd = [
                sys.executable,
                f"{script_dir}/train_incremental_naive.py",
                "--data", os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml"),
                "--save_dir", save_dir,
                "--task_id", str(i),
                "--model_name", model_name,
                "--model_path", model_path,
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--workers", str(workers),
                "--save_period", str(save_period)
            ]
            if task_checkpoint is not None:
                cmd.extend(["--checkpoint", task_checkpoint])
            run(cmd, env)
            # 保留伪标签，方便进行查看
            # shutil.rmtree(os.path.join(save_dir, f"task_{i}_pseudo_labels"))
            # 保留迁移后的模型，方便进行查看
            # os.remove(os.path.join(save_dir, f"{model_name}-task{i}-transferred.pt"))
        elif method == "dual_teachers":
            # 更新checkpoint
            save_checkpoint(save_dir, encountered_classes, merged_classes, model_path,
                encountered_classes_id_to_new_id, task_classes_id_to_new_id, i)
            # 先在新类别上训练一个教师模型
            if os.path.exists(os.path.join(save_dir, f"{model_name}-task{i}-teacher.pt")): # 如果教师模型存在，则跳过训练
                pass
            else:
                cmd = [
                    sys.executable,
                    f"{script_dir}/train_incremental_naive.py",
                    "--data", task_yaml,
                    "--save_dir", save_dir,
                    "--task_id", str(i),
                    "--model_name", model_name,
                    "--model_path", model_path,
                    "--epochs", str(epochs),
                    "--batch", str(batch),
                    "--workers", str(workers),
                    "--save_period", str(save_period),
                    "--model_save_path", os.path.join(save_dir, f"{model_name}-task{i}-teacher.pt")
                ]
                if task_checkpoint is not None:
                    cmd.extend(["--checkpoint", task_checkpoint])
                run(cmd, env)

            # 用两个教师模型共同生成伪标签，如果存在，则跳过生成
            if os.path.exists(os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml")):
                pass
            else:
                cmd = [
                    sys.executable,
                    f"{script_dir}/generate_pseudo_labels_dual_teachers.py",
                    "--data", task_yaml,
                    "--save_dir", save_dir,
                    "--task_id", str(i),
                    "--old_teacher_model_path", os.path.join(save_dir, f"{model_name}-task{i-1}.pt"),
                    "--new_teacher_model_path", os.path.join(save_dir, f"{model_name}-task{i}-teacher.pt"),
                    "--pseudo_labels_conf", str(pseudo_labels_conf),
                    "--encountered_classes_id_to_new_id_json", json.dumps(encountered_classes_id_to_new_id),
                    "--task_classes_id_to_new_id_json", json.dumps(task_classes_id_to_new_id),
                    "--merged_classes_json", json.dumps(merged_classes),
                ]
                run(cmd, env)
            
            cmd = [
                sys.executable,
                f"{script_dir}/train_incremental_naive.py",
                "--data", os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml"),
                "--save_dir", save_dir,
                "--task_id", str(i),
                "--model_name", model_name,
                "--model_path", model_path,
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--workers", str(workers),
                "--save_period", str(save_period)
            ]
            if task_checkpoint is not None:
                cmd.extend(["--checkpoint", task_checkpoint])
            run(cmd, env)
            # 保留伪标签，方便进行查看
            # shutil.rmtree(os.path.join(save_dir, f"task_{i}_pseudo_labels"))
            # 保留教师模型，方便进行查看
            # os.remove(os.path.join(save_dir, f"{model_name}-task{i}-teacher.pt"))
            # 保留迁移后的模型，方便进行查看
            # os.remove(os.path.join(save_dir, f"{model_name}-task{i}-transferred.pt"))
        elif method == "OSR":
            """ One-Shot Replay from paper: One-Shot Replay: Boosting Incremental Object 
                Detection via Retrospecting One Object, https://ojs.aaai.org/index.php/AAAI/article/view/25417
            """
            # 更新checkpoint
            save_checkpoint(save_dir, encountered_classes, merged_classes, model_path,
                            encountered_classes_id_to_new_id, task_classes_id_to_new_id, i)

            # 获取任务数据集的规模，用于计算生成样本的数量
            task_yaml_data = yaml_load(task_yaml)
            task_train_size = len(os.listdir(os.path.join(os.path.dirname(task_yaml), task_yaml_data["train"])))
            
            if os.path.exists(os.path.join(save_dir, f"task_{i}_augmented_dataset/dataset.yaml")):
                pass
            else:
                if os.path.exists(os.path.join(save_dir, f"task_{i-1}_cropped_images")):
                    shutil.rmtree(os.path.join(save_dir, f"task_{i-1}_cropped_images"))
                # 从上一个任务的训练集当中，对每一个任务类别，裁剪下最具有代表性的实例样本，保存到记忆库当中
                last_task_yaml = task_paths[i-1]
                print(f"Generating memory bank for task {i-1}...")
                cmd = [
                    sys.executable,
                    f"{script_dir}/generate_OSR_memory_bank.py",
                    "--data", last_task_yaml,
                    "--save_dir", os.path.join(save_dir, f"task_{i-1}_cropped_images"),
                    "--model_path", os.path.join(save_dir, f"{model_name}-task{i-1}.pt"),
                    "--k", str(1)
                ]
                run(cmd, env)

                # 利用记忆库当中的样本和当前任务的训练集，进行copy-paste增强和mix-up增强，生成新的训练集
                ## 先进行mix-up增强
                print(f"Generating mixed-up samples for task {i}...")
                if os.path.exists(os.path.join(save_dir, f"task_{i}_mixed_up_images")):
                    shutil.rmtree(os.path.join(save_dir, f"task_{i}_mixed_up_images"))
                mix_up_augmentation(task_yaml, os.path.join(save_dir, f"task_{i-1}_cropped_images"), \
                    os.path.join(save_dir, f"task_{i}_mixed_up_images"), "train", int(task_train_size * 0.5))
                
                print(f"Copying task {i} dataset...")
                if os.path.exists(os.path.join(save_dir, f"task_{i}_dataset")):
                    shutil.rmtree(os.path.join(save_dir, f"task_{i}_dataset"))
                shutil.copytree(os.path.dirname(task_yaml), os.path.join(save_dir, f"task_{i}_dataset"))
                for image_file in os.listdir(os.path.join(save_dir, f"task_{i}_mixed_up_images")):
                    shutil.copy(os.path.join(save_dir, f"task_{i}_mixed_up_images", image_file), \
                                os.path.join(save_dir, f"task_{i}_dataset/images/train/"))
                ## 利用上一个任务训练的模型，对当前任务数据集和mix-up增强后的样本增加伪标签
                print(f"Generating pseudo labels for task {i}...")
                if os.path.exists(os.path.join(save_dir, f"task_{i}_pseudo_labels")):
                    shutil.rmtree(os.path.join(save_dir, f"task_{i}_pseudo_labels"))
                cmd = [
                    sys.executable,
                    f"{script_dir}/generate_pseudo_labels.py",
                    "--data", os.path.join(save_dir, f"task_{i}_dataset/{os.path.basename(task_yaml)}"),
                    "--save_dir", save_dir,
                    "--teacher_model_path", os.path.join(save_dir, f"{model_name}-task{i-1}.pt"),
                    "--pseudo_labels_conf", str(pseudo_labels_conf),
                    "--encountered_classes_id_to_new_id_json", json.dumps(encountered_classes_id_to_new_id),
                    "--task_classes_id_to_new_id_json", json.dumps(task_classes_id_to_new_id),
                    "--merged_classes_json", json.dumps(merged_classes),
                    "--task_id", str(i)
                ]
                run(cmd, env)
                ## 删除中间文件
                shutil.rmtree(os.path.join(save_dir, f"task_{i}_mixed_up_images"))
                shutil.rmtree(os.path.join(save_dir, f"task_{i}_dataset"))

                ## 对增加了伪标签以后的数据集，进行copy-paste增强
                print(f"Generating copy-paste augmented samples for task {i}...")
                if os.path.exists(os.path.join(save_dir, f"task_{i}_augmented_dataset")):
                    shutil.rmtree(os.path.join(save_dir, f"task_{i}_augmented_dataset"))
                copy_paste_augmentation(os.path.join(save_dir, f"task_{i}_pseudo_labels/dataconfig.yaml"), \
                    os.path.join(save_dir, f"task_{i-1}_cropped_images"), os.path.join(save_dir, f"task_{i}_augmented_dataset"), "train")
                
                shutil.copytree(os.path.join(save_dir, f"task_{i}_pseudo_labels/images/val"), \
                                os.path.join(save_dir, f"task_{i}_augmented_dataset/images/val"))
                shutil.copytree(os.path.join(save_dir, f"task_{i}_pseudo_labels/labels/val"), \
                                os.path.join(save_dir, f"task_{i}_augmented_dataset/labels/val"))

            # 训练新的模型
            cmd = [
                sys.executable,
                f"{script_dir}/train_incremental_naive.py",
                "--data", os.path.join(save_dir, f"task_{i}_augmented_dataset/dataset.yaml"),
                "--save_dir", save_dir,
                "--task_id", str(i),
                "--model_name", model_name,
                "--model_path", model_path,
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--workers", str(workers),
                "--save_period", str(save_period)
            ]
            if task_checkpoint is not None:
                cmd.extend(["--checkpoint", task_checkpoint])
            run(cmd, env)
            
            # 保留迁移后的模型，方便进行查看
            # os.remove(os.path.join(save_dir, f"{model_name}-task{i}-transferred.pt"))

        encountered_classes = merged_classes

if __name__ == "__main__":
    main()