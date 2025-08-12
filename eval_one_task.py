import argparse
import json
import os
import shutil

from ultralytics import YOLO

from tools.incremental_utils import create_id_converted_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_data_path", type=str, required=True)
    parser.add_argument("--task_classes_json", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_task_idx", type=int, required=True)
    parser.add_argument("--eval_task_idx", type=int, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output_json_path", type=str, required=True)
    args = parser.parse_args()

    task_classes = json.loads(args.task_classes_json)

    # 加载模型
    model = YOLO(args.model_path)
    model_classes = list(model.model.names.values())

    # 构建 类别映射：任务类别ID -> 模型类别ID
    class_mapping = {}
    for task_idx, task_class in enumerate(task_classes):
        if task_class in model_classes:
            model_idx = model_classes.index(task_class)
            class_mapping[task_idx] = model_idx

    class_mapping_reverse = {v: k for k, v in class_mapping.items()}

    # 创建评估数据集（带ID转换）
    dataset_name = f"task_{args.eval_task_idx}_eval_for_model_{args.model_task_idx}"
    data_path = create_id_converted_dataset(
        args.task_data_path, class_mapping, args.save_dir, dataset_name, model_classes
    )

    try:
        # 运行验证
        results = model.val(
            data=data_path, device=args.device, verbose=False, project=args.save_dir, batch=1, workers=2
        )

        # 提取指标
        map50 = float(results.box.map50)
        map50_95 = float(results.box.map)

        class_ap50_pairs = []
        class_ap50_95_pairs = []
        if hasattr(results.box, "ap_class_index") and hasattr(results.box, "ap50"):
            for i, class_idx in enumerate(results.box.ap_class_index):
                if class_idx in class_mapping_reverse:
                    task_class_idx = int(class_mapping_reverse[class_idx])
                    class_ap50_pairs.append([task_class_idx, float(results.box.ap50[i])])
                    class_ap50_95_pairs.append([task_class_idx, float(results.box.ap[i])])

        # 输出结果到JSON
        output = {
            "map50": map50,
            "map50_95": map50_95,
            "class_mapping_pairs": [[int(k), int(v)] for k, v in class_mapping.items()],
            "class_ap50_pairs": class_ap50_pairs,
            "class_ap50_95_pairs": class_ap50_95_pairs,
        }
        os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
        with open(args.output_json_path, "w") as f:
            json.dump(output, f)
    finally:
        # 清理评估数据集
        if os.path.exists(os.path.dirname(data_path)):
            shutil.rmtree(os.path.dirname(data_path))


if __name__ == "__main__":
    main()


