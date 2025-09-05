import argparse
import torch
import os
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="results.pt")
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    
    model = YOLO(args.model_path)
    results = model.val(data=args.data, project=args.project, name="val", device=args.device, verbose=False, workers=2)
    
    # 从results中提取mAP50
    # 提取的结果是字典：{class_name_1: AP50_1, class_name_2: AP50_2, ...}
    # 和一个浮点数：mAP50
    
    # 提取mAP50值
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)
    
    # 提取每个类别的AP50值
    class_ap50_dict = {}
    if hasattr(results.box, "ap_class_index") and hasattr(results.box, "ap50"):
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = model.model.names[class_idx]
            class_ap50_dict[class_name] = float(results.box.ap50[i])
    
    # 构建最终结果
    final_results = {
        "map50": map50,
        "map50_95": map50_95,
        "class_ap50": class_ap50_dict,
    }
    
    # 保存处理后的结果
    torch.save(final_results, os.path.join(args.project, f"{args.save_name}"))
    