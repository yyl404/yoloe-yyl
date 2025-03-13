import argparse
import os
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yoloe-v8l-seg.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image"
    )
    parser.add_argument(
        "--bboxes",
        nargs="+",
        type=float,
        help="Bounding box coordinates as x1 y1 x2 y2 (multiple boxes in sequence)"
    )
    parser.add_argument(
        "--cls",
        nargs="+",
        type=int,
        help="Class IDs corresponding to each bounding box"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Class names corresponding to each class ID"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.source)
        args.output = f"{base}-output{ext}"

    image = Image.open(args.source).convert("RGB")

    model = YOLOE(args.checkpoint)
    model.to(args.device)

    bboxes_array = np.array(args.bboxes, dtype=float).reshape(-1, 4)
    cls_array = np.array(args.cls, dtype=int)
    visuals = dict(bboxes=bboxes_array, cls=cls_array)

    results = model.predict(
        image,
        prompts=visuals,
        predictor=YOLOEVPSegPredictor,
        verbose=False
    )
    model.set_classes(args.names, model.predictor.vpe)
    detections = sv.Detections.from_ultralytics(results[0])

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    annotated_image.save(args.output)
    print(f"Annotated image saved to: {args.output}")

if __name__ == "__main__":
    main()
