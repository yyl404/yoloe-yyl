from ultralytics import YOLO

model = YOLO("/root/datasets/yolov8m_voc_baseline/train/weights/last.pt")
model.train(data="/root/datasets/VOC/VOC.yaml", epochs=100, batch=16, project='/hy-tmp/yolov8m_voc_baseline', workers=2, resume=True)