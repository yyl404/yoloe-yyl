python train_incremental_old.py --data /hy-tmp/datasets/VOC_inc_15_5/incremental_config.yaml \
       --model_name yolov8l-wtconv --model_cfg yolov8l-wtconv.yaml --epochs 100 --batch 16 --device 0 \
       --save_dir /hy-tmp/runs-yoloe-yyl/yolov8l_voc_inc_15_5_fromscratch_OSR/ \
       --method OSR --save_period 10

python train_incremental_old.py --data /hy-tmp/datasets/VOC_inc_15_5/incremental_config.yaml \
       --model_name yolov8l-wtconv --model_cfg yolov8l-wtconv.yaml --epochs 100 --batch 16 --device 0 \
       --save_dir /hy-tmp/runs-yoloe-yyl/yolov8l_voc_inc_15_5_fromscratch_naive/ \
       --method naive --save_period 10