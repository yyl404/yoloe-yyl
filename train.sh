python train_baseline.py

python train_incremental.py --data /hy-tmp/VOC_inc_15_1_1_1_1_1_full-labels/incremental_config.yaml \
       --model_name yolov8m --model_cfg yolov8m.yaml --epochs 100 --batch 16 --device 0 \
       --save_dir /hy-tmp/yolov8m_inc_15_1_1_1_1_1_full-labels_fromscratch_naive \
       --method naive

python test_incremental.py --data /hy-tmp/VOC_inc_15_1_1_1_1_1/incremental_config.yaml \
       --save_dir /hy-tmp/yolov8m_inc_15_1_1_1_1_1_full-labels_fromscratch_naive \
       --model_name yolov8m

python train_incremental.py --data /hy-tmp/VOC_inc_15_1_1_1_1_1/incremental_config.yaml \
       --model_name yolov8m --model_cfg yolov8m.yaml --epochs 100 --batch 16 --device 0 \
       --save_dir /hy-tmp/yolov8m_inc_15_1_1_1_1_1_fromscratch_pseudo_labels \
       --method pseudo_labels

python test_incremental.py --data /hy-tmp/VOC_inc_15_1_1_1_1_1/incremental_config.yaml \
       --save_dir /hy-tmp/yolov8m_inc_15_1_1_1_1_1_fromscratch_pseudo_labels \
       --model_name yolov8m

python train_incremental.py --data /hy-tmp/VOC_inc_15_1_1_1_1_1/incremental_config.yaml \
       --model_name yolov8m --model_cfg yolov8m.yaml --epochs 100 --batch 16 --device 0 \
       --save_dir /hy-tmp/yolov8m_inc_15_1_1_1_1_1_fromscratch_dual_teachers \
       --method dual_teachers

python test_incremental.py --data /hy-tmp/VOC_inc_15_1_1_1_1_1/incremental_config.yaml \
       --save_dir /hy-tmp/yolov8m_inc_15_1_1_1_1_1_fromscratch_dual_teachers \
       --model_name yolov8m