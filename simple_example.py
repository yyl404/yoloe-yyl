#!/usr/bin/env python3
"""
简单示例：如何使用管道通信的YOLO训练脚本
"""

import multiprocessing as mp
import os
import sys
from pathlib import Path

def simple_training_example():
    """简单训练示例"""
    
    # 检查必要文件
    if not os.path.exists('train_model.py'):
        print("❌ 请确保 train_model.py 文件存在")
        return None
    
    # 准备训练参数
    training_args = {
        'data': 'coco.yaml',  # 数据集配置
        'epochs': 5,          # 训练轮数（演示用较少轮数）
        'batch': 16,
        'workers': 2,
        'val_interval': 1,
        'project': 'runs/train',
        'name': 'simple_example',
        'save_period': -1,
        'model_path': 'yolov8n.pt',  # 预训练模型
        'checkpoint': None
    }
    
    print("🚀 启动YOLO训练...")
    print(f"📊 训练参数: {training_args}")
    
    # 创建管道
    parent_conn, child_conn = mp.Pipe()
    
    # 导入训练函数
    from train_model import train_worker
    
    # 启动训练进程
    p = mp.Process(target=train_worker, args=(child_conn, training_args))
    p.start()
    
    # 等待训练完成
    try:
        result = parent_conn.recv()
        
        if result['status'] == 'success':
            print(f"\n✅ 训练成功完成！")
            print(f"📁 模型路径: {result['model_path']}")
            print(f"📊 最佳mAP50: {result['metrics']['best_map']:.4f}")
            print(f"📈 最佳mAP50-95: {result['metrics']['best_map50_95']:.4f}")
            print(f"⏱️  训练时间: {result['metrics']['total_time']:.2f}秒")
            
            # 验证模型文件
            if os.path.exists(result['model_path']):
                print(f"✅ 模型文件验证成功")
                return result['model_path']
            else:
                print(f"❌ 模型文件不存在")
                
        else:
            print(f"\n❌ 训练失败")
            print(f"错误: {result['error_message']}")
            
    except Exception as e:
        print(f"❌ 接收结果时出错: {e}")
    
    finally:
        p.join()
        parent_conn.close()
    
    return None

def load_and_test_model(model_path):
    """加载并测试训练好的模型"""
    if not model_path or not os.path.exists(model_path):
        print("❌ 模型文件不存在，无法测试")
        return
    
    try:
        from ultralytics import YOLO
        
        print(f"\n🔍 加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 获取模型信息
        print(f"📋 模型信息:")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   模型路径: {model.ckpt_path}")
        
        # 如果有测试图像，可以进行推理测试
        test_image = "test.jpg"  # 替换为实际的测试图像路径
        if os.path.exists(test_image):
            print(f"\n🔍 在测试图像上进行推理: {test_image}")
            results = model(test_image)
            
            # 显示结果
            for r in results:
                print(f"   检测到 {len(r.boxes)} 个目标")
                if len(r.boxes) > 0:
                    print(f"   置信度范围: {r.boxes.conf.min():.3f} - {r.boxes.conf.max():.3f}")
        else:
            print(f"⚠️  测试图像不存在: {test_image}")
            print("   您可以手动测试模型:")
            print(f"   model = YOLO('{model_path}')")
            print(f"   results = model('your_image.jpg')")
            
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")

def main():
    """主函数"""
    print("🎯 YOLO训练管道通信示例")
    print("=" * 40)
    
    # 检查环境
    try:
        import ultralytics
        print(f"✅ ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        print("❌ 请先安装ultralytics: pip install ultralytics")
        return
    
    # 检查数据集配置
    if not os.path.exists('coco.yaml'):
        print("⚠️  coco.yaml不存在，请确保数据集配置文件存在")
        print("   或者修改training_args中的'data'参数")
    
    # 检查预训练模型
    if not os.path.exists('yolov8n.pt'):
        print("⚠️  yolov8n.pt不存在，ultralytics会自动下载")
    
    # 开始训练
    model_path = simple_training_example()
    
    # 测试模型
    if model_path:
        load_and_test_model(model_path)
    
    print("\n" + "=" * 40)
    print("🎉 示例完成")

if __name__ == "__main__":
    main()
