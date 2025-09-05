#!/usr/bin/env python3
"""
父进程示例：启动训练进程并通过管道接收训练结果
"""

import multiprocessing as mp
import subprocess
import sys
import os
import time
from pathlib import Path

def start_training_with_pipe():
    """方法1：使用multiprocessing.Pipe直接通信"""
    
    def train_worker(conn, training_args):
        """训练工作进程"""
        try:
            # 导入训练模块
            from train_model import train_worker as yolo_train_worker
            
            # 调用YOLO训练函数
            yolo_train_worker(conn, training_args)
            
        except Exception as e:
            # 发送错误信息
            error_data = {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            conn.send(error_data)
        finally:
            conn.close()
    
    # 准备训练参数
    training_args = {
        'data': 'coco.yaml',  # 数据集配置文件
        'epochs': 10,  # 训练轮数
        'batch': 16,
        'workers': 4,
        'val_interval': 1,
        'project': 'runs/train',
        'name': 'yolo_model',
        'save_period': -1,
        'model_path': 'yolov8n.pt',  # 预训练模型
        'checkpoint': None
    }
    
    print("=== 方法1：使用multiprocessing.Pipe ===")
    print("启动训练进程...")
    
    # 创建管道
    parent_conn, child_conn = mp.Pipe()
    
    # 启动训练进程
    p = mp.Process(target=train_worker, args=(child_conn, training_args))
    p.start()
    
    # 等待并接收结果
    try:
        result = parent_conn.recv()
        
        if result['status'] == 'success':
            print(f"\n✅ 训练成功完成！")
            print(f"📁 模型路径: {result['model_path']}")
            print(f"📊 最佳mAP50: {result['metrics']['best_map']:.4f}")
            print(f"📈 最佳mAP50-95: {result['metrics']['best_map50_95']:.4f}")
            print(f"⏱️  训练时间: {result['metrics']['total_time']:.2f}秒")
            
            # 验证模型文件是否存在
            if os.path.exists(result['model_path']):
                print(f"✅ 模型文件验证成功")
                return result['model_path']
            else:
                print(f"❌ 模型文件不存在: {result['model_path']}")
                
        else:
            print(f"\n❌ 训练失败")
            print(f"错误类型: {result['error_type']}")
            print(f"错误信息: {result['error_message']}")
            
    except Exception as e:
        print(f"❌ 接收结果时出错: {e}")
    
    finally:
        p.join()
        parent_conn.close()
    
    return None

def start_training_with_subprocess():
    """方法2：使用subprocess启动独立进程"""
    
    print("\n=== 方法2：使用subprocess ===")
    print("启动训练进程...")
    
    # 构建命令
    cmd = [
        sys.executable, 'train_model.py',
        '--data', 'coco.yaml',
        '--epochs', '10',
        '--batch', '16',
        '--workers', '4',
        '--project', 'runs/train',
        '--name', 'yolo_model_subprocess',
        '--model_path', 'yolov8n.pt',
        '--use_pipe'  # 启用管道模式
    ]
    
    try:
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 实时显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print("✅ 训练进程成功完成")
            # 这里可以扫描输出目录找到模型文件
            model_dir = Path("runs/train/yolo_model_subprocess/weights")
            if model_dir.exists():
                best_model = model_dir / "best.pt"
                if best_model.exists():
                    print(f"✅ 找到训练好的模型: {best_model}")
                    return str(best_model)
        else:
            print(f"❌ 训练进程失败，返回码: {return_code}")
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"错误信息: {stderr_output}")
                
    except Exception as e:
        print(f"❌ 启动训练进程时出错: {e}")
    
    return None

def monitor_training_progress():
    """方法3：监控训练进度并实时获取中间结果"""
    
    print("\n=== 方法3：监控训练进度 ===")
    
    def training_monitor(conn, training_args):
        """带进度监控的训练进程"""
        try:
            from ultralytics import YOLO
            
            # 创建模型
            model = YOLO(training_args['model_path'])
            
            # 自定义回调函数来监控进度
            class ProgressCallback:
                def __init__(self, conn):
                    self.conn = conn
                    self.epoch_count = 0
                
                def on_train_epoch_end(self, trainer):
                    self.epoch_count += 1
                    # 发送进度信息
                    progress_data = {
                        'type': 'progress',
                        'epoch': self.epoch_count,
                        'total_epochs': trainer.epochs,
                        'metrics': trainer.metrics.copy() if hasattr(trainer, 'metrics') else {}
                    }
                    self.conn.send(progress_data)
                
                def on_train_end(self, trainer):
                    # 发送完成信息
                    completion_data = {
                        'type': 'completion',
                        'final_metrics': trainer.metrics.copy() if hasattr(trainer, 'metrics') else {},
                        'save_dir': trainer.save_dir
                    }
                    self.conn.send(completion_data)
            
            # 创建回调
            callback = ProgressCallback(conn)
            
            # 开始训练
            results = model.train(
                data=training_args['data'],
                epochs=training_args['epochs'],
                batch=training_args['batch'],
                workers=training_args['workers'],
                project=training_args['project'],
                name=training_args['name'],
                callbacks=[callback]
            )
            
            # 发送最终结果
            final_result = {
                'type': 'final_result',
                'model_path': results.save_dir + '/weights/best.pt',
                'save_dir': results.save_dir,
                'metrics': {
                    'best_map': results.results_dict.get('metrics/mAP50(B)', 0),
                    'best_map50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                    'epochs_trained': results.epoch,
                    'total_time': results.t
                }
            }
            conn.send(final_result)
            
        except Exception as e:
            error_data = {
                'type': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            conn.send(error_data)
        finally:
            conn.close()
    
    # 准备训练参数
    training_args = {
        'data': 'coco.yaml',
        'epochs': 5,  # 减少轮数用于演示
        'batch': 16,
        'workers': 4,
        'project': 'runs/train',
        'name': 'yolo_model_monitor',
        'model_path': 'yolov8n.pt'
    }
    
    # 创建管道
    parent_conn, child_conn = mp.Pipe()
    
    # 启动训练进程
    p = mp.Process(target=training_monitor, args=(child_conn, training_args))
    p.start()
    
    # 监控训练进度
    try:
        while True:
            if parent_conn.poll(timeout=1):  # 1秒超时
                data = parent_conn.recv()
                
                if data['type'] == 'progress':
                    print(f"📊 Epoch {data['epoch']}/{data['total_epochs']} 完成")
                    if 'metrics' in data and data['metrics']:
                        print(f"   当前指标: {data['metrics']}")
                
                elif data['type'] == 'completion':
                    print(f"🎉 训练完成！")
                    print(f"   最终指标: {data['final_metrics']}")
                    print(f"   保存目录: {data['save_dir']}")
                
                elif data['type'] == 'final_result':
                    print(f"\n✅ 训练最终结果:")
                    print(f"📁 模型路径: {data['model_path']}")
                    print(f"📊 最佳mAP50: {data['metrics']['best_map']:.4f}")
                    print(f"📈 最佳mAP50-95: {data['metrics']['best_map50_95']:.4f}")
                    break
                
                elif data['type'] == 'error':
                    print(f"❌ 训练错误: {data['error_message']}")
                    break
            else:
                # 超时，继续等待
                pass
                
    except Exception as e:
        print(f"❌ 监控训练时出错: {e}")
    
    finally:
        p.join()
        parent_conn.close()

def main():
    """主函数：演示不同的启动方式"""
    
    print("🚀 YOLO训练父进程示例")
    print("=" * 50)
    
    # 检查必要文件
    if not os.path.exists('train_model.py'):
        print("❌ 找不到 train_model.py 文件")
        return
    
    # 方法1：使用multiprocessing.Pipe
    model_path1 = start_training_with_pipe()
    
    # 方法2：使用subprocess
    model_path2 = start_training_with_subprocess()
    
    # 方法3：监控训练进度
    monitor_training_progress()
    
    print("\n" + "=" * 50)
    print("🎯 所有演示完成")
    
    # 总结结果
    if model_path1:
        print(f"✅ 方法1成功获取模型: {model_path1}")
    if model_path2:
        print(f"✅ 方法2成功获取模型: {model_path2}")

if __name__ == "__main__":
    main()
