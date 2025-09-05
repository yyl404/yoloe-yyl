import argparse
import multiprocessing as mp
import os
from ultralytics import YOLO

def train_worker(conn, args_dict):
    """训练工作进程"""
    try:
        # 解析参数
        data = args_dict['data']
        epochs = args_dict['epochs']
        batch = args_dict['batch']
        workers = args_dict['workers']
        val_interval = args_dict['val_interval']
        project = args_dict['project']
        name = args_dict['name']
        save_period = args_dict['save_period']
        model_path = args_dict['model_path']
        checkpoint = args_dict['checkpoint']
        
        # 开始训练
        if checkpoint:  # 如果checkpoint存在，则加载checkpoint训练
            model = YOLO(checkpoint)
            results = model.train(data=data, epochs=epochs, batch=batch, workers=workers, 
                        resume=True, project=project, name=name, val_interval=val_interval, 
                        save_period=save_period)
        else:
            model = YOLO(model_path)
            results = model.train(data=data, epochs=epochs, batch=batch, workers=workers, 
                        project=project, name=name, val_interval=val_interval, save_period=save_period)
        
        # 获取训练结果和模型路径
        best_model_path = results.save_dir + '/weights/best.pt'
        last_model_path = results.save_dir + '/weights/last.pt'
        
        # 检查模型文件是否存在
        if os.path.exists(best_model_path):
            final_model_path = best_model_path
        elif os.path.exists(last_model_path):
            final_model_path = last_model_path
        else:
            raise FileNotFoundError("训练完成但未找到模型文件")
        
        # 发送训练结果给父进程
        result_data = {
            'status': 'success',
            'model_path': final_model_path,
            'save_dir': results.save_dir,
            'metrics': {
                'best_map': results.results_dict.get('metrics/mAP50(B)', 0),
                'best_map50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'epochs_trained': results.epoch,
                'total_time': results.t
            }
        }
        
        conn.send(result_data)
        print(f"训练完成，模型已保存到: {final_model_path}")
        
    except Exception as e:
        # 发送错误信息给父进程
        error_data = {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }
        conn.send(error_data)
        print(f"训练过程中出现错误: {e}")
    
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--use_pipe", action="store_true", help="使用管道与父进程通信")
    args = parser.parse_args()

    if args.use_pipe:
        # 使用管道模式
        parent_conn, child_conn = mp.Pipe()
        
        # 准备参数字典
        args_dict = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'workers': args.workers,
            'val_interval': args.val_interval,
            'project': args.project,
            'name': args.name,
            'save_period': args.save_period,
            'model_path': args.model_path,
            'checkpoint': args.checkpoint
        }
        
        # 启动训练进程
        print("启动训练进程...")
        p = mp.Process(target=train_worker, args=(child_conn, args_dict))
        p.start()
        
        # 等待训练完成并接收结果
        try:
            result = parent_conn.recv()
            
            if result['status'] == 'success':
                print(f"\n=== 训练成功完成 ===")
                print(f"模型路径: {result['model_path']}")
                print(f"保存目录: {result['save_dir']}")
                print(f"最佳mAP50: {result['metrics']['best_map']:.4f}")
                print(f"最佳mAP50-95: {result['metrics']['best_map50_95']:.4f}")
                print(f"训练轮数: {result['metrics']['epochs_trained']}")
                print(f"总训练时间: {result['metrics']['total_time']:.2f}秒")
                
                # 这里可以添加模型加载和验证代码
                # model = YOLO(result['model_path'])
                # print("模型加载成功，可以进行推理")
                
            else:
                print(f"\n=== 训练失败 ===")
                print(f"错误类型: {result['error_type']}")
                print(f"错误信息: {result['error_message']}")
                
        except Exception as e:
            print(f"接收训练结果时出错: {e}")
        
        finally:
            # 等待进程结束并清理
            p.join()
            parent_conn.close()
            
    else:
        # 原有的直接训练模式
        if args.checkpoint: # 如果checkpoint存在，则加载checkpoint训练
            model = YOLO(args.checkpoint)
            model.train(data=args.data, epochs=args.epochs, batch=args.batch, workers=args.workers, 
                        resume=True, project=args.project, name=args.name, val_interval=args.val_interval, 
                        save_period=args.save_period)
        else:
            model = YOLO(args.model_path)
            model.train(data=args.data, epochs=args.epochs, batch=args.batch, workers=args.workers, 
                        project=args.project, name=args.name, val_interval=args.val_interval, save_period=args.save_period)

if __name__ == "__main__":
    main()
