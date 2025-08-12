import argparse
import os

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model_save_path", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint: # 如果checkpoint存在，则加载checkpoint训练
        model = YOLO(args.checkpoint)
        model.train(data=args.data, epochs=args.epochs, batch=args.batch, workers=args.workers,
                    resume=True, project=args.save_dir, name=f"task{args.task_id}", val_interval=1)
    else:
        model = YOLO(args.model_path)
        model.train(data=args.data, epochs=args.epochs, batch=args.batch, workers=args.workers,
                    project=args.save_dir, name=f"task{args.task_id}", val_interval=1)
    
    if args.model_save_path is not None:
        model.save(args.model_save_path)
    else:
        model.save(os.path.join(args.save_dir, f"{args.model_name}-task{args.task_id}.pt"))
    

if __name__ == "__main__":
    main()
