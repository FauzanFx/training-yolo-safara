import os
import yaml
import argparse
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

def make_adaptive_lr_callback(patience=10, factor=0.5, min_lr=1e-7):
    """
    ReduceLROnPlateau for Ultralytics YOLO.
    """
    state = {
        'best_loss': float('inf'),
        'wait': 0,
    }

    def on_train_epoch_end(trainer):
        warmup_epochs = int(getattr(trainer.args, 'warmup_epochs', 3))
        if trainer.epoch < warmup_epochs:
            return

        try:
            tloss = trainer.tloss
            loss = float(tloss.mean()) if hasattr(tloss, 'mean') else float(tloss)
        except Exception:
            loss = float(trainer.loss.mean()) if hasattr(trainer.loss, 'mean') else float(trainer.loss)

        if loss < state['best_loss'] - 1e-4:
            state['best_loss'] = loss
            state['wait'] = 0
        else:
            state['wait'] += 1

        if state['wait'] >= patience:
            current_lrs = [pg.get('initial_lr', pg['lr']) for pg in trainer.optimizer.param_groups]
            if max(current_lrs) <= min_lr:
                print(f"[AdaptiveLR] Already at min_lr={min_lr:.1e}, not reducing further.")
                state['wait'] = 0
                return

            for pg in trainer.optimizer.param_groups:
                old = pg.get('initial_lr', pg['lr'])
                pg['initial_lr'] = max(old * factor, min_lr)
                pg['lr'] = max(pg['lr'] * factor, min_lr)

            if hasattr(trainer, 'scheduler') and hasattr(trainer.scheduler, 'base_lrs'):
                trainer.scheduler.base_lrs = [
                    max(blr * factor, min_lr)
                    for blr in trainer.scheduler.base_lrs
                ]

            new_lr = max(pg['lr'] for pg in trainer.optimizer.param_groups)
            print(
                f"\n[AdaptiveLR] Epoch {trainer.epoch} — loss stuck for {patience} epochs "
                f"(best={state['best_loss']:.4f}) -> LR reduced to {new_lr:.2e}"
            )
            state['wait'] = 0

    return {'on_train_epoch_end': on_train_epoch_end}

def train_yolo(args):
    # 1. Download Dataset from Roboflow
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not roboflow_api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Please set it in your .env file or environment variables."
        )
    
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(args.roboflow_workspace).project(args.roboflow_project)
    version = project.version(args.roboflow_version)
    dataset = version.download("yolov11")

    # 2. Patch data.yaml
    yaml_path = os.path.join(dataset.location, "data.yaml")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use classes from args or default
    config['nc'] = len(args.class_names)
    config['names'] = args.class_names

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[OK] System configured to recognize {config['nc']} custom classes.")

    # 3. Load model
    model = YOLO(args.checkpoint)

    # 4. Add adaptive LR callback
    lr_callback = make_adaptive_lr_callback(patience=args.lr_patience, factor=args.lr_factor, min_lr=args.min_lr)
    model.add_callback('on_train_epoch_end', lr_callback['on_train_epoch_end'])

    # 5. Training
    print(f"Starting training process with {args.checkpoint}...")
    model.train(
        data          = yaml_path,
        resume        = args.resume,
        epochs        = args.epochs,
        batch         = args.batch_size,
        workers       = args.workers,
        lr0           = args.lr0,
        lrf           = args.lrf,
        warmup_epochs = args.warmup_epochs,
        cos_lr        = args.cos_lr,
        patience      = args.patience,
        save_period   = args.save_period,
        verbose       = args.verbose,
        name          = args.project_name
    )

    # 6. Final Evaluation
    print("Starting final evaluation...")
    # Attempt to find the best model produced by this run or use the one specified
    best_pt = os.path.join(model.trainer.save_dir, 'weights', 'best.pt') if hasattr(model, 'trainer') else args.checkpoint.replace('last.pt', 'best.pt')
    
    if os.path.exists(best_pt):
        model_best = YOLO(best_pt)
        metrics = model_best.val(batch=args.batch_size, workers=args.workers)

        print(f"\n{'='*50}")
        print(f"  mAP50    : {metrics.box.map50:.4f}")
        print(f"  mAP50-95 : {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall   : {metrics.box.mr:.4f}")
        print(f"{'='*50}")

        print("\nPer-class results:")
        for i, name in enumerate(args.class_names):
            ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            ap   = metrics.box.ap[i]   if i < len(metrics.box.ap)   else 0
            print(f"  {name:22s}: mAP50={ap50:.4f}  mAP50-95={ap:.4f}")
    else:
        print(f"[!] Best model not found at {best_pt}, skipping evaluation.")

    print("\nTraining Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Safara Project")
    
    # Roboflow config
    parser.add_argument("--roboflow-workspace", type=str, default="yolotraining-gmh0n")
    parser.add_argument("--roboflow-project", type=str, default="safara-object-detection")
    parser.add_argument("--roboflow-version", type=int, default=4)
    
    # Model config
    parser.add_argument("--checkpoint", type=str, default="models/last.pt", help="Path to initial weights")
    parser.add_argument("--project-name", type=str, default="run_safara_v4", help="Name of the training run")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--cos-lr", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    
    # Adaptive LR config
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    
    # Classes
    parser.add_argument("--class-names", nargs="+", default=[
        'hairnet', 'jas_lab', 'kacamata_pelindung', 'masker',
        'orang', 'sarung_tangan', 'tangan', 'wajah',
    ])

    args = parser.parse_args()
    train_yolo(args)
