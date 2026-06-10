import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO


def make_adaptive_lr_callback(patience=10, factor=0.5, min_lr=1e-7):
    """
    ReduceLROnPlateau untuk Ultralytics YOLO.

    Cara kerja:
    - Memantau rata-rata training loss tiap epoch.
    - Jika loss tidak membaik selama `patience` epoch berturut-turut,
      learning rate dikalikan `factor` (dikecilkan).
    - Modifikasi dilakukan pada `initial_lr` di optimizer DAN `base_lrs`
      di scheduler agar LR decay bawaan YOLO tetap berjalan benar.
    - Warmup epoch dilewati agar tidak mengganggu fase pemanasan awal.
    """
    state = {
        'best_loss': float('inf'),
        'wait': 0,
    }

    def on_train_epoch_end(trainer):
        warmup_epochs = int(getattr(trainer.args, 'warmup_epochs', 3))
        if trainer.epoch < warmup_epochs:
            return

        # Ambil rata-rata training loss epoch ini (tloss = accumulated average)
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
            # Cek apakah LR sudah menyentuh lantai minimum
            current_lrs = [pg.get('initial_lr', pg['lr']) for pg in trainer.optimizer.param_groups]
            if max(current_lrs) <= min_lr:
                print(f"[AdaptiveLR] Sudah di min_lr={min_lr:.1e}, tidak dikecilkan lagi.")
                state['wait'] = 0
                return

            # Kurangi initial_lr di setiap param group
            for pg in trainer.optimizer.param_groups:
                old = pg.get('initial_lr', pg['lr'])
                pg['initial_lr'] = max(old * factor, min_lr)
                pg['lr'] = max(pg['lr'] * factor, min_lr)

            # Sinkronkan base_lrs scheduler agar LambdaLR tetap konsisten
            if hasattr(trainer, 'scheduler') and hasattr(trainer.scheduler, 'base_lrs'):
                trainer.scheduler.base_lrs = [
                    max(blr * factor, min_lr)
                    for blr in trainer.scheduler.base_lrs
                ]

            new_lr = max(pg['lr'] for pg in trainer.optimizer.param_groups)
            print(
                f"\n[AdaptiveLR] Epoch {trainer.epoch} — loss stuck {patience} epoch "
                f"(best={state['best_loss']:.4f}) → LR dikecilkan menjadi {new_lr:.2e}"
            )
            state['wait'] = 0

    return {'on_train_epoch_end': on_train_epoch_end}


# Wajib untuk OS Windows
if __name__ == '__main__':

    # 1. Download Dataset dari Roboflow
    # API key dibaca dari environment variable, jangan hardcode di sini.
    # Set via terminal: $env:ROBOFLOW_API_KEY="your_key"  (PowerShell)
    #                   set ROBOFLOW_API_KEY=your_key      (CMD)
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not roboflow_api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY belum diset. "
            "Jalankan: $env:ROBOFLOW_API_KEY='your_key' di PowerShell."
        )
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace("yolotraining-gmh0n").project("safara-object-detection")
    version = project.version(4)
    dataset = version.download("yolov11")

    # 2. Patch data.yaml — paksa 8 kelas kustom
    yaml_path = os.path.join(dataset.location, "data.yaml")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    config['nc'] = 8
    config['names'] = [
        'hairnet', 'jas_lab', 'kacamata_pelindung', 'masker',
        'orang', 'sarung_tangan', 'tangan', 'wajah',
    ]

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("[OK] Sistem dipaksa mengenali 8 kelas kustom APD.")

    # 3. Load model — ganti path ini ke checkpoint terakhirmu
    CHECKPOINT = r'D:\UGM\Akademik\Semester 4\P_IoT\Prject Safara Gate\runs\detect\YOLOv11_Balanced_Training_Local\run_safara_v4_local-2\weights\last.pt'
    model = YOLO(CHECKPOINT)

    # 4. Daftarkan adaptive LR callback
    # patience=10  → kurangi LR jika 10 epoch berturut-turut tanpa perbaikan
    # factor=0.5   → LR dikali 0.5 setiap kali plateau terdeteksi
    # min_lr=1e-7  → batas bawah LR agar tidak nol
    lr_callback = make_adaptive_lr_callback(patience=10, factor=0.5, min_lr=1e-7)
    model.add_callback('on_train_epoch_end', lr_callback['on_train_epoch_end'])

    # 5. Training
    print("Memulai proses training di Lokal GPU (RTX 2050)...")
    model.train(
        data        = yaml_path,
        resume      = True,
        batch       = 4,      # Aman untuk VRAM 4 GB
        workers     = 2,      # Windows butuh workers rendah
        lr0         = 0.001,  # LR awal; adaptive callback akan mengecilkan jika stuck
        lrf         = 0.01,   # LR akhir = lr0 * lrf (setelah seluruh epoch selesai)
        warmup_epochs = 3,
        cos_lr      = True,   # Cosine decay sebagai base schedule
        patience    = 50,     # Early stopping jika 50 epoch tanpa improvement
        save_period = 10,     # Simpan checkpoint tiap 10 epoch
        verbose     = True,
    )

    # 6. Evaluasi akhir
    print("Memulai evaluasi akhir untuk melihat nilai mAP total...")
    metrics = model.val(batch=2, workers=0)
    print(f"Hasil Akhir mAP50   : {metrics.box.map50:.4f}")
    print(f"Hasil Akhir mAP50-95: {metrics.box.map:.4f}")
    print("Training Selesai!")
