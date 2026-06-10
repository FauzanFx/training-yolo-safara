import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# Bungkus seluruh kode eksekusi utama di dalam blok ini (Wajib untuk OS Windows)
if __name__ == '__main__':

    # 1. Download Dataset dari Roboflow
    rf = Roboflow(api_key="joHqOvMj31ZnQUTgj84W")
    project = rf.workspace("yolotraining-gmh0n").project("safara-object-detection")
    version = project.version(4)
    dataset = version.download("yolov11")

    # === PROSES INJEKSI & PERBAIKAN DATA.YAML ===
    yaml_path = os.path.join(dataset.location, "data.yaml")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paksa konfigurasi mengenali 8 kelas kustom sesuai di dashboard Roboflow
    config['nc'] = 8
    config['names'] = ['hairnet', 'jas_lab', 'kacamata_pelindung', 'masker', 'orang', 'sarung_tangan', 'tangan', 'wajah']

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("[OK] Sistem dipaksa mengenali 8 kelas kustom APD.")
    # ============================================

    # 2. Load Model Dasar (YOLO11 Nano aman untuk VRAM 4GB)
    model = YOLO(r'D:\UGM\Akademik\Semester 4\P_IoT\Prject Safara Gate\runs\detect\YOLOv11_Balanced_Training_Local\run_safara_v4_local-2\weights\last.pt') 
    model.train(resume=True)

    # 3. Mulai Training di Lokal GPU (RTX 2050)
    print("Memulai proses training di Lokal GPU (RTX 2050)...")
    model.train(
        resume=True                 # MATIKAN validasi per epoch. Ini penghemat waktu paling masif (bisa pangkas waktu hingga 50%)
    )
    print("Memulai evaluasi akhir untuk melihat nilai mAP total...")
    metrics = model.val(batch=2, workers=0) 
    print(f"Hasil Akhir mAP50: {metrics.box.map50}")
    print("Training Selesai!")