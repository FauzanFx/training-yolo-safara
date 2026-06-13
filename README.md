# Safara Object Detection & Gate System

Proyek ini terdiri dari dua komponen utama:
1.  **Training YOLOv11**: Script untuk melatih model deteksi objek APD (Alat Pelindung Diri) menggunakan Ultralytics YOLOv11 dan dataset dari Roboflow.
2.  **MQTT Gate Server**: Server Python yang bertindak sebagai gateway untuk memvalidasi akses RFID melalui protokol MQTT.

## Struktur Repository

- `training-yolo.py`: Script utama untuk proses training model.
- `server_mqtt_gate.py`: Server MQTT untuk validasi UID RFID.
- `whitelist.json`: Database sederhana berisi daftar UID RFID yang diizinkan.
- `models/`: Direktori untuk menyimpan checkpoint model (`best.pt`, `last.pt`).
- `requirements.txt`: Daftar dependensi Python yang dibutuhkan.

## Persiapan

### 1. Instalasi Dependensi
Pastikan Anda memiliki Python 3.8+ terinstall. Jalankan perintah berikut untuk menginstall library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Environment
Salin file `.env.example` menjadi `.env` dan isi dengan konfigurasi yang sesuai:

```bash
cp .env.example .env
```

Isi `.env` dengan:
- `ROBOFLOW_API_KEY`: API Key Anda dari Roboflow.
- (Opsional) Konfigurasi MQTT jika berbeda dari default.

## Cara Penggunaan

### Training Model
Untuk memulai training, jalankan:

```bash
python training-yolo.py --epochs 220 --batch-size 4
```

Anda bisa melihat opsi lainnya dengan `python training-yolo.py --help`.

### Menjalankan MQTT Gate Server
Pastikan broker MQTT (seperti Mosquitto) sudah berjalan, lalu jalankan:

```bash
python server_mqtt_gate.py
```

Server akan mendengarkan request pada topik `safara/gate/request` dan mengirim respon pada `safara/gate/response`.

## Fitur Unggulan

- **Adaptive Learning Rate**: Script training dilengkapi dengan callback custom yang menurunkan LR secara otomatis jika loss plateau.
- **Real-time Whitelist**: Server MQTT membaca file `whitelist.json` setiap kali ada request, memungkinkan update data tanpa restart server.
- **Logging & Error Handling**: Implementasi logging standar untuk memudahkan debugging.

## Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).
