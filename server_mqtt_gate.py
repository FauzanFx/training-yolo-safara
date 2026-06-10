import os
import json
import paho.mqtt.client as mqtt

# ==================== KONFIGURASI MQTT ====================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_REQUEST = "safara/gate/request"
TOPIC_RESPONSE = "safara/gate/response"

# File database eksternal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DB_FILE = os.path.join(BASE_DIR, "whitelist.json")

# ============================================================

def load_whitelist():
    """Fungsi untuk membaca data whitelist dari file JSON secara realtime"""
    if not os.path.exists(JSON_DB_FILE):
        print(f"[!] Warning: File {JSON_DB_FILE} tidak ditemukan. Membuat file kosong baru.")
        # Buat file baru kalau belum ada
        with open(JSON_DB_FILE, 'w') as f:
            json.dump({}, f)
        return {}
    
    try:
        with open(JSON_DB_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Gagal membaca file JSON: {e}")
        return {}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[+] Python Server sukses terhubung ke Mosquitto Broker!")
        client.subscribe(TOPIC_REQUEST)
        print(f"[*] Menunggu ketukan data RFID pada topik: '{TOPIC_REQUEST}'...\n")
    else:
        print(f"[!] Gagal terhubung ke broker, return code: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        uid_terbaca = payload.get("uid", "").lower().strip()
        print(f"[*] Menerima permintaan validasi untuk UID: [{uid_terbaca}]")
        
        # BACA DATABASE JSON (Realtime / Modular)
        valid_tags = load_whitelist()
        
        # Logika Validasi Pencocokan Whitelist
        if uid_terbaca in valid_tags:
            nama_pekerja = valid_tags[uid_terbaca]
            response_data = {
                "status": "allowed", 
                "user": nama_pekerja, 
                "message": "Akses Diberikan"
            }
            print(f"[+++] AKSES DIBERIKAN: {nama_pekerja}")
        else:
            response_data = {
                "status": "denied", 
                "message": "Akses Ditolak"
            }
            print(f"[---] AKSES DITOLAK: UID tidak terdaftar.")
            
        # Kirim balik keputusan akses (Publish) ke ESP32
        client.publish(TOPIC_RESPONSE, json.dumps(response_data))
        print(f"[*] Respon balik berhasil di-publish ke topik: '{TOPIC_RESPONSE}'\n")
        
    except Exception as e:
        print(f"[!] Gagal memproses pesan MQTT: {e}")

# Inisialisasi Klien MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Hubungkan ke Mosquitto local
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Jaga agar script Python terus berjalan mendengarkan data tanpa henti
client.loop_forever()