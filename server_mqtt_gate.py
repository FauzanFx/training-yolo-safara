import os
import json
import logging
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== KONFIGURASI ====================
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_REQUEST = os.getenv("TOPIC_REQUEST", "safara/gate/request")
TOPIC_RESPONSE = os.getenv("TOPIC_RESPONSE", "safara/gate/response")
JSON_DB_FILE = os.getenv("WHITELIST_FILE", "whitelist.json")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================

def load_whitelist():
    """Fungsi untuk membaca data whitelist dari file JSON secara realtime"""
    if not os.path.exists(JSON_DB_FILE):
        logger.warning(f"File {JSON_DB_FILE} tidak ditemukan. Membuat file kosong baru.")
        try:
            with open(JSON_DB_FILE, 'w') as f:
                json.dump({}, f)
            return {}
        except Exception as e:
            logger.error(f"Gagal membuat file whitelist: {e}")
            return {}
    
    try:
        with open(JSON_DB_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Gagal membaca file JSON: {e}")
        return {}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Python Server sukses terhubung ke Mosquitto Broker!")
        client.subscribe(TOPIC_REQUEST)
        logger.info(f"Menunggu ketukan data RFID pada topik: '{TOPIC_REQUEST}'...")
    else:
        logger.error(f"Gagal terhubung ke broker, return code: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        uid_terbaca = payload.get("uid", "").lower().strip()
        logger.info(f"Menerima permintaan validasi untuk UID: [{uid_terbaca}]")
        
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
            logger.info(f"AKSES DIBERIKAN: {nama_pekerja}")
        else:
            response_data = {
                "status": "denied", 
                "message": "Akses Ditolak"
            }
            logger.info("AKSES DITOLAK: UID tidak terdaftar.")
            
        # Kirim balik keputusan akses (Publish) ke ESP32
        client.publish(TOPIC_RESPONSE, json.dumps(response_data))
        logger.debug(f"Respon balik berhasil di-publish ke topik: '{TOPIC_RESPONSE}'")
        
    except json.JSONDecodeError:
        logger.error(f"Gagal mendecode JSON dari payload: {msg.payload}")
    except Exception as e:
        logger.error(f"Gagal memproses pesan MQTT: {e}")

def main():
    # Inisialisasi Klien MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        # Hubungkan ke Mosquitto
        logger.info(f"Menghubungkan ke broker {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # Jaga agar script Python terus berjalan mendengarkan data tanpa henti
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Server dihentikan oleh pengguna.")
    except Exception as e:
        logger.error(f"Terjadi kesalahan pada server: {e}")

if __name__ == '__main__':
    main()