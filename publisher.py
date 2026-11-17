"""IoT sensor node (publisher). Sends air-quality readings to the MQTT broker.
MODE=live   -> real recent data for the city (Open-Meteo/CAMS), refreshed every 5 min
MODE=replay -> CPCB dataset replay (offline demo)
Usage: python publisher.py --city Bhopal --interval 2"""
import argparse, json, time
import paho.mqtt.client as mqtt
from common import BROKER, PORT, TOPIC, DEFAULT_CITY
from live_feed import stream

ap = argparse.ArgumentParser()
ap.add_argument("--city", default=DEFAULT_CITY)
ap.add_argument("--interval", type=float, default=2.0, help="seconds between readings")
a = ap.parse_args()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, PORT, 60)
client.loop_start()
print(f"Publishing {a.city} readings to {BROKER}:{PORT}  topic={TOPIC}")
for p in stream(a.city):
    client.publish(TOPIC, json.dumps(p), qos=0)
    print("sent", p["date"], "AQI", p["AQI"], "PM2.5", p["PM2.5"], "|", p["data_mode"])
    time.sleep(a.interval)
