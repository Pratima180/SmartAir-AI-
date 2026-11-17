"""SmartAir AI backend: MQTT subscriber -> ML prediction -> alerts -> dashboard API.
Run: python app.py            (uses MQTT broker; start publisher.py separately)
     $env:SOURCE="local"      (PowerShell) -> skip MQTT, backend runs its own sensor node
     $env:MODE="replay"       -> use CPCB dataset instead of live data"""
import json, os, threading, time
from collections import deque
from datetime import datetime
import joblib, pandas as pd
from flask import Flask, jsonify, render_template
import paho.mqtt.client as mqtt
from common import (BROKER, PORT, TOPIC, DEFAULT_CITY, FEATURES, CORE,
                    MODEL_DIR, category, DeltaModel)
from live_feed import stream

app = Flask(__name__)
model = joblib.load(MODEL_DIR / "model.joblib")
metrics = json.load(open(MODEL_DIR / "metrics.json"))

lock = threading.Lock()
readings = deque(maxlen=120)
alerts = deque(maxlen=15)
state = {"source": "starting", "last_level": "ok", "data_mode": ""}
WARN, CRIT = 200, 300   # alert thresholds on PREDICTED next-day AQI

def ingest(p):
    """Turn one raw sensor reading into features, predict tomorrow's AQI, raise alerts."""
    with lock:
        if readings and readings[-1]["date"] == p["date"]:
            readings.pop()                      # same day re-sent (live refresh): replace it
        cur = float(p["AQI"])
        hist = [r["AQI"] for r in readings]
        lag = lambda k: hist[-k] if len(hist) >= k else (hist[0] if hist else cur)
        last = readings[-1] if readings else None
        d = pd.Timestamp(p["date"])
        row = {k: float(p[k]) for k in CORE}
        row.update({"Month": d.month, "DayOfWeek": d.dayofweek, "AQI": cur,
                    "AQI_lag1": lag(1), "AQI_lag2": lag(2), "AQI_lag3": lag(3),
                    "PM25_lag1": last["PM2.5"] if last else float(p["PM2.5"]),
                    "PM10_lag1": last["PM10"] if last else float(p["PM10"]),
                    "AQI_roll3": sum(([cur] + hist[-2:])) / len([cur] + hist[-2:]),
                    "AQI_roll7": sum(([cur] + hist[-6:])) / len([cur] + hist[-6:])})
        pred = min(500.0, max(0.0, float(model.predict(pd.DataFrame([row])[FEATURES])[0])))
        rec = {**{k: float(p[k]) for k in CORE}, "AQI": cur, "date": p["date"],
               "ts": datetime.now().strftime("%H:%M:%S"), "city": p.get("city", ""),
               "predicted": round(pred, 1)}
        state["data_mode"] = p.get("data_mode", "")
        readings.append(rec)
        level = "critical" if pred > CRIT else "warning" if pred > WARN else "ok"
        if level != "ok" and level != state["last_level"]:
            alerts.appendleft({"time": rec["ts"], "date": p["date"],
                               "level": level, "message":
                               f"Predicted AQI {pred:.0f} ({category(pred)['name']}) for tomorrow - {category(pred)['advice']}"})
        state["last_level"] = level

def run_local_simulator():
    state["source"] = "Local simulator (no broker)"
    while True:
        for p in stream(DEFAULT_CITY):
            ingest(p); time.sleep(2)

def start_mqtt():
    def on_message(c, u, msg):
        try: ingest(json.loads(msg.payload))
        except Exception as e: print("bad payload:", e)
    def on_connect(c, u, flags, rc, props=None):
        c.subscribe(TOPIC); state["source"] = f"MQTT {BROKER} | {TOPIC}"
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.on_connect, c.on_message = on_connect, on_message
    try:
        c.connect(BROKER, PORT, 60); c.loop_start()
    except Exception as e:
        print("MQTT connect failed -> using local simulator:", e)
        threading.Thread(target=run_local_simulator, daemon=True).start()

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/live")
def api_live():
    with lock:
        hist = list(readings)
        latest = hist[-1] if hist else None
        out = {"source": state["source"], "data_mode": state["data_mode"],
               "city": (hist[-1]["city"] if hist else DEFAULT_CITY), "history": hist[-60:],
               "alerts": list(alerts), "latest": latest}
        if latest:
            out["cat_now"] = category(latest["AQI"]); out["cat_pred"] = category(latest["predicted"])
    return jsonify(out)

@app.route("/api/metrics")
def api_metrics(): return jsonify(metrics)

if __name__ == "__main__":
    if os.getenv("SOURCE", "mqtt").lower() == "local":
        threading.Thread(target=run_local_simulator, daemon=True).start()
    else:
        start_mqtt()
    print("Open http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
