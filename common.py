"""Shared config + data helpers for SmartAir AI."""
import os
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "city_day.csv"
MODEL_DIR = BASE / "models"

# MQTT (public test broker). Change TOPIC to something unique to you!
BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "smartairai/mini_project_demo/node1")

DEFAULT_CITY = os.getenv("CITY", "Bhopal")
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
# 6 core pollutants: available in BOTH the CPCB dataset and the live feed
CORE = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
FEATURES = CORE + ["Month", "DayOfWeek", "AQI", "AQI_lag1", "AQI_lag2", "AQI_lag3",
                   "PM25_lag1", "PM10_lag1", "AQI_roll3", "AQI_roll7"]

# Live mode: "virtual sensor node" location (lat, lon). Replay mode uses REPLAY_CITY.
CITY_COORDS = {"Bhopal": (23.2599, 77.4126), "Indore": (22.7196, 75.8577),
               "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
               "Bengaluru": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
               "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
               "Lucknow": (26.8467, 80.9462), "Patna": (25.5941, 85.1376),
               "Jaipur": (26.9124, 75.7873), "Ahmedabad": (23.0225, 72.5714)}
REPLAY_CITY = os.getenv("REPLAY_CITY", "Delhi")
MODE = os.getenv("MODE", "live").lower()      # live = real recent data, replay = CPCB dataset

# CPCB breakpoints: (conc_low, conc_high, index_low, index_high)
# PM2.5/PM10/NO2/SO2 = 24h avg (ug/m3), CO = 8h max (mg/m3), O3 = 8h max (ug/m3)
_BP = {
 "PM2.5": [(0,30,0,50),(30,60,50,100),(60,90,100,200),(90,120,200,300),(120,250,300,400),(250,380,400,500)],
 "PM10":  [(0,50,0,50),(50,100,50,100),(100,250,100,200),(250,350,200,300),(350,430,300,400),(430,550,400,500)],
 "NO2":   [(0,40,0,50),(40,80,50,100),(80,180,100,200),(180,280,200,300),(280,400,300,400),(400,520,400,500)],
 "SO2":   [(0,40,0,50),(40,80,50,100),(80,380,100,200),(380,800,200,300),(800,1600,300,400),(1600,2000,400,500)],
 "CO":    [(0,1,0,50),(1,2,50,100),(2,10,100,200),(10,17,200,300),(17,34,300,400),(34,44,400,500)],
 "O3":    [(0,50,0,50),(50,100,50,100),(100,168,100,200),(168,208,200,300),(208,748,300,400),(748,1000,400,500)],
}

def sub_index(pollutant, c):
    for cl, ch, il, ih in _BP[pollutant]:
        if c <= ch:
            return il + (ih - il) * (max(c, cl) - cl) / (ch - cl)
    return 500.0

def cpcb_aqi(row):
    """CPCB National AQI = max of pollutant sub-indices."""
    return max(sub_index(k, float(row[k])) for k in CORE)

# CPCB AQI bands: (upper bound, name, colour, advice)
BANDS = [
    (50,  "Good",         "#2ecc71", "Air is clean. Normal outdoor activity is fine."),
    (100, "Satisfactory", "#a3d977", "Acceptable. Very sensitive people should take it easy."),
    (200, "Moderate",     "#f1c40f", "Asthma/heart patients should limit outdoor exertion."),
    (300, "Poor",         "#e67e22", "Wear a mask outdoors; avoid long outdoor exercise."),
    (400, "Very Poor",    "#e74c3c", "Avoid outdoor activity; keep windows closed; use a purifier."),
    (10**9, "Severe",     "#8e1b3a", "Health emergency. Stay indoors and use N95 if you must go out."),
]

def category(aqi):
    for ub, name, color, advice in BANDS:
        if aqi <= ub:
            return {"name": name, "color": color, "advice": advice}

def load_clean(path=DATA_FILE):
    """Read CPCB city_day.csv, fill gaps per city, and compute AQI with the CPCB formula
    (same formula the live feed uses, so training and live data are consistent)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=["AQI"]).dropna(subset=["PM2.5", "PM10"], how="all")
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    for c in CORE:
        df[c] = df.groupby("City")[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())
    df["AQI"] = df.apply(cpcb_aqi, axis=1).round(1)
    return df

def add_features(df):
    """Add lag/rolling features and the target AQI_Tomorrow (consecutive days only)."""
    df = df.copy()
    g = df.groupby("City")
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    for k in (1, 2, 3):
        df[f"AQI_lag{k}"] = g["AQI"].shift(k)
    df["PM25_lag1"] = g["PM2.5"].shift(1)
    df["PM10_lag1"] = g["PM10"].shift(1)
    df["AQI_roll3"] = g["AQI"].transform(lambda s: s.rolling(3).mean())
    df["AQI_roll7"] = g["AQI"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["AQI_Tomorrow"] = g["AQI"].shift(-1)
    next_gap = (g["Date"].shift(-1) - df["Date"]).dt.days
    prev3_gap = (df["Date"] - g["Date"].shift(3)).dt.days
    df = df[(next_gap == 1) & (prev3_gap == 3)]      # only truly consecutive days
    return df.dropna(subset=FEATURES + ["AQI_Tomorrow"]).reset_index(drop=True)

class DeltaModel:
    """Wraps any regressor so it predicts the CHANGE in AQI (tomorrow - today)."""
    def __init__(self, base):
        self.base = base
    def fit(self, X, y):
        self.base.fit(X, y - X["AQI"].values); return self
    def predict(self, X):
        return self.base.predict(X) + X["AQI"].values
    @property
    def feature_importances_(self):
        return self.base.feature_importances_

def sensor_stream(city=None, n=300):
    """REPLAY mode: yield readings from the last n days of a city in the CPCB dataset."""
    city = city or REPLAY_CITY
    df = load_clean()
    d = df[df["City"] == city].tail(n)
    if d.empty:
        raise SystemExit(f"City '{city}' not in dataset")
    for _, r in d.iterrows():
        p = {k: round(float(r[k]), 3) for k in CORE}
        p.update({"AQI": float(r["AQI"]), "date": str(r["Date"].date()), "city": city,
                  "node_id": "node1", "data_mode": "replay (CPCB dataset)"})
        yield p
