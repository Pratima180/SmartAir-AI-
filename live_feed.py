"""LIVE mode: virtual IoT sensor node fed by real, recent air-quality data.
Source: Open-Meteo Air Quality API (CAMS ENSEMBLE / Copernicus). Free, no API key.
Attribution: air-quality data by Open-Meteo.com / Copernicus Atmosphere Monitoring Service (CAMS).
Note: this is model-based data for the city's coordinates, not a physical sensor."""
import time
import pandas as pd
import requests
from common import CITY_COORDS, CORE, cpcb_aqi, load_clean

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
REFRESH_SECONDS = 300   # how often 'today' is re-fetched after the history is delivered

def fetch_hourly(city, past_days=10):
    lat, lon = CITY_COORDS[city]
    r = requests.get(URL, params={"latitude": lat, "longitude": lon, "hourly": HOURLY,
                                  "past_days": past_days, "forecast_days": 1,
                                  "timezone": "Asia/Kolkata"}, timeout=20)
    r.raise_for_status()
    return r.json()

def to_daily(resp, now=None):
    """Hourly API response -> one row per day with CPCB-style averaging.
    PM2.5/PM10/NO2/SO2 = daily mean; CO and O3 = max 8-hour rolling mean (CPCB convention)."""
    h = pd.DataFrame(resp["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    now = now if now is not None else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    h = h[h["time"] <= now].dropna().set_index("time")           # drop future forecast hours
    h["CO"] = h["carbon_monoxide"] / 1000.0                       # ug/m3 -> mg/m3 (CPCB unit)
    h["O3"] = h["ozone"].rolling(8, min_periods=4).mean()
    h["CO8"] = h["CO"].rolling(8, min_periods=4).mean()
    g = h.groupby(h.index.date)
    d = pd.DataFrame({"PM2.5": g["pm2_5"].mean(), "PM10": g["pm10"].mean(),
                      "NO2": g["nitrogen_dioxide"].mean(), "SO2": g["sulphur_dioxide"].mean(),
                      "CO": g["CO8"].max(), "O3": g["O3"].max(), "hours": g["pm2_5"].count()})
    return d.dropna()

def _payload(city, day, row):
    p = {k: round(float(row[k]), 3) for k in CORE}
    p.update({"AQI": round(cpcb_aqi(row), 1), "date": str(day), "city": city, "node_id": "node1",
              "data_mode": "live (Open-Meteo/CAMS)"})
    return p

def live_stream(city, history_days=10):
    """First yields the last `history_days` days, then keeps refreshing today's value."""
    if city not in CITY_COORDS:
        raise ValueError(f"No coordinates for {city}. Add it to CITY_COORDS in common.py")
    daily = to_daily(fetch_hourly(city, past_days=history_days))
    for day, row in daily.iterrows():
        yield _payload(city, day, row)
    while True:
        time.sleep(REFRESH_SECONDS)
        daily = to_daily(fetch_hourly(city, past_days=2))
        day, row = daily.index[-1], daily.iloc[-1]
        yield _payload(city, day, row)

def stream(city):
    """live if possible, otherwise fall back to dataset replay (so demo never breaks)."""
    from common import MODE, REPLAY_CITY, sensor_stream
    if MODE == "live":
        try:
            yield from live_stream(city)
            return
        except Exception as e:
            print(f"[live feed failed: {e}] -> falling back to dataset replay ({REPLAY_CITY})")
    yield from sensor_stream(REPLAY_CITY)
