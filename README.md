# Development of an Intelligent Air Quality Monitoring and Pollution Prediction System Using IoT and Machine Learning
Mini project #3

## Architecture
Sensor node (IoT) -> MQTT broker -> Flask backend -> ML model (next-day AQI) -> live dashboard + alerts

| Title keyword | Implementation |
|---|---|
| IoT | `publisher.py` = sensor node publishing JSON over MQTT (HiveMQ). Hardware-ready: an ESP32 can publish the same JSON to the same topic |
| Air Quality Monitoring | Real-time dashboard: PM2.5, PM10, NO2, SO2, CO, O3, CPCB AQI + category, trend graph |
| Pollution Prediction | Next-day AQI forecast (XGBoost / Random Forest / Linear Regression compared against a naive baseline) |
| Intelligent | Alert engine (Warning > 200, Critical > 300 predicted AQI) + health advice |

## Data
- **Training:** CPCB city_day.csv (Kaggle, 2015-2020, 26 Indian cities), AQI recomputed with the CPCB sub-index formula.
- **Live (default):** Open-Meteo Air Quality API (CAMS/Copernicus), free, no key. Acts as a *virtual sensor node*; values are model-based for the city's coordinates, not a physical sensor.
- **Replay:** CPCB dataset replay (automatic fallback when offline).

## Run (Windows PowerShell)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
```
Demo (2 terminals, both with venv active):
```
python publisher.py --city Bhopal --interval 2     # IoT sensor node
python app.py                                       # backend + dashboard
```
Open http://127.0.0.1:5000

Without internet or broker: `$env:SOURCE="local"; python app.py`
Force dataset replay: `$env:MODE="replay"`
Live cities: Bhopal, Indore, Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Lucknow, Patna, Jaipur, Ahmedabad (`common.py` -> CITY_COORDS).

IMPORTANT: change `TOPIC` in `common.py` to a unique name (public broker is shared).

## Evaluation
Time-based split (train on past, test on future; no shuffling). Results in `models/metrics.json`, figure in `models/model_evaluation.png`.

## Limitations (write these in the report)
- Live feed is CAMS model data, not a physical sensor; add real hardware as future scope (ESP32 + PMS5003 + MQ-135).
- Only 6 pollutants shared by dataset and live feed are used.
- Daily AQI is noisy; models improve on the naive baseline mainly in RMSE (large errors), not dramatically in MAE.

Attribution: Air-quality data by Open-Meteo.com (CAMS ENSEMBLE / Copernicus Atmosphere Monitoring Service); CPCB dataset via Kaggle.
