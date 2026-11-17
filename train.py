"""Train + compare models for next-day AQI. Run: python train.py"""
import json, joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from common import load_clean, add_features, FEATURES, MODEL_DIR, DeltaModel

MODEL_DIR.mkdir(exist_ok=True)
df = add_features(load_clean())
print(f"Usable rows: {len(df):,} | cities: {df['City'].nunique()}")

# Time-based split: train on the past, test on the future (no shuffling => no leakage)
cutoff = df["Date"].sort_values().iloc[int(len(df) * 0.8)]
train, test = df[df["Date"] < cutoff], df[df["Date"] >= cutoff]
Xtr, ytr, Xte, yte = train[FEATURES], train["AQI_Tomorrow"], test[FEATURES], test["AQI_Tomorrow"]
print(f"Train: {len(train):,} (< {cutoff.date()}) | Test: {len(test):,} (>= {cutoff.date()})")

def score(pred):
    return {"MAE": round(mean_absolute_error(yte, pred), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 2),
            "R2": round(r2_score(yte, pred), 4)}

results = {"Baseline (tomorrow = today)": score(Xte["AQI"].values)}
models = {
    "Linear Regression": DeltaModel(LinearRegression()),
    "Random Forest": DeltaModel(RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)),
    "XGBoost": DeltaModel(XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.04,
                            subsample=0.8, colsample_bytree=0.8, objective="reg:absoluteerror",
                            random_state=42, n_jobs=-1)),
}
preds = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    preds[name] = m.predict(Xte)
    results[name] = score(preds[name])
    print(f"{name:30s} {results[name]}")
print(f"{'Baseline (tomorrow = today)':30s} {results['Baseline (tomorrow = today)']}")

best = min(models, key=lambda k: results[k]["MAE"])
print("Best model:", best)
joblib.dump(models[best], MODEL_DIR / "model.joblib", compress=3)

imp = getattr(models[best], "feature_importances_", None)
importance = dict(sorted(zip(FEATURES, map(float, imp)), key=lambda x: -x[1])) if imp is not None else {}
json.dump({"best": best, "results": results, "train_rows": len(train), "test_rows": len(test),
           "split_date": str(cutoff.date()), "importance": importance},
          open(MODEL_DIR / "metrics.json", "w"), indent=2)

# Report figures
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(results); ax[0].bar(names, [results[n]["MAE"] for n in names], color="#2b8cbe")
    ax[0].set_title("Model comparison (MAE, lower is better)"); ax[0].tick_params(axis="x", rotation=20)
    t = test.assign(pred=preds[best])
    t = t[t["City"] == "Delhi"] if (t["City"] == "Delhi").any() else t
    ax[1].plot(t["Date"], t["AQI_Tomorrow"], label="Actual", lw=1.2)
    ax[1].plot(t["Date"], t["pred"], label=f"Predicted ({best})", lw=1.2)
    ax[1].set_title("Next-day AQI: actual vs predicted (test period)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(MODEL_DIR / "model_evaluation.png", dpi=130)
    print("Saved models/model_evaluation.png")
except Exception as e:
    print("Plot skipped:", e)
