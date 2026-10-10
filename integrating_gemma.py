import json
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "knn": ROOT_DIR / "Pkl_Files" / "KNNClassifier.pkl",
    "rf": ROOT_DIR / "Pkl_Files" / "RandomForestClassifier.pkl",
    "lr": ROOT_DIR / "Pkl_Files" / "LogisticRegression.pkl",
}
SCORE_PATHS = {
    "knn": ROOT_DIR / "Scores" / "KNNClassifier_scores.pkl",
    "rf": ROOT_DIR / "Scores" / "RandomForestClassifier_scores.pkl",
    "lr": ROOT_DIR / "Scores" / "LogisticRegression_scores.pkl",
}
SCALER_PATH = ROOT_DIR / "scaler.pkl"
DATASET_PATH = ROOT_DIR / "Dataset" / "disaster_cleaned_dataset.csv"
GEMMA_MODEL = "gemma4:e4b"
WEIGHTS = {"knn": 0.3, "rf": 0.5, "lr": 0.2}
CATEGORICAL_COLS = ["region", "climate_zone", "hemisphere"]
EXCLUDE_COLS = ["Unnamed: 0", "event_id", "date", "disaster_type", "event_within_7d",
                "event_within_30d", "event_within_90d", "days_to_event", "timeframe_bucket",
                "post_magnitude", "post_depth_km", "post_severity", "post_impact_score"]
TEXT_FEATURES = ["temperature_c", "humidity_pct", "rainfall_24h_mm", "rainfall_7d_mm",
                 "rainfall_30d_mm", "wind_speed_ms", "slope_deg", "elevation_m",
                 "dist_to_river_km", "dist_to_coast_km", "latitude", "longitude",
                 "population_density", "pressure_hpa"]
RANGES = {"humidity_pct": (0, 100), "temperature_c": (-60, 60), "latitude": (-90, 90),
          "longitude": (-180, 180), "slope_deg": (0, 90), "wind_speed_ms": (0, 120),
          "pressure_hpa": (850, 1090)}
GEMMA_TIMEOUT = 600
GEMMA_KEEP_ALIVE = "60m"
GEMMA_MAX_TOKENS = 150

def load_pkl(path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

knn, rf, lr = (load_pkl(MODEL_PATHS[name]) for name in ("knn", "rf", "lr"))
scaler = load_pkl(SCALER_PATH)
MODELS = {k: m for k, m in {"knn": knn, "rf": rf, "lr": lr}.items() if m is not None}
if not MODELS:
    raise FileNotFoundError(f"No model files found in {ROOT_DIR / 'Pkl_Files'}.")

def is_pipeline(m):
    return hasattr(m, "named_steps")

def find_feature_cols():
    for m in list(MODELS.values()) + [scaler]:
        names = getattr(m, "feature_names_in_", None)
        if names is not None:
            return list(names)
        if is_pipeline(m) and getattr(m.steps[0][1], "feature_names_in_", None) is not None:
            return list(m.steps[0][1].feature_names_in_)
    if os.path.exists(DATASET_PATH):
        cols = [c for c in pd.read_csv(DATASET_PATH, nrows=5).columns if c not in EXCLUDE_COLS]
        return cols
    sys.exit("Cannot determine feature columns.")

FEATURE_COLS = find_feature_cols()

TRAINING_DATA = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else pd.DataFrame()
MEDIANS = pd.Series(0.0, index=FEATURE_COLS, dtype=object)
CATEGORICAL_FEATURES = set()
CATEGORICAL_OPTIONS = {}
if not TRAINING_DATA.empty:
    for c in FEATURE_COLS:
        if c not in TRAINING_DATA:
            continue
        if pd.api.types.is_numeric_dtype(TRAINING_DATA[c]):
            MEDIANS[c] = TRAINING_DATA[c].median()
        else:
            CATEGORICAL_FEATURES.add(c)
            values = TRAINING_DATA[c].dropna().astype(str)
            if not values.empty:
                MEDIANS[c] = values.mode().iloc[0]
                CATEGORICAL_OPTIONS[c] = sorted(values.unique().tolist())

def build_row(features: dict) -> pd.DataFrame:
    feats = {k: v for k, v in features.items() if v is not None and k in FEATURE_COLS}
    for k, (lo, hi) in RANGES.items():
        if k in feats:
            try:
                feats[k] = float(np.clip(float(feats[k]), lo, hi))
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{k}' must be a number.")
    row = MEDIANS.reindex(FEATURE_COLS).copy()
    for k, v in feats.items():
        if k in CATEGORICAL_FEATURES:
            row[k] = str(v)
        else:
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{k}' must be a number.")
    return pd.DataFrame([row.to_dict()], columns=FEATURE_COLS)

def predict_proba_one(name, model, row_df):
    X = row_df
    if not is_pipeline(model) and name in ("knn", "lr") and scaler is not None:
        X = scaler.transform(row_df)
        if getattr(model, "feature_names_in_", None) is not None:
            X = pd.DataFrame(X, columns=FEATURE_COLS)
    elif getattr(model, "feature_names_in_", None) is None and not is_pipeline(model):
        X = row_df.values
    return model.predict_proba(X)[0]

def predict(features: dict, top_k=3):
    row = build_row(features)
    base = next(iter(MODELS.values()))
    classes = list(getattr(base, "classes_", None) if not is_pipeline(base) else base.classes_)
    total = np.zeros(len(classes))
    wsum = 0.0
    per_model = {}
    for name, model in MODELS.items():
        p = predict_proba_one(name, model, row)
        mc = list(model.classes_)
        p = np.array([p[mc.index(c)] if c in mc else 0.0 for c in classes])
        total += WEIGHTS.get(name, 1 / len(MODELS)) * p
        wsum += WEIGHTS.get(name, 1 / len(MODELS))
        per_model[name] = classes[int(np.argmax(p))]
    total /= wsum
    order = np.argsort(total)[::-1][:top_k]
    return {classes[i]: round(float(total[i]), 3) for i in order}, per_model

def _chat(messages, options, stream=False, json_mode=False):
    import ollama
    client = ollama.Client(timeout=GEMMA_TIMEOUT)
    args = dict(model=GEMMA_MODEL, messages=messages, options=options,
                keep_alive=GEMMA_KEEP_ALIVE, stream=stream)
    if json_mode:
        args["format"] = "json"
    try:
        return client.chat(think=False, **args)
    except TypeError:
        return client.chat(**args)

def gemma(prompt, json_mode=False, temperature=0.2, max_tokens=GEMMA_MAX_TOKENS):
    r = _chat([{"role": "user", "content": prompt}],
              {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 2048},
              json_mode=json_mode)
    return r["message"]["content"]

def warm_up():
    try:
        _chat([{"role": "user", "content": "hi"}], {"num_predict": 1})
    except Exception:
        pass

def text_to_features(text: str) -> dict:
    prompt = (f"Extract values from the text into JSON using exactly these keys: {TEXT_FEATURES}. "
              f"Also include optional keys \"climate_zone\" and \"hemisphere\" (N or S) if stated. "
              f"Use null for anything not clearly stated. Convert units to match key names "
              f"(mm, m/s, degrees C, km). Output JSON only.\n\nText: {text}")
    try:
        data = json.loads(gemma(prompt, json_mode=True, temperature=0, max_tokens=300))
        return {k: v for k, v in data.items() if v is not None}
    except Exception as e:
        return {}

def _explain_prompt(features, preds):
    p = next(iter(preds.values()))
    return (f"Disaster-risk assistant. Use ONLY this data, invent nothing.\n"
            f"Conditions given: {json.dumps(features)}\n"
            f"Model probabilities: {json.dumps(preds)}\n"
            f"In under 90 words: the likely disaster, why, and 3 short precautions.{' Say the prediction is uncertain.' if p < 0.5 else ''}")

def explain(features: dict, preds: dict, per_model: dict) -> str:
    top, p = next(iter(preds.items()))
    try:
        return gemma(_explain_prompt(features, preds))
    except Exception as e:
        return f"[Gemma unavailable: {e}]\nMost likely: {top} ({p:.0%})."

def explain_stream(features: dict, preds: dict, per_model: dict):
    stream = _chat([{"role": "user", "content": _explain_prompt(features, preds)}],
                   {"temperature": 0.2, "num_predict": GEMMA_MAX_TOKENS, "num_ctx": 2048},
                   stream=True)
    for chunk in stream:
        piece = chunk["message"].get("content", "")
        if piece:
            yield piece

def run(user_input: str) -> str:
    user_input = user_input.strip()
    if user_input.startswith("{"):
        features = json.loads(user_input)
    elif "=" in user_input and " " not in user_input.replace(", ", ",").split(",")[0].strip():
        features = {}
        for part in user_input.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                v = v.strip()
                try:
                    features[k.strip()] = float(v)
                except ValueError:
                    features[k.strip()] = v
    else:
        features = text_to_features(user_input)
    preds, per_model = predict(features)
    out = [f"Extracted/used inputs: {features}",
           f"Ensemble probabilities: {preds}",
           f"Model votes: {per_model}", "", explain(features, preds, per_model)]
    return "\n".join(out)

def main():
    while True:
        s = input("> ").strip()
        if s.lower() in ("quit", "exit", ""):
            break
        try:
            print("\n" + run(s) + "\n")
        except Exception as e:
            print(f"[error] {e}\n")

if __name__ == "__main__":
    main()