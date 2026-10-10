import datetime
import pandas as pd
import streamlit as st
import requests

import integrating_gemma as core

MODEL_LABELS = {"knn": "K-Nearest Neighbors", "rf": "Random Forest", "lr": "Logistic Regression"}

ICONS = {"Earthquake": "🌋", "Flood": "🌊", "Cyclone": "🌀", "Wildfire": "🔥",
         "Drought": "🏜️", "Landslide": "⛰️", "Heatwave": "🥵", "No_Event": "✅"}

FEATURE_GROUPS = {
    "Time and context": ["region", "year", "month", "day_of_year",
                         "hemisphere", "climate_zone"],
    "Weather": ["temperature_c", "temp_anomaly_c", "humidity_pct", "pressure_hpa",
                "wind_speed_ms", "sea_surface_temp_c"],
    "Rain and land moisture": ["rainfall_24h_mm", "rainfall_7d_mm", "rainfall_30d_mm",
                               "rainfall_90d_mm", "soil_moisture", "spei_drought_index",
                               "ndvi", "fire_danger_index"],
    "Terrain and location": ["latitude", "longitude", "elevation_m", "slope_deg",
                             "dist_to_coast_km", "dist_to_river_km", "dist_to_fault_km",
                             "dist_to_plate_boundary_km"],
    "Seismic activity and history": ["regional_quakes_30d", "regional_max_mag_30d",
                                     "events_past_10y_100km", "days_since_last_event_100km"],
    "Population and services": ["population_25km", "population_density", "nearest_hospital_km",
                                "hospital_count_25km", "nearest_clinic_km", "clinic_count_25km",
                                "nearest_school_km", "school_count_25km", "nearest_firestation_km",
                                "firestation_count_25km"],
}

SLIDER_RANGES = {
    "temperature_c": (-30.0, 50.0, 0.5),
    "temp_anomaly_c": (-5.0, 8.0, 0.1),
    "humidity_pct": (0.0, 100.0, 1.0),
    "pressure_hpa": (500.0, 1050.0, 1.0),
    "wind_speed_ms": (0.0, 60.0, 0.5),
    "sea_surface_temp_c": (10.0, 33.0, 0.5),
    "rainfall_24h_mm": (0.0, 500.0, 1.0),
    "rainfall_7d_mm": (0.0, 600.0, 1.0),
    "rainfall_30d_mm": (0.0, 1000.0, 5.0),
    "rainfall_90d_mm": (0.0, 2000.0, 10.0),
    "soil_moisture": (0.0, 1.0, 0.01),
    "spei_drought_index": (-3.0, 3.0, 0.1),
    "ndvi": (0.0, 0.95, 0.01),
    "fire_danger_index": (0.0, 100.0, 1.0),
    "latitude": (-60.0, 70.0, 0.1),
    "longitude": (-180.0, 180.0, 0.1),
    "elevation_m": (0.0, 8000.0, 10.0),
    "slope_deg": (0.0, 60.0, 0.5),
    "dist_to_coast_km": (0.0, 1500.0, 5.0),
    "dist_to_river_km": (0.0, 100.0, 0.5),
    "dist_to_fault_km": (0.0, 500.0, 1.0),
    "dist_to_plate_boundary_km": (0.0, 1500.0, 5.0),
    "regional_quakes_30d": (0, 40, 1),
    "regional_max_mag_30d": (0.0, 8.0, 0.1),
    "events_past_10y_100km": (0, 30, 1),
    "days_since_last_event_100km": (0.0, 3000.0, 10.0),
    "population_density": (0.0, 5000.0, 10.0),
    "nearest_hospital_km": (0.0, 100.0, 0.5),
    "hospital_count_25km": (0, 30, 1),
}

PRESETS = {
    "(none)": {},
    "Flood-like": dict(rainfall_24h_mm=120, rainfall_7d_mm=300, rainfall_30d_mm=450,
                       soil_moisture=0.55, dist_to_river_km=1, elevation_m=20, slope_deg=1,
                       humidity_pct=92, pressure_hpa=1004),
    "Landslide-like": dict(slope_deg=40, rainfall_7d_mm=250, rainfall_30d_mm=400,
                           soil_moisture=0.55, elevation_m=1800, humidity_pct=90,
                           latitude=28, longitude=84.5),
    "Cyclone-like": dict(sea_surface_temp_c=29.5, dist_to_coast_km=15, pressure_hpa=995,
                         wind_speed_ms=22, humidity_pct=88, rainfall_24h_mm=90,
                         elevation_m=5, latitude=20, longitude=88),
    "Wildfire-like": dict(temperature_c=36, temp_anomaly_c=4, humidity_pct=15, wind_speed_ms=14,
                          rainfall_30d_mm=3, rainfall_7d_mm=0, fire_danger_index=85,
                          spei_drought_index=-1.5),
    "Drought-like": dict(spei_drought_index=-2.5, rainfall_90d_mm=20, rainfall_30d_mm=5,
                         humidity_pct=25, ndvi=0.2, soil_moisture=0.05, temp_anomaly_c=2),
    "Heatwave-like": dict(temperature_c=42, temp_anomaly_c=6, humidity_pct=45, wind_speed_ms=3,
                          rainfall_30d_mm=10, rainfall_7d_mm=0, soil_moisture=0.1),
    "Earthquake-like": dict(dist_to_fault_km=3, dist_to_plate_boundary_km=10,
                            regional_quakes_30d=14, regional_max_mag_30d=5.6,
                            latitude=35.7, longitude=139.7),
    "Calm / no event": dict(rainfall_24h_mm=0, rainfall_7d_mm=10, rainfall_30d_mm=50,
                            humidity_pct=55, slope_deg=1, dist_to_fault_km=300,
                            dist_to_river_km=30, wind_speed_ms=3),
}

DEFAULT_DESCRIPTION = ("Heavy rain for 3 days, about 250 mm, steep 35 degree hillside, "
                       "humidity 90%, river nearby.")

def display_name(feature):
    return feature.replace("_", " ").capitalize()

def dataset_unavailable():
    st.error(f"Dataset unavailable. Expected to find it at `{core.DATASET_PATH}`.")

def is_integer_feature(feature):
    return (feature in core.TRAINING_DATA
            and pd.api.types.is_integer_dtype(core.TRAINING_DATA[feature].dtype))

@st.cache_resource
def warm_gemma(model_tag):
    if hasattr(core, "warm_up"):
        core.warm_up()
    return True

def fetch_live_weather(location_name):
    try:
        geo_res = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1", timeout=5).json()
        if "results" not in geo_res or len(geo_res["results"]) == 0:
            return None, "Location not found via Geocoding API."
        
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,surface_pressure,wind_speed_10m"
        w_res = requests.get(weather_url, timeout=5).json()
        current = w_res.get("current", {})
        
        now = datetime.datetime.now()
        
        return {
            "temperature_c": float(current.get("temperature_2m", 20.0)),
            "humidity_pct": float(current.get("relative_humidity_2m", 70.0)),
            "rainfall_24h_mm": float(current.get("precipitation", 0.0) or current.get("rain", 0.0)) * 24,
            "pressure_hpa": float(current.get("surface_pressure", 1013.0)),
            "wind_speed_ms": float(current.get("wind_speed_10m", 3.0)),
            "latitude": float(lat),
            "longitude": float(lon),
            "year": int(now.year),
            "month": int(now.month),
            "day_of_year": int(now.timetuple().tm_yday)
        }, None
    except Exception as e:
        return None, str(e)

def render_dashboard(dataset):
    st.title("Disaster prediction dashboard")
    st.caption("Explore the training data, review model performance, or submit local conditions for an ensemble prediction.")

    if dataset.empty:
        dataset_unavailable()
        return

    event_count = len(dataset)
    class_count = dataset["disaster_type"].nunique() if "disaster_type" in dataset else 0
    model_count = len(core.MODELS)
    first, second, third = st.columns(3)
    first.metric("Historical records", f"{event_count:,}")
    second.metric("Disaster classes", class_count)
    third.metric("Models ready", f"{model_count} / 3")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Records by disaster type")
        if "disaster_type" in dataset:
            st.bar_chart(dataset["disaster_type"].value_counts().rename("Records"))
    with right:
        st.subheader("How to use")
        st.markdown(
            "1. Review data patterns in **EDA report**.\n"
            "2. Compare classifier metrics in **Models**.\n"
            "3. Describe or enter conditions in **Form** to receive the ensemble result."
        )
        st.info("This tool is for decision support only. Always follow official disaster-agency guidance.")

def render_eda(dataset):
    st.title("Exploratory data analysis")
    st.caption("Summary of the cleaned historical dataset used to prepare the prediction features.")
    if dataset.empty:
        dataset_unavailable()
        return

    target = "disaster_type"
    numeric_columns = dataset.select_dtypes(include="number").columns.tolist()
    records, missing = st.columns(2)
    records.metric("Rows", f"{len(dataset):,}")
    missing.metric("Columns with missing values", int((dataset.isna().sum() > 0).sum()))

    if target in dataset:
        st.subheader("Disaster type distribution")
        distribution = dataset[target].value_counts().rename_axis(target).rename("Records")
        st.bar_chart(distribution)

    if "year" in dataset and target in dataset:
        st.subheader("Records over time")
        yearly = dataset.groupby(["year", target], observed=True).size().unstack(fill_value=0)
        st.line_chart(yearly)

    if numeric_columns:
        st.subheader("Numeric feature distribution")
        selected = st.selectbox("Choose a feature", numeric_columns, key="eda_feature")
        values = dataset[selected].dropna()
        if not values.empty:
            counts = pd.cut(values, bins=20).value_counts().sort_index()
            histogram = pd.DataFrame({
                "Range": [str(interval) for interval in counts.index],
                "Records": counts.to_numpy(),
            })
            st.bar_chart(histogram, x="Range", y="Records")
            st.caption(f"Median {display_name(selected).lower()}: {values.median():,.2f}")

    with st.expander("Dataset quality and summary"):
        missing_rows = dataset.isna().sum().rename("Missing values")
        missing_rows = missing_rows[missing_rows > 0].sort_values(ascending=False)
        if missing_rows.empty:
            st.success("No missing values in this dataset.")
        else:
            st.dataframe(missing_rows.to_frame(), width="stretch")
        st.dataframe(dataset.describe(include="all").transpose(), width="stretch")

def render_models():
    st.title("Model performance")
    st.caption("Saved validation metrics for each classifier. The final prediction combines available models using the configured ensemble weights.")

    metric_rows = []
    for name in core.MODELS:
        score_path = core.SCORE_PATHS[name]
        scores = core.load_pkl(score_path)
        if scores is None:
            st.warning(f"Metrics not found for {MODEL_LABELS[name]} ({score_path.name}).")
            continue
        metric_rows.append({
            "Model": MODEL_LABELS[name],
            "Accuracy": scores.get("accuracy"),
            "Precision (macro)": scores.get("precision"),
            "Recall (macro)": scores.get("recall"),
            "F1 (macro)": scores.get("f1"),
        })

    if metric_rows:
        metrics = pd.DataFrame(metric_rows).set_index("Model")
        st.dataframe(metrics.style.format("{:.1%}"), width="stretch")
        st.subheader("Macro F1 comparison")
        st.bar_chart(metrics["F1 (macro)"])

    st.subheader("Ensemble configuration")
    weight_rows = [
        {"Model": MODEL_LABELS[name], "Weight": core.WEIGHTS.get(name, 0)}
        for name in core.MODELS
    ]
    st.dataframe(pd.DataFrame(weight_rows).set_index("Model"), width="stretch")
    st.caption("Metrics describe saved validation runs and are not a guarantee of future predictive accuracy.")

def render_feature_input(feature, preset_name, preset_values):
    label = display_name(feature)
    key = f"input_{preset_name}_{feature}"

    if feature in core.CATEGORICAL_FEATURES:
        options = core.CATEGORICAL_OPTIONS.get(feature, [])
        if options:
            default = str(core.MEDIANS.get(feature, options[0]))
            index = options.index(default) if default in options else 0
            return st.selectbox(label, options, index=index, key=key)
        return st.text_input(label, str(core.MEDIANS.get(feature, "")), key=key)

    default = preset_values.get(feature, core.MEDIANS.get(feature, 0))
    if pd.isna(default):
        default = 0

    if feature in SLIDER_RANGES:
        low, high, step = SLIDER_RANGES[feature]
        cast = int if isinstance(low, int) else float
        value = min(max(cast(default), cast(low)), cast(high))
        return st.slider(label, cast(low), cast(high), value, cast(step), key=key)

    cast = int if is_integer_feature(feature) else float
    return st.number_input(label, value=cast(default), step=1 if cast is int else 0.1, key=key)

def run_prediction(values):
    st.session_state.pop("explanation", None)
    try:
        predictions, votes = core.predict(values)
        st.session_state["prediction_result"] = (values, predictions, votes)
    except Exception as error:
        st.session_state.pop("prediction_result", None)
        st.error(f"Prediction failed ({type(error).__name__}): {error}")

def render_explanation(features, predictions, votes):
    st.subheader("🤖 Gemma explanation")
    cached = st.session_state.get("explanation")
    if cached:
        st.write(cached)
        return

    try:
        if hasattr(core, "explain_stream"):
            text = st.write_stream(core.explain_stream(features, predictions, votes))
        else:
            with st.spinner("Gemma is preparing an explanation..."):
                text = core.explain(features, predictions, votes)
            if text.startswith("[Gemma unavailable"):
                st.error(text.splitlines()[0])
                return
            st.write(text)
        st.session_state["explanation"] = text
    except Exception as error:
        st.error(f"Gemma unavailable: {error}")
        st.info("Use 'Test Gemma connection' in the sidebar to see the exact problem.")

def render_prediction_result(result):
    if not result:
        return

    features, predictions, votes = result
    top, probability = next(iter(predictions.items()))
    st.divider()
    st.subheader("Prediction result")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Most likely outcome", f"{ICONS.get(top, '')} {top}".strip(),
                  f"{probability:.1%} model confidence")
        if probability < 0.5:
            st.warning("The top probability is below 50%; treat this prediction as uncertain.")
        st.bar_chart(pd.Series(predictions, name="Probability"))
    with right:
        st.subheader("Individual model votes")
        vote_rows = [
            {"Model": MODEL_LABELS.get(name, name), "Prediction": prediction}
            for name, prediction in votes.items()
        ]
        st.dataframe(pd.DataFrame(vote_rows), hide_index=True, width="stretch")
        with st.expander("Inputs used (unspecified values use dataset medians or most common categories)"):
            st.json(features)

    if st.session_state.get("use_gemma", False):
        render_explanation(features, predictions, votes)
    else:
        st.info("Gemma explanations are turned off in the sidebar.")
    st.caption("Follow official warnings from your national disaster agency.")

def render_text_tab():
    description = st.text_area("Describe the conditions", DEFAULT_DESCRIPTION, height=110)
    if st.button("Predict from description", type="primary"):
        with st.spinner("Gemma is extracting values..."):
            extracted = core.text_to_features(description)
        if extracted:
            run_prediction(extracted)
        else:
            st.session_state.pop("prediction_result", None)
            st.warning("Couldn't extract any values. Check the Gemma connection, or use the 'Enter values' tab.")

def render_manual_tab():
    st.markdown("### 📡 Live Weather Auto-Sync")
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        weather_location = st.text_input("Enter city or location for live telemetry", "Kathmandu", key="weather_location_input")
    with col_w2:
        st.write("")
        st.write("")
        if st.button("Fetch Live Weather"):
            with st.spinner("Fetching live weather data from Open-Meteo..."):
                w_data, err = fetch_live_weather(weather_location)
                if w_data:
                    for k, v in w_data.items():
                        st.session_state[f"input_(none)_{k}"] = v
                    st.success(f"Successfully synced live weather and date for {weather_location}!")
                else:
                    st.error(f"Failed to fetch weather: {err}")
    st.divider()

    preset_name = st.selectbox("Load a test preset", list(PRESETS), key="preset")
    preset_values = PRESETS[preset_name]

    feature_groups = {}
    assigned = set()
    for group, features in FEATURE_GROUPS.items():
        available = [feature for feature in features if feature in core.FEATURE_COLS]
        if available:
            feature_groups[group] = available
            assigned.update(available)

    with st.form("disaster_prediction_form"):
        values = {}
        for group, features in feature_groups.items():
            st.subheader(group)
            left, right = st.columns(2)
            for index, feature in enumerate(features):
                with (left if index % 2 == 0 else right):
                    values[feature] = render_feature_input(feature, preset_name, preset_values)
        submitted = st.form_submit_button("Predict disaster type", type="primary", use_container_width=True)

    if submitted:
        run_prediction(values)

def render_form():
    st.title("Disaster prediction form")
    st.caption("Describe the situation in words, or enter values directly. Unset values start at the dataset median or most common category.")
    st.info("Predictions are estimates from historical data, not official warnings or a substitute for emergency services.")

    tab_text, tab_manual = st.tabs(["💬 Describe in words", "🎚️ Enter values"])
    with tab_text:
        render_text_tab()
    with tab_manual:
        render_manual_tab()

    render_prediction_result(st.session_state.get("prediction_result"))

def render_dashboard_page():
    render_dashboard(core.TRAINING_DATA)

def render_eda_page():
    render_eda(core.TRAINING_DATA)

def render_sidebar(pages):
    with st.sidebar:
        st.title("🌍 Disaster Risk")
        st.caption("Explore data and estimate likely disaster types")
        for nav_page in pages:
            st.page_link(nav_page)
        st.divider()

        st.write(f"**Models loaded:** {len(core.MODELS)} / 3")
        st.write(f"**Model input features:** {len(core.FEATURE_COLS)}")

        st.session_state["use_gemma"] = st.toggle("Gemma explanations", value=True)
        if st.session_state["use_gemma"]:
            core.GEMMA_MODEL = st.text_input(
                "Ollama model tag", value=core.GEMMA_MODEL,
                help="Must match a name shown by `ollama list`",
            )
            if st.button("Test Gemma connection"):
                try:
                    import ollama
                    names = [m.get("model") or m.get("name")
                             for m in ollama.list().get("models", [])]
                    st.success(f"Ollama reachable. Installed: {names}")
                except Exception as error:
                    st.error(f"{type(error).__name__}: {error}")
            warm_gemma(core.GEMMA_MODEL)

        st.caption("Decision support only. Follow official warnings from your national disaster agency.")

def render_app():
    st.set_page_config(page_title="Disaster Risk Explorer", page_icon="🌍", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {background: #f3f6f8;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        st.Page(render_dashboard_page, title="Dashboard", icon=":material/dashboard:"),
        st.Page(render_eda_page, title="EDA report", icon=":material/analytics:"),
        st.Page(render_models, title="Models", icon=":material/model_training:"),
        st.Page(render_form, title="Form", icon=":material/edit_note:"),
    ]
    page = st.navigation(pages, position="hidden")
    render_sidebar(pages)
    page.run()

if __name__ == "__main__":
    render_app()