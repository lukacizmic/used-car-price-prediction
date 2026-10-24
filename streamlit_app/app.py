import os

import requests
import streamlit as st

API_URL = os.getenv("CAR_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Estimator")
st.caption("Manual input app backed by a tuned XGBoost pipeline.")


@st.cache_data(show_spinner=False)
def load_schema() -> dict:
    response = requests.get(f"{API_URL}/schema", timeout=20)
    response.raise_for_status()
    return response.json()


def dropdown(label: str, options: list[str], default: str | None = None) -> str:
    if not options:
        return "Unknown"
    index = 0
    if default is not None and default in options:
        index = options.index(default)
    return st.selectbox(label, options=options, index=index)


schema_payload: dict | None = None
try:
    schema_payload = load_schema()
except requests.RequestException as exc:
    st.error(f"Could not reach API at {API_URL}: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model schema: {exc}")
    st.stop()

if schema_payload is None:
    st.error("Schema payload is empty. Check API and model artifacts.")
    st.stop()

schema = schema_payload["schema"]
metadata = schema_payload.get("metadata", {})

st.subheader("Vehicle Inputs")
num = schema["numeric"]
cat = schema["categorical"]

col1, col2 = st.columns(2)
with col1:
    year = st.number_input(
        "year", value=float(num["year"]["default"]), min_value=float(num["year"]["min"]), max_value=2100.0
    )
    odometer = st.number_input(
        "odometer", value=float(num["odometer"]["default"]), min_value=0.0, max_value=max(1.0, float(num["odometer"]["max"]) * 1.5)
    )
with col2:
    use_coordinates = st.checkbox(
        "Provide latitude/longitude",
        value=False,
        help="If unchecked, the API uses training-default coordinates.",
    )
    if use_coordinates:
        lat = st.number_input("lat", value=float(num["lat"]["default"]), format="%.6f")
        long = st.number_input("long", value=float(num["long"]["default"]), format="%.6f")
    else:
        lat = None
        long = None

manufacturer = dropdown("manufacturer", cat["manufacturer"])
condition = dropdown("condition", cat["condition"])
cylinders = dropdown("cylinders", cat["cylinders"])
fuel = dropdown("fuel", cat["fuel"])
title_status = dropdown("title_status", cat["title_status"])
transmission = dropdown("transmission", cat["transmission"])
drive = dropdown("drive", cat["drive"])
size = dropdown("size", cat["size"])
type_ = dropdown("type", cat["type"])
paint_color = dropdown("paint_color", cat["paint_color"])
state = dropdown("state", cat["state"])
model_choice = dropdown("model", cat["model"], default="__other__")
model = st.text_input("model (only used if model='__other__')", value="Unknown")

payload = {
    "year": year,
    "odometer": odometer,
    "lat": lat,
    "long": long,
    "manufacturer": manufacturer,
    "condition": condition,
    "cylinders": cylinders,
    "fuel": fuel,
    "title_status": title_status,
    "transmission": transmission,
    "drive": drive,
    "size": size,
    "type": type_,
    "paint_color": paint_color,
    "state": state,
    "model": model if model_choice == "__other__" else model_choice,
}

if st.button("Estimate Price", type="primary"):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        st.stop()

    result = response.json()
    st.success(f"Estimated price: ${result['predicted_price']:,.2f}")

    warnings = result.get("warnings", [])
    if warnings:
        st.warning("Warnings:")
        for warning in warnings:
            st.write(f"- {warning}")

    metrics = result.get("model_metrics", {})
    if metrics:
        st.caption(
            "Model test metrics — "
            f"MAE: {metrics.get('test_mae', float('nan')):,.0f}, "
            f"R²: {metrics.get('test_r2', float('nan')):.3f}"
        )

st.divider()
st.caption(
    "API: "
    f"{API_URL} | "
    f"Drop columns: {', '.join(metadata.get('drop_columns', []))}"
)
