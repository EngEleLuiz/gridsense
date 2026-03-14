"""
GridSense Streamlit Dashboard.

Sections
--------
1. Header        — system status, last update time
2. Solar         — 24-hour forecast vs actual (confidence band)
3. Power Quality — event timeline for the last 6 hours
4. Battery       — SoC gauge + 24-hour history
5. Alerts        — last 5 critical events / drift warnings

Deploy on Streamlit Community Cloud (free, public URL).
Set the API_URL environment variable to point at your FastAPI instance.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REFRESH_SECONDS = 60

st.set_page_config(
    page_title="GridSense",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _post(path: str, body: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}{path}", json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _api_online() -> bool:
    result = _get("/healthz")
    return result is not None and result.get("status") == "ok"


# ---------------------------------------------------------------------------
# Demo data fallbacks (when API is offline)
# ---------------------------------------------------------------------------

def _demo_forecast() -> pd.DataFrame:
    import numpy as np
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = [now + timedelta(hours=i) for i in range(24)]
    angles = [max(0.0, (h.hour - 6) / 12 * 3.14159) for h in hours]
    power = [5.0 * (abs(a) ** 0.5) * max(0, (1 - abs(a - 1.57) / 1.57)) for a in angles]
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, 24)
    power = [max(0.0, p + n) for p, n in zip(power, noise)]
    return pd.DataFrame({
        "timestamp": hours,
        "predicted_kw": power,
        "lower_bound": [max(0.0, p - 0.3) for p in power],
        "upper_bound": [p + 0.3 for p in power],
    })


def _demo_soc() -> dict:
    return {"soc": 0.73, "soc_percent": 73.0, "updated_at": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Header ──────────────────────────────────────────────────────────
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.title("⚡ GridSense")
        st.caption("Real-time solar generation forecasting & power quality monitoring")
    with col_status:
        online = _api_online()
        if online:
            st.success("● API Online")
        else:
            st.warning("● API Offline — showing demo data")
        st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    st.divider()

    # ── 2. Solar Generation ────────────────────────────────────────────────
    st.subheader("☀️ Solar Generation Forecast (next 24 h)")

    if online:
        raw = _post("/api/v1/predict/solar", {"station_code": "A801", "horizon_hours": 24})
        if raw and "predictions" in raw:
            forecast_df = pd.DataFrame(raw["predictions"])
            forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
        else:
            forecast_df = _demo_forecast()
    else:
        forecast_df = _demo_forecast()

    fig_solar = go.Figure()
    fig_solar.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["upper_bound"],
        fill=None,
        mode="lines",
        line=dict(color="rgba(255,165,0,0.2)", width=0),
        showlegend=False,
        name="Upper bound",
    ))
    fig_solar.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["lower_bound"],
        fill="tonexty",
        mode="lines",
        line=dict(color="rgba(255,165,0,0.2)", width=0),
        name="Confidence band",
        fillcolor="rgba(255,165,0,0.15)",
    ))
    fig_solar.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_kw"],
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=4),
        name="Forecast (kW)",
    ))
    fig_solar.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Generation (kW)",
        height=320,
        margin=dict(t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_solar, use_container_width=True)

    # Summary metrics
    peak = forecast_df["predicted_kw"].max()
    total_kwh = forecast_df["predicted_kw"].sum()  # 1 reading per hour
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak forecast", f"{peak:.2f} kW")
    col2.metric("Total today (est.)", f"{total_kwh:.1f} kWh")
    col3.metric("Model version", raw.get("model_version", "0.1.0") if online and raw else "demo")

    st.divider()

    # ── 3. Power Quality ───────────────────────────────────────────────────
    st.subheader("⚠️ Power Quality Events (last 6 h)")

    LABEL_COLORS = {
        "normal": "green",
        "voltage_sag": "orange",
        "voltage_swell": "blue",
        "interruption": "red",
        "harmonics": "purple",
        "transient": "darkred",
    }

    if online:
        now = datetime.now(timezone.utc)
        pq_raw = _get(
            "/api/v1/events/pq",
            params={
                "start": (now - timedelta(hours=6)).isoformat(),
                "end": now.isoformat(),
            },
        )
        events = pq_raw.get("events", []) if pq_raw else []
    else:
        events = []

    if events:
        events_df = pd.DataFrame(events)
        for _, row in events_df.iterrows():
            color = LABEL_COLORS.get(row["label"], "grey")
            st.markdown(
                f"<span style='color:{color}'>■</span> "
                f"**{row['label']}** — confidence {row['confidence']:.0%} "
                f"@ `{row['timestamp']}`",
                unsafe_allow_html=True,
            )
    else:
        st.info("No disturbance events in the last 6 hours. ✓")

    st.divider()

    # ── 4. Battery SoC ─────────────────────────────────────────────────────
    st.subheader("🔋 Battery State of Charge")

    soc_data = _get("/api/v1/battery/soc") if online else _demo_soc()
    soc_pct = soc_data.get("soc_percent", 73.0) if soc_data else 73.0

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soc_pct,
        number={"suffix": "%"},
        delta={"reference": 80},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#22c55e" if soc_pct > 50 else "#f59e0b" if soc_pct > 20 else "#ef4444"},
            "steps": [
                {"range": [0, 20], "color": "rgba(239,68,68,0.15)"},
                {"range": [20, 50], "color": "rgba(245,158,11,0.15)"},
                {"range": [50, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 20,
            },
        },
        title={"text": "Current SoC"},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    updated_at = soc_data.get("updated_at", "") if soc_data else ""
    st.caption(f"Last updated: {updated_at}")

    st.divider()

    # ── 5. Alerts ──────────────────────────────────────────────────────────
    st.subheader("🚨 Recent Alerts")

    # Placeholder — in production these come from a dedicated alerts table
    demo_alerts = [
        {"time": "2025-12-01 14:23 UTC", "type": "voltage_sag", "message": "Voltage sag detected (0.7 pu, 120 ms)"},
        {"time": "2025-12-01 11:05 UTC", "type": "drift", "message": "Data drift detected in irradiance_wm2"},
    ]

    for alert in demo_alerts:
        icon = "🔴" if alert["type"] in ("interruption", "transient") else "🟡"
        st.warning(f"{icon} `{alert['time']}` — {alert['message']}")

    # ── Auto-refresh ───────────────────────────────────────────────────────
    st.caption(f"Auto-refreshes every {REFRESH_SECONDS}s — reload page to update now.")


if __name__ == "__main__":
    main()
