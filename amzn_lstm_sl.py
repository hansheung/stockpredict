# -*- coding: utf-8 -*-
"""
AMZN single‑ticker Streamlit app with date selector and LSTM forecast
- Defaults to AMZN (no multiselect)
- Lets the user pick a historical date range
- Trains a small LSTM to forecast N future business days
- Replaces Best/Worst metrics with a prediction card

Requirements (install in your environment if missing):
  pip install streamlit yfinance pandas altair numpy scikit-learn tensorflow
"""

import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import yfinance as yf
import streamlit as st

from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="AMZN Forecast Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.markdown(
    """
    # :material/query_stats: AMZN Stock Forecast
    Analyze historical **AMZN** closing prices and generate a short‑term LSTM forecast.
    """
)

cols = st.columns([1, 3])

# ---------------------------
# Left column: settings
# ---------------------------
TICKER = "AMZN"

def compat_container(col, **kwargs):
    try:
        # Try modern API first (supports height="stretch" on newer releases)
        return col.container(**kwargs)
    except TypeError:
        # Fallback: remove unsupported args
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in {"height", "vertical_alignment"}}
        return col.container(**safe_kwargs)

top_left = compat_container(cols[0], border=True)               # or height=480 on old versions


# top_left = cols[0].container(border=True, height="stretch")

with top_left:
    st.subheader("Settings")
    st.write(f"**Stock:** {TICKER}")

    # Date range picker
    default_start = dt.date.today() - dt.timedelta(days=365 * 5)  # last 5 years
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=dt.date.today())

    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        st.stop()

    st.markdown("---")
    st.subheader("LSTM Forecast Settings")
    forecast_days = st.number_input("Forecast horizon (business days)", min_value=1, max_value=60, value=7, step=1)
    train_epochs = st.slider("Training epochs", 1, 50, 8)
    seq_len = st.slider("Sequence length (lookback)", 10, 200, 60)

# ---------------------------
# Right column container
# ---------------------------
right = compat_container(cols[1], border=True)

# right = cols[1].container(border=True, height="stretch")

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data(show_spinner=False)
def load_single_close(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError("YFinance returned no data.")
    return df[["Close"]].rename(columns={"Close": ticker})

try:
    data = load_single_close(TICKER, start_date, end_date)
except yf.exceptions.YFRateLimitError:
    st.warning("YFinance is rate-limiting us :( Try again later.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if data[TICKER].isna().empty:
    st.error("No valid close prices returned for the selected period.")
    st.stop()

# ---------------------------
# Prepare series for LSTM
# ---------------------------
series = data[TICKER].dropna().values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)


def make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i, 0])
        y.append(arr[i, 0])
    X = np.array(X)
    y = np.array(y)
    # reshape to [samples, timesteps, features]
    return X.reshape((X.shape[0], X.shape[1], 1)), y


if len(scaled) <= seq_len + 1:
    st.error("Not enough data for the chosen sequence length. Reduce the lookback or extend the date range.")
    st.stop()

X, y = make_sequences(scaled, seq_len)

# Simple train/validation split
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# ---------------------------
# Build and train the model
# ---------------------------
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(seq_len, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=3, restore_best_weights=True)

    with st.spinner("Training LSTM..."):
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_epochs,
            batch_size=32,
            verbose=0,
            callbacks=[es],
        )

    # ---------------------------
    # Recursive multi‑day forecast
    # ---------------------------
    last_window = scaled[-seq_len:].copy()
    preds_scaled = []
    window = last_window.copy()

    for _ in range(forecast_days):
        x_in = window.reshape(1, seq_len, 1)
        yhat = model.predict(x_in, verbose=0)[0][0]
        preds_scaled.append(yhat)
        # slide window forward
        window = np.vstack([window[1:], [[yhat]]])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

    # Build future business-day index
    last_date = data.index[-1]
    future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({TICKER: preds}, index=future_index)

    # ---------------------------
    # Bottom-left: prediction card
    # ---------------------------
    bottom_left = compat_container(cols[0], border=True)
    # bottom_left = cols[0].container(border=True, height="stretch")
    with bottom_left:
        st.metric(
            label="Next predicted close",
            value=f"${forecast_df.iloc[0, 0]:,.2f}",
        )
        st.caption(f"Forecast for {TICKER} on {forecast_df.index[0].date()}")

    # ---------------------------
    # Plot actual + forecast
    # ---------------------------
    with right:
        actual = data.reset_index().rename(columns={TICKER: "Price"})
        actual["Type"] = "Actual"

        fc_plot = forecast_df.reset_index().rename(columns={"index": "Date", TICKER: "Price"})
        fc_plot["Type"] = "Forecast"

        chart = alt.Chart(pd.concat([actual[["Date", "Price", "Type"]], fc_plot[["Date", "Price", "Type"]]], axis=0)).mark_line().encode(
            x=alt.X("Date:T"),
            y=alt.Y("Price:Q").scale(zero=False),
            color="Type:N",
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

    # ---------------------------
    # Raw tables
    # ---------------------------
    st.markdown("## Raw data")
    st.dataframe(data)

    st.markdown("## Forecast")
    st.dataframe(forecast_df)

except Exception as e:
    st.error(
        "LSTM section failed. Ensure TensorFlow is installed (pip install tensorflow) and that your environment has sufficient resources.\n\n"
        f"Error: {e}"
    )
