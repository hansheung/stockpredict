import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------
# Helpers
# ------------------

def make_xy(df_scaled, look_back):
    X, y = [], []
    for i in range(len(df_scaled) - look_back):
        X.append(df_scaled[i : i + look_back, 0])
        y.append(df_scaled[i + look_back, 0])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def build_model(look_back, units, dense, d1, d2):
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(units, return_sequences=True),
        Dropout(d1),
        LSTM(units, return_sequences=False),
        Dropout(d2),
        Dense(dense, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["RootMeanSquaredError"])
    return model


def series_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE_$": mae,
        "RMSE_$": rmse,
        "R2": r2,
    }


# ------------------
# UI
# ------------------

st.set_page_config(page_title="Stock Closing Price Forecast (LSTM)", layout="wide")
st.title("📈 Stock Closing Price Forecast — LSTM")
st.caption("Data source: Yahoo Finance (3 years history by default)")

with st.sidebar:
    st.header("1) Data")
    period = st.selectbox("Download period", ["1y", "2y", "3y", "5y", "10y"], index=2)
    ticker = st.selectbox("Ticker", ["AAPL", "AMZN", "MSFT", "GOOGL", "NVDA", "TSLA", "META"], index=1)

    st.header("2) Window & Horizon")
    look_back = st.number_input("Look-back window (days)", min_value=10, max_value=200, value=30, step=5)
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=60, value=14, step=1)

    units = 64
    dense = 32
    d1 = 0.2
    d2 = 0.1
    epochs = 50
    
    test_frac = 0.2
    
try:
    df = yf.Ticker(ticker).history(period=period)
except Exception as e:
    st.error(f"yfinance error: {e}")
    st.stop()

if df is None or df.empty:
    st.error("No data returned. Try another period or check ticker symbol.")
    st.stop()

# Ensure Date index and keep Close
df = df.reset_index().set_index("Date").sort_index()
prices = df[["Close"]].dropna().copy()

# normalize timezone -> naive (no tz)
try:
    prices.index = prices.index.tz_localize(None)
except TypeError:
    pass

min_d = prices.index.min().date() 
max_d = prices.index.max().date()
c1, c2 = st.columns(2)

with c1:
    start_date = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
with c2:
    end_date = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)

mask = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
series = prices.loc[mask].copy()

# 50 is way to just say buffer to say 
if len(series) < look_back + 50:
    st.warning("Selected window looks short for LSTM. Consider more history.")

st.subheader(f"{ticker} Stock Data preview")
st.dataframe(series.tail(10))

# Chart raw close
raw_chart = (
    alt.Chart(series.reset_index())
    .mark_line()
    .encode(
        x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d", title="Date")),
        y=alt.Y("Close:Q", title="Close ($)"),
        tooltip=[
            alt.Tooltip("Date:T", format="%Y-%m-%d"),
            alt.Tooltip("Close:Q", format="$.2f"),
        ],
    )
    .properties(height=250)
)
st.altair_chart(raw_chart, use_container_width=True)

values = series.values.astype(float)
N = len(values)

if N <= look_back + 1:
    st.error("Not enough data for chosen look-back.")
    st.stop()

# Time-based split
train_size = int(N * (1 - float(test_frac)))
train_vals = values[:train_size] # 0... train_size
test_vals = values[train_size:]  # train_size...end

# Scaling (fit on train only)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_vals)
test_scaled = scaler.transform(test_vals)

# Windows
X_train, y_train = make_xy(train_scaled, look_back)
X_test, y_test = make_xy(test_scaled, look_back)

if len(X_train) == 0 or len(X_test) == 0:
    st.error("Look-back too large for the selected split; reduce look-back or expand date range.")
    st.stop()

# Train & Forecast
if st.button("🚀 Train & Forecast",  use_container_width=True):
    with st.spinner("Training model…"):
        model = build_model(look_back, units, dense, d1, d2)
        callbacks = [
            EarlyStopping(patience = 10, restore_best_weights = True),
            ReduceLROnPlateau(),
        ]
        model.fit(
            X_train, 
            y_train, 
            batch_size=32,
            validation_data=(X_test, y_test), 
            epochs=int(epochs), 
            callbacks=callbacks
        )

    # Predict test
    y_pred_scaled = model.predict(X_test) ## (n_samples,1)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel() ## flaten to 1D array 
    y_true = scaler.inverse_transform(y_test).ravel()

    # Index alignment
    test_index = series.index[train_size + look_back:]
    test_df = pd.DataFrame({"Actual_$": y_true, "Pred_$": y_pred}, index=test_index)

    # Metrics
    met = series_metrics(y_true, y_pred)

    st.subheader("Evaluation Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE ($)", f"{met['MAE_$']:.2f}")
    m2.metric("RMSE ($)", f"{met['RMSE_$']:.2f}")
    m3.metric("R²", f"{met['R2']:.3f}")

    # Plot test
    st.markdown("### Test: Actual vs Predicted")

    test_chart = (
      alt.Chart(test_df.reset_index())
       #transform_fold is the key: it turns two columns into a single “series” field so Altair can draw two colored lines from one tidy table.
      .transform_fold(["Actual_$", "Pred_$"], as_=["variable", "value"])
      .mark_line()
      .encode(
          x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d", title="Date")),
          y=alt.Y("value:Q", title="Close ($)"),
          color=alt.Color("variable:N", title="Series"),
          tooltip=[
              alt.Tooltip("Date:T", format="%Y-%m-%d"),
              alt.Tooltip("variable:N", title="Series"),
              alt.Tooltip("value:Q", title="Close", format="$.2f"),
          ],
      )
      .properties(height=300)
    )
    
    st.altair_chart(test_chart, use_container_width=True)

    # Multi-step forecast
    with st.spinner("Forecasting…"):
        full_scaled = scaler.transform(values)
        window = full_scaled[-look_back:].reshape(1, look_back, 1) # -look_back: go back 30 or 60 days until the end. Reshape it into 3 dimen.
        fut_scaled = []
        for _ in range(int(horizon)): # Loop through each forcast horizon, example, 14days.
            # 1. Initially, it will from last data go back 30 or 60 days.
            # 3. Recursive function, the window will be predict again
            nxt = model.predict(window)
            fut_scaled.append(nxt.item())
            # 2. window will remove the last element and add the next prediction in the window, and do the predict again.
            window = np.concatenate(
                [window[:, 1:, :],   # will remove the oldest value, 1: slice the 1st element, from 1..end
                 nxt.reshape(1, 1, 1)],  # reshape and append the new predicted value
                axis=1
            )
        
        fut_scaled = np.array(fut_scaled).reshape(-1, 1) # H×1 scaled predictions
        fut = scaler.inverse_transform(fut_scaled).ravel() # back to dollars

    last_date = series.index[-1] # last timestamp in your selected data
    forecast_idx = pd.date_range(
        last_date + pd.Timedelta(days=1), # start the day *after* last_date
        periods=int(horizon), # how many future timestamps
        freq="B" # 'Business day' frequency (Mon–Fri)
    )
    forecast_df = pd.DataFrame({"Forecast_$": fut}, index=forecast_idx)

    st.markdown("### Forecast")
    c1, c2 = st.columns([2, 1])
    with c1:
        recent = series.iloc[-max(100, look_back + 1):].copy()
        recent["Type"] = "Recent Close"
        fc = forecast_df.copy()
        fc["Type"] = "Forecast"
        plot_df = pd.concat([
            recent.rename(columns={"Close": "Close_$"}),
            fc.rename(columns={"Forecast_$": "Close_$"})
        ])

        chart = (
            alt.Chart(
                plot_df.reset_index(names="Date")
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d", title="Date")),
                y=alt.Y("Close_$:Q", title="Close ($)"),
                color=alt.Color("Type:N"),
                tooltip=[
                    alt.Tooltip("Date:T", format="%Y-%m-%d"),
                    alt.Tooltip("Close_$:Q", title="Close", format="$.2f"),  # <- use Close_$
                    alt.Tooltip("Type:N", title="Series"),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.dataframe(forecast_df.round(2))
        csv = forecast_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")

    st.info(f"Last known date in range: {last_date.date()} — Forecast starts on {(last_date + pd.Timedelta(days=1)).date()} for {int(horizon)} day(s).")

else:
    st.info("Adjust parameters on the left, then click **Train & Forecast**.")
