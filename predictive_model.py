
 # %% 
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import tensorflow as tf

data_dir = Path("quant_data")
quant_data = list(data_dir.glob("*.csv"))

def preprocess_data(data, datetime_col=None):
    """Preprocess the data by handling missing values and resampling."""
    data = data.copy()

    if not isinstance(data.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
        if datetime_col is None and len(data.columns) > 0:
            datetime_col = data.columns[0]

        data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
        data = data.set_index(datetime_col)
        data = data[~data.index.isna()]

    data = data.sort_index()
    data = data.resample("D").mean()  # Resample to daily frequency
    data = data.fillna(method="ffill").fillna(method="bfill")
    return data

def decompose_time_series(data, column):
    """Decompose the time series into trend, seasonal, and residual components."""
    decomposition = seasonal_decompose(data[column], model='additive')
    return decomposition

# ─── Exploratory Data Analysis & Decomposition ─────────────────────────────

 # %%
for file in quant_data:
    data = pd.read_csv(file)
    print(data.head())
    data = preprocess_data(data)

    value_col = "Value" if "Value" in data.columns else None
    if value_col is None:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
    if value_col is None:
        continue

    decomposition = decompose_time_series(data, value_col)

    # Plot the decomposed components
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Observed')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()



 
 # %%
# =============================================================================
# TensorFlow LSTM Predictive Model
# =============================================================================
# Predicts next-day closing prices using daily OHLCV data 

import os
import warnings
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Config ───────────────────────────────────────────────────────────────────
# The model predicts next-day RETURNS rather than price levels. Price levels trend
# out of the training range, so a scaler fit on training prices extrapolates badly
# on the test period. Returns are roughly stationary, so the scaler stays valid.

FEATURES = ["r_open", "r_high", "r_low", "r_close", "d_log_volume"]
TARGET_COL = "r_close"     # next-day close-to-close return
SEQUENCE_LEN = 30          # look-back window (days)
TRAIN_RATIO = 0.80
BATCH_SIZE = 32
EPOCHS = 100
LSTM_UNITS = [128, 64]
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3

# Set a single stock symbol to train, or None to iterate over ALL files --> can analyse any specific stock by changing this variable
STOCK_SYMBOL = None  # e.g. "AXISBANK" | None = run all

# ─── Data Cleaning  ─────────────────────────────────────────────────────────────

def load_and_resample(filepath: Path) -> pd.DataFrame:
    """Load minute-level CSV, resample to daily OHLCV, and add return features."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    daily = df.resample("D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    daily = daily[daily["volume"] > 0]

    # Each price is expressed relative to the previous trading day's close.
    prev_close = daily["close"].shift(1)
    daily["r_open"]  = daily["open"]  / prev_close - 1
    daily["r_high"]  = daily["high"]  / prev_close - 1
    daily["r_low"]   = daily["low"]   / prev_close - 1
    daily["r_close"] = daily["close"] / prev_close - 1
    daily["d_log_volume"] = np.log1p(daily["volume"]).diff()
    return daily.dropna()


def make_sequences(arr: np.ndarray, seq_len: int):
    """Slide a window over arr to create (X, y) pairs for LSTM.
    X[i] holds days i..i+seq_len-1 and y[i] is the return on day i+seq_len."""
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i : i + seq_len])
        y.append(arr[i + seq_len, FEATURES.index(TARGET_COL)])
    # return X as (n_samples, seq_len, n_features) and y as column vector (n_samples, 1)
    return np.array(X), np.array(y).reshape(-1, 1)


def split_and_scale(daily: pd.DataFrame):
    """Scale features (fit on training rows only), build sequences, and split chronologically."""
    values = daily[FEATURES].values.astype(np.float32)
    split_idx = int(len(values) * TRAIN_RATIO)

    scaler = StandardScaler()
    scaler.fit(values[:split_idx])
    scaled = scaler.transform(values)

    X_train, y_train = make_sequences(scaled[:split_idx], SEQUENCE_LEN)
    X_test,  y_test  = make_sequences(scaled[split_idx - SEQUENCE_LEN :], SEQUENCE_LEN)

    t = FEATURES.index(TARGET_COL)
    target_mean, target_std = scaler.mean_[t], scaler.scale_[t]
    return X_train, y_train, X_test, y_test, (target_mean, target_std), split_idx


# ─── Model ────────────────────────────────────────────────────────────────────

def build_lstm_model(seq_len: int, n_features: int) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(seq_len, n_features)),
            LSTM(LSTM_UNITS[0], return_sequences=True),
            Dropout(DROPOUT_RATE),
            LSTM(LSTM_UNITS[1], return_sequences=False),
            Dropout(DROPOUT_RATE),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="StockLSTM",
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model

# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,   # Keras takes the LAST 15% of training data, so validation stays chronological
        callbacks=callbacks,
        verbose=0,
    )
    return history


# ─── Evaluation & plotting ────────────────────────────────────────────────────

def price_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return rmse, mae, mape


def evaluate_and_plot(model, X_test, y_test, target_stats, daily, split_idx, symbol, history):
    """Turn predicted returns into prices, compare with a naive random-walk baseline, and plot."""
    mean, std = target_stats
    r_pred = model.predict(X_test, verbose=0).flatten() * std + mean
    r_true = y_test.flatten() * std + mean

    closes = daily["close"].values
    prev_close = closes[split_idx - 1 : split_idx - 1 + len(r_true)]   # last known close before each test day
    y_true = closes[split_idx : split_idx + len(r_true)]
    y_pred = prev_close * (1 + r_pred)
    y_naive = prev_close                                                # random walk: tomorrow = today

    rmse, mae, mape = price_metrics(y_true, y_pred)
    n_rmse, n_mae, n_mape = price_metrics(y_true, y_naive)
    hit_rate = np.mean(np.sign(r_pred) == np.sign(r_true)) * 100

    print(f"  {symbol:12s} | LSTM RMSE {rmse:9.2f} MAE {mae:9.2f} MAPE {mape:5.2f}% "
          f"| Naive RMSE {n_rmse:9.2f} MAE {n_mae:9.2f} MAPE {n_mape:5.2f}% | direction {hit_rate:5.1f}%")

    test_dates = daily.index[split_idx : split_idx + len(y_true)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{symbol} – LSTM Price Prediction", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(test_dates, y_true, label="Actual Close",    color="steelblue",  linewidth=1.5)
    ax.plot(test_dates, y_pred, label="Predicted Close", color="darkorange", linewidth=1.5, linestyle="--")
    ax.set_title(f"Test Set  |  LSTM MAPE={mape:.2f}%  vs  naive MAPE={n_mape:.2f}%")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history.history["loss"],     label="Train Loss", color="steelblue")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="darkorange")
    ax2.set_title("Training / Validation Loss (MSE, scaled returns)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return {"symbol": symbol, "rmse": rmse, "mae": mae, "mape": mape,
            "naive_rmse": n_rmse, "naive_mae": n_mae, "naive_mape": n_mape,
            "direction_acc": hit_rate, "test_days": len(y_true)}


# ─── Per-stock runner ─────────────────────────────────────────────────────────

def run_for_file(filepath: Path):
    symbol = filepath.stem.replace("_minute", "")
    print(f"\n[{symbol}] Loading & resampling …")

    daily = load_and_resample(filepath)
    if len(daily) < SEQUENCE_LEN * 3:
        print(f"  \u26a0  Not enough data ({len(daily)} days). Skipping.")
        return None

    X_train, y_train, X_test, y_test, target_stats, split_idx = split_and_scale(daily)
    print(
        f"  Train sequences: {len(X_train):,}  |  "
        f"Test sequences: {len(X_test):,}  |  "
        f"Features: {len(FEATURES)}"
    )

    model = build_lstm_model(SEQUENCE_LEN, len(FEATURES))
    print(f"  Training LSTM …")
    history = train_model(model, X_train, y_train)
    print(f"  Training complete ({len(history.history['loss'])} epochs).")

    result = evaluate_and_plot(
        model, X_test, y_test, target_stats, daily, split_idx, symbol, history
    )

    return result


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    all_files = sorted(data_dir.glob("*.csv"))

    if STOCK_SYMBOL:
        target_files = [f for f in all_files if STOCK_SYMBOL in f.stem]
        if not target_files:
            raise FileNotFoundError(f"No CSV found for symbol '{STOCK_SYMBOL}'")
    else:
        target_files = all_files

    print(f"  TensorFlow LSTM  |  v{tf.__version__}")
    print(f"  Stocks to process: {len(target_files)}")
    print(f"  Sequence length  : {SEQUENCE_LEN} days")
    print(f"  LSTM units       : {LSTM_UNITS}")


    results = []
    for fp in target_files:
        r = run_for_file(fp)
        if r:
            results.append(r)

    if results:
        summary = pd.DataFrame(results).set_index("symbol")
        print(summary.round(3).to_string())
        print(f"\nMedian MAPE  LSTM {summary['mape'].median():.2f}%  vs  naive {summary['naive_mape'].median():.2f}%")
        print(f"LSTM beats naive on RMSE for {(summary['rmse'] < summary['naive_rmse']).sum()} of {len(summary)} stocks")


# %%
