from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from django.conf import settings

OHLC_COLUMNS = ("Open", "High", "Low", "Close")
TICKER = "EURUSD=X"
INTERVAL = "1h"
LOOKBACK_DAYS = 7

INITIAL_CANDLES = 20
VIDEO_FPS = 12
FIGURE_SIZE = (12, 6)
FIGURE_DPI = 100


def fetch_eurusd_data() -> pd.DataFrame:
    """Fetch the last 7 days of EUR/USD hourly OHLC data from Yahoo Finance."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    raw = yf.download(
        TICKER,
        start=start,
        end=end,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        return _empty_frame()

    df = _normalize_columns(raw)
    df = df.dropna(subset=list(OHLC_COLUMNS), how="any")
    df = df.sort_values("DateTime").reset_index(drop=True)
    return _cast_dtypes(df)


def generate_replay_video(
    df: pd.DataFrame,
    output_filename: str = "eurusd_replay.mp4",
) -> Path:
    """Compile an animated candlestick MP4 that grows one candle per frame."""
    _validate_replay_data(df)

    plot_data = _to_mpf_format(df)
    output_path = _replays_output_dir() / output_filename
    frame_size = (int(FIGURE_SIZE[0] * FIGURE_DPI), int(FIGURE_SIZE[1] * FIGURE_DPI))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, VIDEO_FPS, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer for {output_path}")

    try:
        first_frame = min(INITIAL_CANDLES, len(plot_data))
        for end in range(first_frame, len(plot_data) + 1):
            frame = _render_candlestick_frame(plot_data.iloc[:end])
            if frame.shape[1::-1] != frame_size:
                frame = cv2.resize(frame, frame_size)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def _validate_replay_data(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("Cannot generate replay video: DataFrame is empty.")

    missing = [col for col in (*OHLC_COLUMNS, "DateTime") if col not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot generate replay video: missing required columns {missing}."
        )


def _to_mpf_format(df: pd.DataFrame) -> pd.DataFrame:
    """Return mplfinance-ready OHLC data indexed by UTC DateTime."""
    ohlc = df.loc[:, ["DateTime", *OHLC_COLUMNS]].copy()
    ohlc["DateTime"] = pd.to_datetime(ohlc["DateTime"], utc=True)
    ohlc = ohlc.dropna(subset=list(OHLC_COLUMNS), how="any")
    if ohlc.empty:
        raise ValueError("Cannot generate replay video: no valid OHLC rows found.")
    return ohlc.set_index("DateTime")


def _render_candlestick_frame(ohlc_df: pd.DataFrame) -> np.ndarray:
    """Render a candlestick chart to an in-memory BGR frame for OpenCV."""
    fig, _axes = mpf.plot(
        ohlc_df,
        type="candle",
        style="charles",
        returnfig=True,
        figsize=FIGURE_SIZE,
        title="EUR/USD Market Replay",
        volume=False,
        warn_too_much_data=10000,
    )

    try:
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height, width, 4
        )
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    finally:
        plt.close(fig)


def _replays_output_dir() -> Path:
    """Return media/replays/, creating it when missing."""
    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root:
        media_root = Path(settings.BASE_DIR) / "media"
    replays_dir = Path(media_root) / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    return replays_dir


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output and keep only OHLC + DateTime."""
    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    df = df.loc[:, [col for col in OHLC_COLUMNS if col in df.columns]]

    if df.index.name is None:
        df.index.name = "DateTime"
    df = df.reset_index()

    datetime_col = "Datetime" if "Datetime" in df.columns else "DateTime"
    if datetime_col in df.columns:
        df = df.rename(columns={datetime_col: "DateTime"})

    return df


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected column types for downstream consumers."""
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    for col in OHLC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=list(OHLC_COLUMNS), how="any")


def _empty_frame() -> pd.DataFrame:
    """Return a typed, column-stable empty frame when no data is available."""
    return pd.DataFrame(
        columns=["DateTime", *OHLC_COLUMNS],
    ).astype(
        {
            "DateTime": "datetime64[ns, UTC]",
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
        }
    )
