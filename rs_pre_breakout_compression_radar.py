from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# ENV / CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUTPUT_CSV = os.getenv(
    "OUTPUT_CSV", "output/rs_pre_breakout_compression_candidates.csv"
)

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0") or "0")  # 0 = unlimited

# 하드컷
MIN_PRICE = float(os.getenv("MIN_PRICE", "10"))
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "500000000"))  # 5억 달러
MIN_DOLLAR_VOLUME_21D = float(os.getenv("MIN_DOLLAR_VOLUME_21D", "6000000"))  # 600만 달러

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "420"))
TOP_OUTPUT = int(os.getenv("TOP_OUTPUT", "12"))

# 입장권
PIVOT_LOOKBACK = int(os.getenv("PIVOT_LOOKBACK", "55"))
RS_LOOKBACK = int(os.getenv("RS_LOOKBACK", "252"))
PRICE_NEAR_RATIO_MIN = float(os.getenv("PRICE_NEAR_RATIO_MIN", "0.95"))  # 전고점 95% 이상
RS_NEAR_RATIO_MIN = float(os.getenv("RS_NEAR_RATIO_MIN", "0.95"))  # RS 최고값의 95% 이상

# 상단 밴드 및 유지율
TOP_BAND_LOW = float(os.getenv("TOP_BAND_LOW", "0.95"))
TOP_BAND_HIGH = float(os.getenv("TOP_BAND_HIGH", "1.05"))
TOP_BAND_LOOKBACK_DAYS = int(os.getenv("TOP_BAND_LOOKBACK_DAYS", "20"))

# 압축 측정
CLOSE_RANGE_DAYS = int(os.getenv("CLOSE_RANGE_DAYS", "10"))
VOLATILITY_SHORT_DAYS = int(os.getenv("VOLATILITY_SHORT_DAYS", "10"))
VOLATILITY_LONG_DAYS = int(os.getenv("VOLATILITY_LONG_DAYS", "30"))
VOLUME_SHORT_DAYS = int(os.getenv("VOLUME_SHORT_DAYS", "10"))
VOLUME_LONG_DAYS = int(os.getenv("VOLUME_LONG_DAYS", "30"))

# 추가 점수
SMA_DISTANCE_WINDOW = int(os.getenv("SMA_DISTANCE_WINDOW", "50"))

# 점수 비중
W_CLOSE_TIGHTNESS = 30.0
W_VOLATILITY_CONTRACTION = 30.0
W_VOLUME_DRY_UP = 20.0
W_SMA50_DISTANCE = 10.0
W_TOP_BAND_HOLD = 10.0

# 등급 기준
GRADE_A_MIN = float(os.getenv("GRADE_A_MIN", "75"))
GRADE_B_MIN = float(os.getenv("GRADE_B_MIN", "60"))
GRADE_WATCH_MIN = float(os.getenv("GRADE_WATCH_MIN", "50"))

# 실행 표시용
BREAKOUT_CONFIRM_BUFFER = float(os.getenv("BREAKOUT_CONFIRM_BUFFER", "0.001"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.08"))
PYRAMID_STEP_1 = float(os.getenv("PYRAMID_STEP_1", "0.08"))
PYRAMID_STEP_2 = float(os.getenv("PYRAMID_STEP_2", "0.16"))

SPY_TICKER = "SPY"
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class MarketRegime:
    spy_close: float
    spy_sma50: float
    spy_sma200: float
    passed: bool
    note: str


@dataclass
class RadarRow:
    ticker: str
    name: str
    sector: str
    industry: str

    close: float
    market_cap: float
    dollar_volume_21d: float

    pivot_55: float
    price_near_ratio: float
    rs_value: float
    rs_high_252: float
    rs_near_ratio: float

    close_range_pct_10d: float
    volatility_ratio_10v30: float
    volume_ratio_10v30: float
    sma50_distance_ratio: float

    band_first_entry_offset: int
    band_hold_days: int
    band_elapsed_days: int
    band_hold_ratio: float

    total_score: float
    grade: str

    score_close_tightness: float
    score_volatility_contraction: float
    score_volume_dry_up: float
    score_sma50_distance: float
    score_top_band_hold: float

    entry_price: float
    stop_price: float
    pyramid_1_price: float
    pyramid_2_price: float


# =========================================================
# UTIL
# =========================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def send_telegram_message(text: str) -> None:
    if not SEND_TELEGRAM:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
        pass


def safe_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def latest(series: pd.Series, n_back: int = 0) -> float:
    s = series.dropna()
    if len(s) <= n_back:
        return np.nan
    return float(s.iloc[-1 - n_back])


def safe_div(a: float, b: float, default: float = np.nan) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return default
    return float(a) / float(b)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def linear_score_low_better(
    value: float, low: float, high: float, max_points: float
) -> float:
    """
    low 쪽이 좋고 high 쪽이 나쁠 때 사용.
    value <= low 면 max_points
    value >= high 면 0점
    """
    if pd.isna(value):
        return 0.0
    if high == low:
        return max_points if value <= low else 0.0
    x = clamp((high - value) / (high - low), 0.0, 1.0)
    return x * max_points


def linear_score_high_better(
    value: float, low: float, high: float, max_points: float
) -> float:
    if pd.isna(value):
        return 0.0
    if high == low:
        return max_points if value >= high else 0.0
    x = clamp((value - low) / (high - low), 0.0, 1.0)
    return x * max_points


def linear_score_mid_best(
    value: float,
    best_low: float,
    best_high: float,
    bad_low: float,
    bad_high: float,
    max_points: float,
) -> float:
    """
    적정 구간(best_low~best_high)이 최고 점수.
    bad_low 이하 또는 bad_high 이상은 0점.
    """
    if pd.isna(value):
        return 0.0

    if bad_low < value < best_low:
        x = (value - bad_low) / (best_low - bad_low)
        return clamp(x, 0.0, 1.0) * max_points

    if best_low <= value <= best_high:
        return max_points

    if best_high < value < bad_high:
        x = (bad_high - value) / (bad_high - best_high)
        return clamp(x, 0.0, 1.0) * max_points

    return 0.0


def rolling_high(series: pd.Series, window: int, exclude_current: bool = False) -> float:
    s = series.dropna()
    if len(s) < window + (1 if exclude_current else 0):
        return np.nan
    if exclude_current:
        s = s.iloc[:-1]
    return float(s.iloc[-window:].max())


def avg_dollar_volume(df: pd.DataFrame, window: int = 21) -> float:
    if "Close" not in df.columns or "Volume" not in df.columns:
        return np.nan
    return float((df["Close"] * df["Volume"]).tail(window).mean())


def load_universe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"ticker", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].map(safe_text)

    if "sector" in df.columns:
        df["sector"] = df["sector"].map(safe_text)
    else:
        df["sector"] = ""

    if "industry" in df.columns:
        df["industry"] = df["industry"].map(safe_text)
    else:
        df["industry"] = ""

    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    else:
        df["market_cap"] = np.nan

    if MAX_SYMBOLS > 0:
        df = df.head(MAX_SYMBOLS).copy()

    return df.reset_index(drop=True)


def download_price_history(tickers: List[str], period_days: int) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: Dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            cols = raw.get(ticker)
            if cols is None:
                continue
            df = cols.copy()
            if "Close" not in df.columns:
                continue
            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                out[ticker] = df
    else:
        t = tickers[0]
        df = raw.copy()
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                out[t] = df

    return out


# =========================================================
# MARKET REGIME
# =========================================================

def compute_market_regime(price_map: Dict[str, pd.DataFrame]) -> MarketRegime:
    spy = price_map.get(SPY_TICKER)
    if spy is None or len(spy) < 220:
        raise RuntimeError("Missing or insufficient SPY history.")

    spy_close = latest(spy["Close"])
    spy_sma50 = float(spy["Close"].rolling(50).mean().iloc[-1])
    spy_sma200 = float(spy["Close"].rolling(200).mean().iloc[-1])

    passed = bool(spy_close > spy_sma200 and spy_sma50 > spy_sma200)
    note = "정상" if passed else "주의"

    return MarketRegime(
        spy_close=round(spy_close, 2),
        spy_sma50=round(spy_sma50, 2),
        spy_sma200=round(spy_sma200, 2),
        passed=passed,
        note=note,
    )


# =========================================================
# RADAR CORE
# =========================================================

def passes_hardcut(df: pd.DataFrame, market_cap_value: float) -> bool:
    if len(df) < max(220, RS_LOOKBACK + 5):
        return False

    close = latest(df["Close"])
    if pd.isna(close) or close < MIN_PRICE:
        return False

    if not pd.isna(market_cap_value) and market_cap_value < MIN_MARKET_CAP:
        return False

    dv21 = avg_dollar_volume(df, 21)
    if pd.isna(dv21) or dv21 < MIN_DOLLAR_VOLUME_21D:
        return False

    sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    sma150 = float(df["Close"].rolling(150).mean().iloc[-1])

    if pd.isna(sma50) or pd.isna(sma150):
        return False

    if not (close > sma50 and sma50 > sma150):
        return False

    return True


def calc_close_range_pct(df: pd.DataFrame, days: int = 10) -> float:
    closes = df["Close"].dropna().tail(days)
    if len(closes) < days:
        return np.nan
    hi = float(closes.max())
    lo = float(closes.min())
    if hi <= 0:
        return np.nan
    return (hi / lo - 1.0) * 100.0


def calc_avg_daily_range_pct(df: pd.DataFrame, days: int) -> float:
    part = df.tail(days).copy()
    if len(part) < days:
        return np.nan
    day_range = (part["High"] - part["Low"]) / part["Close"].replace(0, np.nan)
    return float(day_range.mean()) * 100.0


def calc_band_hold_stats(
    df: pd.DataFrame,
    pivot_55: float,
    lookback_days: int,
    band_low: float,
    band_high: float,
) -> tuple[int, int, int, float]:
    """
    최근 lookback_days 안에서 상단 밴드(0.95~1.05)에 가장 최근 처음 진입한 시점 이후
    유지일수 / 경과일수 계산.

    return:
      first_entry_offset : 최근 lookback_days 기준 몇 일 전 처음 진입했는지 (0=오늘)
      hold_days          : 진입 후 밴드 내 유지한 일수
      elapsed_days       : 진입 후 오늘까지의 총 경과일수
      hold_ratio         : hold_days / elapsed_days
    """
    closes = df["Close"].dropna().tail(lookback_days)
    if len(closes) < 2 or pd.isna(pivot_55) or pivot_55 <= 0:
        return -1, 0, 0, 0.0

    ratio = closes / pivot_55
    in_band = (ratio >= band_low) & (ratio <= band_high)

    idx = list(range(len(in_band)))
    values = in_band.tolist()

    # 가장 최근 False -> True 전환 지점, 또는 맨 처음부터 True인 경우 첫 True를 진입점으로 사용
    entry_idx = None
    for i in range(len(values) - 1, -1, -1):
        if not values[i]:
            continue
        if i == 0:
            entry_idx = 0
            break
        if not values[i - 1]:
            entry_idx = i
            break

    if entry_idx is None:
        # 최근 lookback 기간 내 밴드 진입이 없었다.
        return -1, 0, 0, 0.0

    segment = values[entry_idx:]
    elapsed_days = len(segment)
    hold_days = int(sum(segment))
    hold_ratio = safe_div(hold_days, elapsed_days, default=0.0)
    first_entry_offset = len(values) - 1 - entry_idx

    return first_entry_offset, hold_days, elapsed_days, float(hold_ratio)


def grade_from_score(score: float) -> str:
    if score >= GRADE_A_MIN:
        return "A"
    if score >= GRADE_B_MIN:
        return "B"
    if score >= GRADE_WATCH_MIN:
        return "WATCH"
    return "DROP"


def calc_radar_row(
    ticker: str,
    name: str,
    sector: str,
    industry: str,
    df: pd.DataFrame,
    market_cap_value: float,
    spy_df: pd.DataFrame,
) -> Optional[RadarRow]:
    close = latest(df["Close"])
    sma50 = float(df["Close"].rolling(SMA_DISTANCE_WINDOW).mean().iloc[-1])

    pivot_55 = rolling_high(df["High"], PIVOT_LOOKBACK, exclude_current=True)
    if pd.isna(pivot_55) or pivot_55 <= 0:
        return None

    # 1차 입장권 1: 전고점 근접
    price_near_ratio = safe_div(close, pivot_55)
    if pd.isna(price_near_ratio) or price_near_ratio < PRICE_NEAR_RATIO_MIN:
        return None

    # 1차 입장권 2: RS 근접
    joined = pd.DataFrame(
        {
            "stock_close": df["Close"],
            "spy_close": spy_df["Close"],
        }
    ).dropna()

    if len(joined) < RS_LOOKBACK + 5:
        return None

    joined["rs_line"] = joined["stock_close"] / joined["spy_close"]
    rs_value = float(joined["rs_line"].iloc[-1])
    rs_high_252 = float(joined["rs_line"].iloc[:-1].tail(RS_LOOKBACK).max())

    rs_near_ratio = safe_div(rs_value, rs_high_252)
    if pd.isna(rs_near_ratio) or rs_near_ratio < RS_NEAR_RATIO_MIN:
        return None

    # 점수 항목
    close_range_pct_10d = calc_close_range_pct(df, CLOSE_RANGE_DAYS)

    base_for_vol = df.iloc[:-VOLATILITY_SHORT_DAYS].copy()
    vol_short = calc_avg_daily_range_pct(df, VOLATILITY_SHORT_DAYS)
    vol_long = calc_avg_daily_range_pct(base_for_vol, VOLATILITY_LONG_DAYS)
    volatility_ratio = safe_div(vol_short, vol_long)

    vol10 = float(df["Volume"].tail(VOLUME_SHORT_DAYS).mean())
    vol30 = float(df["Volume"].iloc[:-VOLUME_SHORT_DAYS].tail(VOLUME_LONG_DAYS).mean())
    volume_ratio = safe_div(vol10, vol30)

    sma50_distance_ratio = safe_div(close, sma50)

    band_first_entry_offset, band_hold_days, band_elapsed_days, band_hold_ratio = calc_band_hold_stats(
        df=df,
        pivot_55=pivot_55,
        lookback_days=TOP_BAND_LOOKBACK_DAYS,
        band_low=TOP_BAND_LOW,
        band_high=TOP_BAND_HIGH,
    )

    # 점수화
    score_close_tightness = linear_score_low_better(
        close_range_pct_10d, 2.0, 10.0, W_CLOSE_TIGHTNESS
    )
    score_volatility_contraction = linear_score_low_better(
        volatility_ratio, 0.55, 1.00, W_VOLATILITY_CONTRACTION
    )
    score_volume_dry_up = linear_score_low_better(
        volume_ratio, 0.50, 1.00, W_VOLUME_DRY_UP
    )

    # 50일선 위 적절 거리: 너무 붙어도 애매, 너무 멀어도 과열
    # 대략 1.03 ~ 1.12가 최적
    score_sma50_distance = linear_score_mid_best(
        sma50_distance_ratio,
        best_low=1.03,
        best_high=1.12,
        bad_low=0.99,
        bad_high=1.25,
        max_points=W_SMA50_DISTANCE,
    )

    # 상단 밴드 첫 진입 후 유지율
    score_top_band_hold = linear_score_high_better(
        band_hold_ratio, 0.50, 1.00, W_TOP_BAND_HOLD
    )

    total_score = (
        score_close_tightness
        + score_volatility_contraction
        + score_volume_dry_up
        + score_sma50_distance
        + score_top_band_hold
    )

    grade = grade_from_score(total_score)
    if grade == "DROP":
        return None

    entry_price = pivot_55 * (1 + BREAKOUT_CONFIRM_BUFFER)
    stop_price = entry_price * (1 - STOP_LOSS_PCT)
    pyramid_1_price = entry_price * (1 + PYRAMID_STEP_1)
    pyramid_2_price = entry_price * (1 + PYRAMID_STEP_2)

    return RadarRow(
        ticker=ticker,
        name=name,
        sector=sector,
        industry=industry,

        close=round(close, 2),
        market_cap=float(market_cap_value) if not pd.isna(market_cap_value) else np.nan,
        dollar_volume_21d=round(avg_dollar_volume(df, 21), 2),

        pivot_55=round(pivot_55, 2),
        price_near_ratio=round(price_near_ratio, 4),
        rs_value=round(rs_value, 6),
        rs_high_252=round(rs_high_252, 6),
        rs_near_ratio=round(rs_near_ratio, 4),

        close_range_pct_10d=round(close_range_pct_10d, 2),
        volatility_ratio_10v30=round(volatility_ratio, 3) if not pd.isna(volatility_ratio) else np.nan,
        volume_ratio_10v30=round(volume_ratio, 3) if not pd.isna(volume_ratio) else np.nan,
        sma50_distance_ratio=round(sma50_distance_ratio, 4) if not pd.isna(sma50_distance_ratio) else np.nan,

        band_first_entry_offset=band_first_entry_offset,
        band_hold_days=band_hold_days,
        band_elapsed_days=band_elapsed_days,
        band_hold_ratio=round(band_hold_ratio, 3),

        total_score=round(total_score, 2),
        grade=grade,

        score_close_tightness=round(score_close_tightness, 2),
        score_volatility_contraction=round(score_volatility_contraction, 2),
        score_volume_dry_up=round(score_volume_dry_up, 2),
        score_sma50_distance=round(score_sma50_distance, 2),
        score_top_band_hold=round(score_top_band_hold, 2),

        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        pyramid_1_price=round(pyramid_1_price, 2),
        pyramid_2_price=round(pyramid_2_price, 2),
    )


def build_radar(universe: pd.DataFrame, price_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy_df = price_map[SPY_TICKER]
    rows: List[Dict] = []

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = safe_text(row.get("sector", ""))
        industry = safe_text(row.get("industry", ""))
        market_cap_value = row["market_cap"]

        if ticker == SPY_TICKER:
            continue

        df = price_map.get(ticker)
        if df is None:
            continue

        if not passes_hardcut(df, market_cap_value):
            continue

        radar_row = calc_radar_row(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            df=df,
            market_cap_value=market_cap_value,
            spy_df=spy_df,
        )
        if radar_row is None:
            continue

        rows.append(asdict(radar_row))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    grade_rank = {"A": 3, "B": 2, "WATCH": 1}
    out["grade_rank"] = out["grade"].map(grade_rank).fillna(0)

    out = out.sort_values(
        by=[
            "grade_rank",
            "total_score",
            "band_hold_ratio",
            "price_near_ratio",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return out.head(TOP_OUTPUT).copy()


# =========================================================
# OUTPUT
# =========================================================

def save_output(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_summary_message(regime: MarketRegime, radar_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("RS Pre-Breakout Compression Radar")
    lines.append(f"시각: {utc_now()}")
    lines.append("")
    lines.append("시장")
    lines.append(f"- 상태: {regime.note}")
    lines.append(
        f"- SPY: {regime.spy_close:.2f} / 50MA {regime.spy_sma50:.2f} / 200MA {regime.spy_sma200:.2f}"
    )
    lines.append("")

    if radar_df.empty:
        lines.append("후보 없음")
        return "\n".join(lines)

    lines.append(f"후보 수: {len(radar_df)}")
    lines.append("")

    for i in range(min(len(radar_df), TOP_OUTPUT)):
        row = radar_df.iloc[i]

        lines.append(f"[{i+1}] {row['ticker']} {row['name']}")
        lines.append(
            f"등급 {row['grade']} | 총점 {row['total_score']:.1f} | "
            f"RS근접 {row['rs_near_ratio']*100:.1f}% | 전고근접 {row['price_near_ratio']*100:.1f}%"
        )
        lines.append(
            f"진입 {row['entry_price']:.2f} | 손절 {row['stop_price']:.2f} | "
            f"+8% {row['pyramid_1_price']:.2f} | +16% {row['pyramid_2_price']:.2f}"
        )
        lines.append(
            f"압축  종가범위 {row['close_range_pct_10d']:.2f}% | "
            f"변동성비 {row['volatility_ratio_10v30']:.3f} | 거래량비 {row['volume_ratio_10v30']:.3f}"
        )
        lines.append(
            f"추가  50일거리 {row['sma50_distance_ratio']:.3f} | "
            f"상단유지 {int(row['band_hold_days'])}/{int(row['band_elapsed_days'])}"
        )
        lines.append(
            f"점수  종가압축 {row['score_close_tightness']:.1f}/30 | "
            f"변동성축소 {row['score_volatility_contraction']:.1f}/30"
        )
        lines.append(
            f"      거래량축소 {row['score_volume_dry_up']:.1f}/20 | "
            f"50일거리 {row['score_sma50_distance']:.1f}/10 | "
            f"상단유지 {row['score_top_band_hold']:.1f}/10"
        )

        if i < min(len(radar_df), TOP_OUTPUT) - 1:
            lines.append("")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    universe = load_universe(UNIVERSE_CSV)

    all_tickers = sorted(set([SPY_TICKER] + universe["ticker"].tolist()))
    print(f"[INFO] Downloading data for {len(all_tickers)} tickers...")

    price_map = download_price_history(all_tickers, LOOKBACK_DAYS)

    if SPY_TICKER not in price_map:
        raise RuntimeError("SPY download failed.")

    regime = compute_market_regime(price_map)
    radar_df = build_radar(universe, price_map)

    save_output(radar_df, OUTPUT_CSV)

    message = build_summary_message(regime, radar_df)
    print(message)
    send_telegram_message(message)

    print("")
    print(f"[INFO] Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
