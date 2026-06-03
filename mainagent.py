#!/usr/bin/env python3



import os
import sys
import time
import logging
import json
import threading
import math
import csv
import traceback
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date, time as dt_time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from config import *

QUANTUM_OK = False

# ── Optional deps ─────────────────────────────────────────────────────────
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_OK = True
except ImportError:
    BINANCE_OK = False
    print("[WARN] python-binance not installed — SIMULATION mode")

try:
    from flask import Flask, jsonify
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

ML_AVAILABLE = XGB_AVAILABLE = RF_AVAILABLE = TORCH_AVAILABLE = False
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = RF_AVAILABLE = True
except ImportError:
    print("[WARN] scikit-learn not installed")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from llm_brain import LLMBrainAgent
    LLM_BRAIN_AVAILABLE = True
except ImportError:
    LLM_BRAIN_AVAILABLE = False

# ── Config defaults for any missing keys ─────────────────────────────────
_DEFAULTS = {
    'MIN_TRADE_USDT': 10.0,
    'MAX_HOLD_BARS': 12,
    'ENABLE_HOLD_UNTIL_POSITIVE': False,
    'HARD_STOP_PCT': 5.0,
    'USE_VWAP_AGENT': True,
    'ENABLE_DASHBOARD': True,
    'DASHBOARD_PORT': 5050,
    'LOG_LEVEL': 'INFO',
    'MAX_POSITION_PCT': 0.10,
    'FEES': 0.0005,
    'SLIPPAGE': 0.0010,
    'DAILY_LOSS_LIMIT_PCT': 5.0,
    'MAX_CORR_TRADES': 1,
    'CORRELATION_THRESHOLD': 0.80,
    'ATR_STOP_MULT': 2.0,
    'ATR_TAKE_MULT': 4.0,
    'PARTIAL_TAKE_MULT': 0.5,
    'TRAILING_ACTIVATION': 2.0,
    'TRAILING_RETREAT': 0.025,
    'MAX_CONSECUTIVE_LOSSES': 3,
    'MIN_AGREEING_AGENTS': 4,
    'GATE_MIN_CONFIDENCE': 0.62,
    'GATE_MAX_CONFLICT': 0.55,
    'GATE_MIN_REGIME_CONF': 0.55,
    'SESSION_BLOCK_BELOW': 0.70,
    'USE_LSTM_AGENT': True,
    'USE_ORDER_FLOW': True,
    'USE_CROSS_ASSET_AGENT': True,
    'USE_SESSION_FILTER': True,
    'ENSEMBLE_WEIGHT_FLOOR': 0.05,
    'USE_LLM_BRAIN': False,
    'BRAIN_PROVIDER': "groq",
    'BRAIN_CACHE_TTL': 180,
    'BRAIN_MIN_CONF': 0.70,
    'ANTHROPIC_API_KEY': '',
    'OPENAI_API_KEY': '',
    'USE_META_LEARNER': True,
    'META_LEARNER_FEATURES': [
        "ensemble_conf","ensemble_buy_score","ensemble_sell_score",
        "regime","adx","volatility","btc_trend","eth_trend",
        "consecutive_losses","hour","avg_agent_conf","num_agreeing_buy",
        "num_agreeing_sell","lstm_prob","llm_sentiment"
    ],
    'META_LEARNER_RETRAIN_EVERY': 30,
    'META_MIN_SAMPLES': 100,
    'META_CONFIDENCE_THRESHOLD': 0.62,
    'USE_MEAN_REVERSION_AGENT': True,
    'USE_TREND_BREAKOUT_AGENT': True,
    'TREND_BREAKOUT_LOOKBACK': 20,
    'TREND_BREAKOUT_LONG_MA': 200,
    'TREND_BREAKOUT_VOL_SPIKE': 2.0,
    'TREND_BREAKOUT_ADX_THRESH': 30,
    'TREND_BREAKOUT_LONG_ONLY': True,
    'TREND_BREAKOUT_USE_BOLLINGER': True,
    'TREND_BREAKOUT_BB_STD': 2,
    'MEAN_REVERSION_LOOKBACK': 20,
    'MEAN_REVERSION_ZSCORE_THRESH': 2.0,
    'MEAN_REVERSION_RSI_OVERSOLD': 28,
    'MEAN_REVERSION_RSI_OVERBOUGHT': 72,
    'MEAN_REVERSION_BB_POSITION': 0.15,
    'USE_NOISE_SCALPER': False,
    'NOISE_SCALPER_LOOKBACK': 20,
    'NOISE_SCALPER_RANGE_PCT': 1.5,
    'NOISE_SCALPER_ADX_MAX': 20,
    'NOISE_SCALPER_PROFIT_TARGET': 0.4,
    'NOISE_SCALPER_STOP_LOSS': 0.2,
    'NOISE_SCALPER_MIN_VOLUME_RATIO': 0.8,
    'NOISE_SCALPER_MAX_HOLD_BARS': 3,
    'TOP_PAIRS_BY_VOLUME': 10,
    'MIN_24H_VOLUME_USDT': 20_000_000,
}
for _k, _v in _DEFAULTS.items():
    if _k not in globals():
        globals()[_k] = _v

# ── Paths ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
Path(ML_SAVE_PATH).mkdir(parents=True, exist_ok=True)
STATE_FILE       = Path("logs/state.json")
TRADES_FILE      = Path("logs/trade_history.jsonl")
SIGNALS_CSV      = Path("logs/signals.csv")
META_MODEL_PATH  = Path("logs/meta_learner.joblib")
META_SCALER_PATH = Path("logs/meta_scaler.joblib")

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("CryptoBot")

# ════════════════════════════════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════════════════════════════════
class Telegram:
    _url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" if 'TELEGRAM_TOKEN' in globals() and TELEGRAM_TOKEN else None

    @staticmethod
    def send(text: str):
        if not Telegram._url:
            return
        try:
            requests.post(Telegram._url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        except Exception:
            pass

    @staticmethod
    def trade_opened(symbol, direction, price, qty, conf):
        Telegram.send(f"<b>OPENED</b> {symbol} {direction} @ {price:.4f}  qty={qty:.6f}  conf={conf:.1%}")

    @staticmethod
    def trade_closed(symbol, direction, entry, exit_price, pnl, reason):
        emoji = "✅" if pnl >= 0 else "❌"
        Telegram.send(f"{emoji} <b>CLOSED</b> {symbol} {direction}  entry={entry:.4f} exit={exit_price:.4f}  PnL={pnl:+.4f} USDT  [{reason}]")

    @staticmethod
    def circuit_breaker(daily_pnl):
        Telegram.send(f"🚨 <b>CIRCUIT BREAKER</b>  Daily PnL={daily_pnl:+.2f} USDT")

    @staticmethod
    def daily_summary(summary: dict):
        Telegram.send(f"📊 Daily  trades={summary['total_trades']}  WR={summary['win_rate']:.1f}%  PnL={summary['total_pnl']:+.2f} USDT")


# ════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════
def rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    s = pd.Series(prices, dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    return float(100 - 100 / (1 + rs))

def ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return float(prices[-1]) if prices else 0.0
    s = pd.Series(prices, dtype=float)
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1])

def atr(candles: list, period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    highs  = pd.Series([c["high"]  for c in candles], dtype=float)
    lows   = pd.Series([c["low"]   for c in candles], dtype=float)
    closes = pd.Series([c["close"] for c in candles], dtype=float)
    tr = pd.concat([highs - lows, (highs - closes.shift(1)).abs(), (lows - closes.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())

def adx(candles: list, period: int = 14) -> float:
    if len(candles) < period * 2:
        return 0.0
    highs  = pd.Series([c["high"]  for c in candles], dtype=float)
    lows   = pd.Series([c["low"]   for c in candles], dtype=float)
    closes = pd.Series([c["close"] for c in candles], dtype=float)
    prev_h, prev_l, prev_c = highs.shift(1), lows.shift(1), closes.shift(1)
    tr   = pd.concat([highs - lows, (highs - prev_c).abs(), (lows - prev_c).abs()], axis=1).max(axis=1)
    pdm  = (highs - prev_h).clip(lower=0).where((highs - prev_h) > (prev_l - lows), 0)
    ndm  = (prev_l - lows).clip(lower=0).where((prev_l - lows) > (highs - prev_h), 0)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    pdi   = 100 * pdm.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
    ndi   = 100 * ndm.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return float(dx.ewm(span=period, adjust=False).mean().iloc[-1])

def stochastic_rsi(prices: list, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3) -> Tuple[float, float]:
    if len(prices) < rsi_period + stoch_period + k_period:
        return 50.0, 50.0
    s     = pd.Series(prices, dtype=float)
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rsi_s = 100 - 100 / (1 + gain / (loss + 1e-9))
    lo = rsi_s.rolling(stoch_period).min()
    hi = rsi_s.rolling(stoch_period).max()
    k  = 100 * (rsi_s - lo) / (hi - lo + 1e-9)
    d  = k.rolling(k_period).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])

def ichimoku(candles: list, conversion: int = 9, base: int = 26, span_b: int = 52) -> dict:
    if len(candles) < span_b:
        return {}
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    tenkan   = (max(highs[-conversion:]) + min(lows[-conversion:])) / 2
    kijun    = (max(highs[-base:])       + min(lows[-base:]))       / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (max(highs[-span_b:])     + min(lows[-span_b:]))     / 2
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b}

def precompute_indicators(candles: list) -> dict:
    """
    Returns indicators computed ONLY on closed candles.
    The current (incomplete) candle is EXCLUDED from indicator calculations.
    This prevents look‑ahead bias in backtesting.
    """
    if len(candles) < 2:
        return {
            "prices": [], "volumes": [], "rsi14": 50.0, "ema9": 0, "ema12": 0,
            "ema20": 0, "ema21": 0, "ema26": 0, "atr14": 0, "adx14": 0,
            "stoch_k": 50.0, "stoch_d": 50.0, "ichimoku": {}, "avg_vol20": 0,
            "stoch_k_classic": 50.0, "stoch_d_classic": 50.0, "williams_r": 0.0,
            "efficiency_ratio": 0.5, "zscore_20": 0.0, "bb_position": 0.5,
            "vwap": 0, "vwap_dev": 0, "current_price": 0, "current_candle": {}
        }

    closed_candles = candles[:-1]
    current_candle = candles[-1]

    prices  = [c["close"] for c in closed_candles]
    volumes = [c["volume"] for c in closed_candles]
    n = len(prices)

    rsi_val = rsi(prices, 14) if n >= 15 else 50.0
    ema9  = ema(prices, 9)  if n >= 9  else (prices[-1] if prices else 0)
    ema12 = ema(prices, 12) if n >= 12 else (prices[-1] if prices else 0)
    ema20 = ema(prices, 20) if n >= 20 else (prices[-1] if prices else 0)
    ema21 = ema(prices, 21) if n >= 21 else (prices[-1] if prices else 0)
    ema26 = ema(prices, 26) if n >= 26 else (prices[-1] if prices else 0)
    atr_val = atr(closed_candles, 14) if n >= 15 else 0.0
    adx_val = adx(closed_candles, 14) if n >= 28 else 0.0
    stoch_k, stoch_d = stochastic_rsi(prices, 14, 14, 3) if n >= 31 else (50.0, 50.0)
    ichi = ichimoku(closed_candles) if n >= 52 else {}
    avg_vol20 = sum(volumes[-20:]) / 20 if n >= 21 else (volumes[-1] if volumes else 0)

    if n >= 14:
        low14  = pd.Series([c["low"] for c in closed_candles]).rolling(14).min().values
        high14 = pd.Series([c["high"] for c in closed_candles]).rolling(14).max().values
        sk_raw = 100 * (np.array(prices) - low14) / (high14 - low14 + 1e-9)
        stoch_k_classic = float(sk_raw[-1])
        stoch_d_classic = float(pd.Series(sk_raw).rolling(3).mean().values[-1])
        williams_r = -100 * (high14[-1] - prices[-1]) / (high14[-1] - low14[-1] + 1e-9)
    else:
        stoch_k_classic = stoch_d_classic = 50.0
        williams_r = 0.0

    if n >= 11:
        change = abs(prices[-1] - prices[-11])
        path   = max(c["high"] for c in closed_candles[-10:]) - min(c["low"] for c in closed_candles[-10:])
        efficiency_ratio = change / (path + 1e-9)
    else:
        efficiency_ratio = 0.5

    if n >= 20:
        s20 = pd.Series(prices)
        ma20 = s20.rolling(20).mean().values[-1]
        sd20 = s20.rolling(20).std().values[-1]
        zscore_20   = (prices[-1] - ma20) / (sd20 + 1e-9)
        bb_position = (prices[-1] - (ma20 - 2 * sd20)) / (4 * sd20 + 1e-9)
    else:
        zscore_20 = 0.0
        bb_position = 0.5

    if n >= 24:
        typical = [(c["high"] + c["low"] + c["close"]) / 3 for c in closed_candles[-24:]]
        vol24   = [c["volume"] for c in closed_candles[-24:]]
        vwap    = sum(t * v for t, v in zip(typical, vol24)) / (sum(vol24) + 1e-9)
    else:
        vwap = prices[-1] if prices else 0

    vwap_dev = (current_candle["close"] - vwap) / (vwap + 1e-9)

    return {
        "prices": prices,
        "volumes": volumes,
        "rsi14": rsi_val,
        "ema9":  ema9,
        "ema12": ema12,
        "ema20": ema20,
        "ema21": ema21,
        "ema26": ema26,
        "atr14": atr_val,
        "adx14": adx_val,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "ichimoku": ichi,
        "avg_vol20": avg_vol20,
        "stoch_k_classic": stoch_k_classic,
        "stoch_d_classic": stoch_d_classic,
        "williams_r": williams_r,
        "efficiency_ratio": efficiency_ratio,
        "zscore_20": zscore_20,
        "bb_position": bb_position,
        "vwap": vwap,
        "vwap_dev": vwap_dev,
        "current_price": current_candle["close"],
        "current_candle": current_candle,
    }


# ════════════════════════════════════════════════════════════════════════════
#  AGENTS
# ════════════════════════════════════════════════════════════════════════════

class TrendHunterAgent:
    name = "TrendHunter"

    def analyze(self, ind: dict, candles: list) -> dict:
        prices = ind["prices"]
        if len(prices) < 27:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        macd  = ind["ema12"] - ind["ema26"]
        sig   = ema([ind["ema12"] - ind["ema26"]] * len(prices), 9)
        macd_bullish = macd > 0 and ind["ema9"] > ind["ema21"]
        macd_bearish = macd < 0 and ind["ema9"] < ind["ema21"]
        separation = abs(ind["ema9"] - ind["ema21"]) / (ind["ema21"] + 1e-9)
        conf = min(0.88, 0.55 + separation * 50)
        if macd_bullish:
            return {"signal": "BUY",  "confidence": round(conf, 3), "agent": self.name}
        if macd_bearish:
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class ADXAgent:
    name = "ADX"

    def __init__(self, threshold=25):
        self.threshold = threshold

    def analyze(self, ind: dict, candles: list) -> dict:
        adx_val = ind["adx14"]
        if adx_val < self.threshold:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        trend = "up" if ind["prices"][-1] > ind["ema20"] else "down"
        conf = min(0.85, 0.55 + (adx_val - self.threshold) / 60)
        return {"signal": "BUY" if trend == "up" else "SELL", "confidence": round(conf, 3), "agent": self.name}


class VolumeSentinelAgent:
    name = "VolumeSentinel"

    def analyze(self, ind: dict, candles: list) -> dict:
        if len(ind["volumes"]) < 20:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        ratio = ind["volumes"][-1] / (ind["avg_vol20"] + 1e-9)
        if ratio < 2.0:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        price_chg = (ind["prices"][-1] - ind["prices"][-2]) / (ind["prices"][-2] + 1e-9)
        if abs(price_chg) < 0.002:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        conf = min(0.88, 0.55 + (ratio - 2.0) * 0.10)
        direction = "BUY" if price_chg > 0 else "SELL"
        return {"signal": direction, "confidence": round(conf, 3), "agent": self.name}


class StochRSIAgent:
    name = "StochRSI"

    def __init__(self, overbought=80, oversold=20):
        self.overbought = overbought
        self.oversold   = oversold

    def analyze(self, ind: dict, candles: list) -> dict:
        k, d = ind["stoch_k"], ind["stoch_d"]
        if k < self.oversold and d < self.oversold and k > d:
            depth = self.oversold - k
            conf  = min(0.88, 0.60 + depth / self.oversold * 0.3)
            return {"signal": "BUY", "confidence": round(conf, 3), "agent": self.name}
        if k > self.overbought and d > self.overbought and k < d:
            height = k - self.overbought
            conf   = min(0.88, 0.60 + height / (100 - self.overbought) * 0.3)
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class IchimokuAgent:
    name = "Ichimoku"

    def analyze(self, ind: dict, candles: list) -> dict:
        ichi  = ind["ichimoku"]
        if not ichi:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        price     = ind["prices"][-1]
        cloud_top = max(ichi["senkou_a"], ichi["senkou_b"])
        cloud_bot = min(ichi["senkou_a"], ichi["senkou_b"])
        tk_bull   = ichi["tenkan"] > ichi["kijun"]
        tk_bear   = ichi["tenkan"] < ichi["kijun"]

        if price > cloud_top and tk_bull:
            dist_pct = (price - cloud_top) / cloud_top
            conf     = min(0.85, 0.55 + dist_pct * 20)
            return {"signal": "BUY", "confidence": round(conf, 3), "agent": self.name}
        if price < cloud_bot and tk_bear:
            dist_pct = (cloud_bot - price) / cloud_bot
            conf     = min(0.85, 0.55 + dist_pct * 20)
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class OrderBookAgent:
    name = "OrderBook"

    def __init__(self, imbalance_threshold=1.5):
        self.threshold = imbalance_threshold

    def analyze(self, ind: dict, candles: list, bids=0.0, asks=0.0) -> dict:
        if bids == 0 or asks == 0:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        ratio = bids / (asks + 1e-9)
        if ratio > self.threshold:
            conf = min(0.85, 0.55 + (ratio - self.threshold) * 0.15)
            return {"signal": "BUY",  "confidence": round(conf, 3), "agent": self.name}
        if ratio < 1 / self.threshold:
            conf = min(0.85, 0.55 + (1 / ratio - self.threshold) * 0.15)
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class VWAPAgent:
    name = "VWAP"

    def analyze(self, ind: dict, candles: list) -> dict:
        if len(candles) < 20:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        typical = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles]
        volumes = [c["volume"] for c in candles]
        vwap    = sum(t * v for t, v in zip(typical, volumes)) / (sum(volumes) + 1e-9)
        price   = candles[-1]["close"]
        dev_pct = (price - vwap) / (vwap + 1e-9)
        if dev_pct < -0.012:
            conf = min(0.85, 0.55 + abs(dev_pct) * 20)
            return {"signal": "BUY",  "confidence": round(conf, 3), "agent": self.name}
        if dev_pct > 0.012:
            conf = min(0.85, 0.55 + dev_pct * 20)
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class CrossAssetAgent:
    name = "CrossAsset"
    BASE_ASSETS = {"BTCUSDT", "ETHUSDT"}

    def __init__(self):
        self._btc_trend = "flat"
        self._eth_trend = "flat"
        self._last_fetch = 0
        self._cache_ttl = 60

    def _fetch_klines(self, symbol: str, limit: int = 30) -> list:
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit={limit}"
            r   = requests.get(url, timeout=5)
            if r.status_code == 200:
                return [{"close": float(k[4])} for k in r.json()]
        except Exception:
            pass
        return []

    def _refresh(self):
        if time.time() - self._last_fetch < self._cache_ttl:
            return
        self._last_fetch = time.time()
        for symbol, attr in [("BTCUSDT", "_btc_trend"), ("ETHUSDT", "_eth_trend")]:
            data = self._fetch_klines(symbol)
            if data:
                prices = [c["close"] for c in data]
                e9, e21 = ema(prices, 9), ema(prices, 21)
                trend = "up" if e9 > e21 * 1.003 else "down" if e9 < e21 * 0.997 else "flat"
                setattr(self, attr, trend)

    def analyze(self, ind: dict, candles: list, symbol: str = "") -> dict:
        if symbol in self.BASE_ASSETS:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        self._refresh()
        btc, eth = self._btc_trend, self._eth_trend
        if btc == "up"   and eth == "up":
            return {"signal": "BUY",  "confidence": 0.68, "agent": self.name}
        if btc == "down" and eth == "down":
            return {"signal": "SELL", "confidence": 0.68, "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class MeanReversionAgent:
    name = "MeanReversion"

    def __init__(self, lookback=20, zscore_thresh=2.0, rsi_oversold=28, rsi_overbought=72, bb_exit=0.15):
        self.lookback        = lookback
        self.zscore_thresh   = zscore_thresh
        self.rsi_oversold    = rsi_oversold
        self.rsi_overbought  = rsi_overbought
        self.bb_exit         = bb_exit

    def analyze(self, ind: dict, candles: list) -> dict:
        if len(ind["prices"]) < self.lookback + 5:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        rsi_val = ind["rsi14"]
        zscore  = ind.get("zscore_20", 0.0)
        bb_pos  = ind.get("bb_position", 0.5)
        vol_ok  = ind["volumes"][-1] > ind["avg_vol20"] * 0.6

        if zscore < -self.zscore_thresh and rsi_val < self.rsi_oversold and vol_ok:
            strength = min(1.0, abs(zscore) / (self.zscore_thresh * 2))
            conf = 0.60 + strength * 0.20
            if bb_pos < self.bb_exit:
                conf += 0.05
            return {"signal": "BUY", "confidence": round(min(0.85, conf), 3), "agent": self.name,
                    "meta": {"zscore": round(zscore, 2), "rsi": round(rsi_val, 1)}}

        if zscore > self.zscore_thresh and rsi_val > self.rsi_overbought and vol_ok:
            strength = min(1.0, zscore / (self.zscore_thresh * 2))
            conf = 0.60 + strength * 0.20
            if bb_pos > (1 - self.bb_exit):
                conf += 0.05
            return {"signal": "SELL", "confidence": round(min(0.85, conf), 3), "agent": self.name,
                    "meta": {"zscore": round(zscore, 2), "rsi": round(rsi_val, 1)}}

        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class TrendBreakoutAgent:
    name = "TrendBreakout"

    def __init__(self, lookback=20, long_ma=200, vol_spike=2.0, adx_thresh=30,
                 use_bollinger=True, bb_std=2, long_only=True):
        self.lookback     = lookback
        self.long_ma      = long_ma
        self.vol_spike    = vol_spike
        self.adx_thresh   = adx_thresh
        self.use_bollinger = use_bollinger
        self.bb_std       = bb_std
        self.long_only    = long_only

    def analyze(self, ind: dict, candles: list) -> dict:
        if len(candles) < self.lookback + 5:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        closes, volumes = ind["prices"], ind["volumes"]
        sma20  = np.mean(closes[-self.lookback:])
        sma200 = np.mean(closes[-self.long_ma:]) if len(closes) >= self.long_ma else sma20
        avg_vol = np.mean(volumes[-self.lookback:])
        vol_ok  = volumes[-1] > avg_vol * self.vol_spike
        adx_ok  = ind.get("adx14", 0) >= self.adx_thresh
        if not (vol_ok and adx_ok):
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        bb_std_val = np.std(closes[-self.lookback:]) if len(closes) >= self.lookback else 0
        bb_upper   = sma20 + self.bb_std * bb_std_val
        bb_lower   = sma20 - self.bb_std * bb_std_val
        price = closes[-1]
        if price > bb_upper and price > sma200:
            excess = (price - bb_upper) / (bb_upper + 1e-9)
            conf   = min(0.85, 0.58 + excess * 10)
            return {"signal": "BUY", "confidence": round(conf, 3), "agent": self.name,
                    "meta": {"adx": round(ind["adx14"], 1), "vol_ratio": round(volumes[-1]/avg_vol, 2)}}
        if not self.long_only and price < bb_lower and price < sma200:
            excess = (bb_lower - price) / (price + 1e-9)
            conf   = min(0.85, 0.58 + excess * 10)
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class NoiseScalperAgent:
    name = "NoiseScalper"

    def __init__(self, lookback=20, range_pct=1.5, adx_max=20,
                 profit_target=0.4, stop_loss=0.2, min_vol_ratio=0.8, max_hold_bars=3):
        self.lookback      = lookback
        self.range_pct     = range_pct
        self.adx_max       = adx_max
        self.profit_target = profit_target
        self.stop_loss     = stop_loss
        self.min_vol_ratio = min_vol_ratio
        self.max_hold_bars = max_hold_bars

    def analyze(self, ind: dict, candles: list) -> dict:
        if ind.get("adx14", 50) >= self.adx_max:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        if len(candles) < self.lookback + 2:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        recent_high = max(c["high"] for c in candles[-self.lookback:])
        recent_low  = min(c["low"]  for c in candles[-self.lookback:])
        price       = candles[-1]["close"]
        range_width = (recent_high - recent_low) / (price + 1e-9) * 100
        if range_width > self.range_pct:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        vol_ratio = ind["volumes"][-1] / (ind["avg_vol20"] + 1e-9)
        if vol_ratio < self.min_vol_ratio:
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        position = (price - recent_low) / (recent_high - recent_low + 1e-9)
        if position < 0.20:
            return {"signal": "BUY",  "confidence": round(min(0.78, 0.65 + (0.20 - position) * 0.5), 3), "agent": self.name}
        if position > 0.80:
            return {"signal": "SELL", "confidence": round(min(0.78, 0.65 + (position - 0.80) * 0.5), 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class OrderFlowAgent:
    name = "OrderFlow"

    def __init__(self, lookback_trades=100, cache_ttl=30):
        self.lookback  = lookback_trades
        self.cache_ttl = cache_ttl
        self._cache    = {}

    def _fetch_trades(self, symbol: str) -> Optional[list]:
        try:
            url = f"https://fapi.binance.com/fapi/v1/trades?symbol={symbol}&limit={self.lookback}"
            r   = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def get_imbalance(self, symbol: str) -> float:
        now    = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached[1] < self.cache_ttl:
            return cached[0]
        trades = self._fetch_trades(symbol)
        if not trades:
            return 0.0
        buy_vol  = sum(float(t["qty"]) * float(t["price"]) for t in trades if not t.get("isBuyerMaker", True))
        sell_vol = sum(float(t["qty"]) * float(t["price"]) for t in trades if t.get("isBuyerMaker", True))
        total    = buy_vol + sell_vol
        imb      = (buy_vol - sell_vol) / (total + 1e-9)
        self._cache[symbol] = (imb, now)
        return imb

    def analyze(self, ind: dict, candles: list, symbol: str = "") -> dict:
        imb = self.get_imbalance(symbol)
        if imb > 0.30:
            conf = min(0.85, 0.58 + imb * 0.5)
            return {"signal": "BUY",  "confidence": round(conf, 3), "agent": self.name}
        if imb < -0.25:
            conf = min(0.85, 0.58 + abs(imb) * 0.5)
            return {"signal": "SELL", "confidence": round(conf, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class LLMSentimentAgent:
    name = "LLMSentiment"

    def __init__(self):
        self.url       = OLLAMA_URL
        self.model     = AI_MODEL
        self.temp      = AI_TEMPERATURE
        self._sem      = threading.Semaphore(LLM_SEMAPHORE)
        self._last     = {}
        self._cooldown = LLM_COOLDOWN

    def get_sentiment(self, symbol: str) -> float:
        now = time.time()
        if symbol in self._last and now - self._last[symbol] < self._cooldown:
            return 0.5
        self._last[symbol] = now
        prompt = (
            f"Crypto pair: {symbol}. Output ONLY a number 0-1 representing current market sentiment. "
            f"0=extremely bearish, 0.5=neutral, 1=extremely bullish. Just the number, nothing else."
        )
        acquired = self._sem.acquire(timeout=15)
        if not acquired:
            return 0.5
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": {"temperature": self.temp, "num_predict": 5}},
                timeout=LLM_TIMEOUT
            )
            if resp.status_code == 200:
                return max(0.0, min(1.0, float(resp.json()["response"].strip())))
        except Exception:
            pass
        finally:
            self._sem.release()
        return 0.5

    def analyze(self, ind: dict, candles: list, symbol: str = "") -> dict:
        s = self.get_sentiment(symbol)
        if s > 0.70:
            return {"signal": "BUY",  "confidence": round(0.55 + (s - 0.5) * 1.0, 3), "agent": self.name,
                    "meta": {"sentiment": round(s, 3)}}
        if s < 0.30:
            return {"signal": "SELL", "confidence": round(0.55 + (0.5 - s) * 1.0, 3), "agent": self.name,
                    "meta": {"sentiment": round(s, 3)}}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name, "meta": {"sentiment": round(s, 3)}}


# ════════════════════════════════════════════════════════════════════════════
#  LSTM AGENT
# ════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm    = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(self.dropout(out[:, -1, :]))).squeeze(-1)

    class LSTMAgent:
        name = "LSTM"
        SEQ_LEN    = LSTM_SEQ_LEN
        INPUT_FEATS = 12
        HIDDEN     = LSTM_HIDDEN_SIZE
        TRAIN_EVERY = LSTM_TRAIN_EVERY
        MIN_SAMPLES = LSTM_MIN_SAMPLES

        def __init__(self, symbol: str, save_path: str = "logs/models"):
            self.symbol     = symbol
            self.save_path  = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.model_file = self.save_path / f"LSTM_{symbol}.pt"
            self._X_buf     = deque(maxlen=500)
            self._y_buf     = deque(maxlen=500)
            self._model     = _LSTMNet(self.INPUT_FEATS, self.HIDDEN)
            self._trained   = False
            self._update_count = 0
            self._lock      = threading.Lock()
            self._total = self._wins = 0
            self._try_load()

        def _try_load(self):
            if self.model_file.exists():
                try:
                    self._model.load_state_dict(torch.load(self.model_file, map_location="cpu"))
                    self._model.eval()
                    self._trained = True
                except Exception as e:
                    log.warning(f"[LSTM] Load failed {self.symbol}: {e}")

        def _extract_features(self, ind: dict, candles: list) -> Optional[np.ndarray]:
            if len(candles) < self.SEQ_LEN + 2:
                return None
            window = candles[-(self.SEQ_LEN + 1):]
            prices  = [c["close"]  for c in window]
            volumes = [c["volume"] for c in window]
            s = pd.Series(prices, dtype=float)
            e9  = s.ewm(span=9,  adjust=False).mean()
            e21 = s.ewm(span=21, adjust=False).mean()
            gain = s.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-s.diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rsi_s  = 100 - 100 / (1 + gain / (loss + 1e-9))
            vol_ma = pd.Series(volumes, dtype=float).rolling(14).mean()
            feats = []
            for i in range(1, len(window)):
                p, pp = prices[i], prices[i - 1]
                c = window[i]
                ret = (p - pp) / (pp + 1e-9)
                feats.append([
                    ret,
                    (c["high"] - c["low"])  / (pp + 1e-9),
                    (c["close"] - c["open"]) / (pp + 1e-9),
                    (c["high"] - max(c["open"], c["close"])) / (pp + 1e-9),
                    (min(c["open"], c["close"]) - c["low"])  / (pp + 1e-9),
                    (p - float(e9.iloc[i]))  / (float(e9.iloc[i])  + 1e-9),
                    (p - float(e21.iloc[i])) / (float(e21.iloc[i]) + 1e-9),
                    float(rsi_s.iloc[i]) / 100.0,
                    volumes[i] / (float(vol_ma.iloc[i]) + 1e-9) - 1.0,
                    (float(e9.iloc[i]) - float(e21.iloc[i])) / (p + 1e-9),
                    ret - ((pp - prices[i - 2]) / (prices[i - 2] + 1e-9) if i > 1 else 0),
                    np.clip(ret * 100, -5, 5),
                ])
            arr = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0, posinf=1, neginf=-1)
            arr = np.clip(arr, -10, 10)
            return arr if arr.shape[0] == self.SEQ_LEN else None

        def update(self, ind: dict, candles: list, next_candle: dict):
            feats = self._extract_features(ind, candles)
            if feats is None:
                return
            target = 1.0 if next_candle["close"] > candles[-1]["close"] else 0.0
            with self._lock:
                self._X_buf.append(feats)
                self._y_buf.append(target)
                self._update_count += 1

        def train(self):
            with self._lock:
                if len(self._X_buf) < self.MIN_SAMPLES:
                    return
                X = np.stack(list(self._X_buf))
                y = np.array(list(self._y_buf), dtype=np.float32)
            X_t, y_t = torch.tensor(X), torch.tensor(y)
            model   = _LSTMNet(self.INPUT_FEATS, self.HIDDEN)
            opt     = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            loss_fn = nn.BCELoss()
            model.train()
            for _ in range(30):
                idx = torch.randperm(len(X_t))
                for start in range(0, len(X_t), 32):
                    b = idx[start:start + 32]
                    opt.zero_grad()
                    l = loss_fn(model(X_t[b]), y_t[b])
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()
            with self._lock:
                self._model   = model
                self._trained = True
            try:
                torch.save(model.state_dict(), self.model_file)
            except Exception:
                pass

        def predict(self, ind: dict, candles: list) -> float:
            if not self._trained:
                e9, e21 = ind.get("ema9", 0), ind.get("ema21", 0)
                return 0.60 if e9 > e21 else 0.40
            feats = self._extract_features(ind, candles)
            if feats is None:
                return 0.50
            with torch.no_grad():
                return float(self._model(torch.tensor(feats).unsqueeze(0)).item())

        def record_outcome(self, win: bool):
            self._total += 1
            if win:
                self._wins += 1

        def analyze(self, ind: dict, candles: list) -> dict:
            prob = self.predict(ind, candles)
            if prob > 0.65:
                return {"signal": "BUY",  "confidence": round((prob - 0.5) * 2, 3), "agent": self.name}
            if prob < 0.35:
                return {"signal": "SELL", "confidence": round((0.5 - prob) * 2, 3), "agent": self.name}
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}

else:
    class LSTMAgent:
        name = "LSTM"
        def __init__(self, symbol: str, **kw):
            self.symbol  = symbol
            self._total  = self._wins = 0
            self._update_count = 0
        def record_outcome(self, win: bool):
            self._total += 1
            if win: self._wins += 1
        def analyze(self, ind: dict, candles: list) -> dict:
            e9, e21 = ind.get("ema9", 0), ind.get("ema21", 0)
            if e9 > e21 * 1.002:
                return {"signal": "BUY",  "confidence": 0.62, "agent": self.name}
            if e9 < e21 * 0.998:
                return {"signal": "SELL", "confidence": 0.62, "agent": self.name}
            return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}
        def update(self, ind, candles, next_candle): pass
        def train(self): pass
        def predict(self, ind, candles): return 0.5


# ════════════════════════════════════════════════════════════════════════════
#  ML AGENTS — FIXED: Train on actual trade PnL, not next-candle direction
# ════════════════════════════════════════════════════════════════════════════
class BaseMLAgent:
    FEATURE_COUNT = 11

    def __init__(self, symbol: str, train_interval: int, feature_window: int, min_samples: int):
        self.symbol         = symbol
        self.train_interval = train_interval
        self.feature_window = feature_window
        self.min_samples    = min_samples
        self.features = deque(maxlen=feature_window)
        self.targets  = deque(maxlen=feature_window)
        self.model    = None
        self.scaler   = StandardScaler() if ML_AVAILABLE else None
        self.trained  = False
        self.update_count = 0
        self.model_path  = Path(ML_SAVE_PATH) / f"{self.name}_{symbol}.joblib"
        self.scaler_path = Path(ML_SAVE_PATH) / f"{self.name}_{symbol}_scaler.joblib"
        self._load_model()

    def _load_model(self):
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                self.model  = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.FEATURE_COUNT:
                    log.warning(f"[{self.name}] Feature count mismatch — ignoring saved model")
                    self.trained = False
                    self.model = None
                    self.scaler = StandardScaler()
                    return
                self.trained = True
            except Exception:
                self.trained = False

    def _save_model(self):
        if self.model and self.scaler:
            try:
                joblib.dump(self.model,  self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
            except Exception:
                pass

    def _compute_features(self, ind: dict, candles: list) -> np.ndarray:
        prices  = ind.get("prices", [])
        volumes = ind.get("volumes", [])
        # Safe handling for empty lists
        if len(prices) < 11 or len(volumes) < 20:
            return np.zeros(self.FEATURE_COUNT)
        def ret(n): return (prices[-1] - prices[-n]) / (prices[-n] + 1e-9) if len(prices) > n else 0
        return np.array([
            ret(2), ret(6), ret(11),
            ind["atr14"] / (prices[-1] + 1e-9) if prices[-1] != 0 else 0,
            volumes[-1] / (ind["avg_vol20"] + 1e-9) if ind.get("avg_vol20", 0) != 0 else 1.0,
            ind.get("rsi14", 50) / 100.0,
            ind.get("stoch_k", 50) / 100.0,
            ind.get("adx14", 0) / 100.0,
            (prices[-1] - ind.get("ema20", prices[-1])) / (ind.get("ema20", prices[-1]) + 1e-9) if ind.get("ema20", 0) != 0 else 0,
            ind.get("zscore_20", 0) / 4.0,
            ind.get("bb_position", 0.5) - 0.5,
        ])

    def _create_model(self): raise NotImplementedError

    def update_with_trade_outcome(self, current_ind: dict, current_candles: list, trade_pnl: float):
        """
        Train on actual trade profitability, not next-candle direction.
        trade_pnl > 0 → target = 1 (win), else 0 (loss)
        """
        if not ML_AVAILABLE: return
        features = self._compute_features(current_ind, current_candles)
        if features is None:
            return
        target = 1 if trade_pnl > 0 else 0
        self.features.append(features)
        self.targets.append(target)
        self.update_count += 1

    def train(self):
        if not ML_AVAILABLE or len(self.features) < self.min_samples: return
        X = np.array(list(self.features))
        y = np.array(list(self.targets))
        Xs = self.scaler.fit_transform(X)
        self.model = self._create_model()
        if self.model is None: return
        self.model.fit(Xs, y)
        self.trained = True
        self._save_model()

    def predict(self, ind: dict, candles: list) -> float:
        if not ML_AVAILABLE or not self.trained:
            prices = ind.get("prices", [])
            if prices:
                return 0.60 if prices[-1] > ind.get("ema20", prices[-1]) else 0.40
            return 0.50
        features = self._compute_features(ind, candles)
        if features is None:
            return 0.50
        Xs = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict_proba(Xs)[0][1]) if hasattr(self.model, "predict_proba") else float(self.model.predict(Xs)[0])

    def analyze(self, ind: dict, candles: list) -> dict:
        prob = self.predict(ind, candles)
        if prob > 0.62:
            return {"signal": "BUY",  "confidence": round((prob - 0.5) * 2, 3), "agent": self.name}
        if prob < 0.38:
            return {"signal": "SELL", "confidence": round((0.5 - prob) * 2, 3), "agent": self.name}
        return {"signal": "HOLD", "confidence": 0.0, "agent": self.name}


class ElasticNetAgent(BaseMLAgent):
    name = "ElasticNet"
    def _create_model(self):
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1000, C=1.0) if ML_AVAILABLE else None

class XGBoostAgent(BaseMLAgent):
    name = "XGBoost"
    def _create_model(self):
        return xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=3, learning_rate=0.1) if XGB_AVAILABLE else None

class RandomForestAgent(BaseMLAgent):
    name = "RandomForest"
    def _create_model(self):
        return RandomForestClassifier(n_estimators=100, max_depth=5) if RF_AVAILABLE else None


# ════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
class AdaptiveEnsemble:
    def __init__(self, agents, decay=0.97, window=50, weight_floor=0.05):
        self.agents      = agents
        self.decay       = decay
        self.weight_floor = weight_floor
        self.weights     = {a.name: 1.0 / len(agents) for a in agents}
        self.performance = {a.name: deque(maxlen=window) for a in agents}

    def _ensure_agent(self, name: str):
        if name not in self.performance:
            self.weights[name]     = self.weight_floor
            self.performance[name] = deque(maxlen=50)

    def update_weight(self, agent_name: str, correct: bool):
        self._ensure_agent(agent_name)
        self.performance[agent_name].append(1 if correct else 0)
        perf     = self.performance[agent_name]
        win_rate = sum(perf) / len(perf) if perf else 0.5
        self.weights[agent_name] = self.weights[agent_name] * self.decay + win_rate * (1 - self.decay)
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def vote(self, signals: list) -> dict:
        buy_score = sell_score = 0.0
        for sig in signals:
            if sig is None:
                continue
            w = self.weights.get(sig["agent"], self.weight_floor)
            if sig["signal"] == "BUY":
                buy_score  += w * sig["confidence"]
            elif sig["signal"] == "SELL":
                sell_score += w * sig["confidence"]
        total = buy_score + sell_score
        if total < 1e-9:
            return {"signal": "HOLD", "confidence": 0.0}
        buy_prob  = buy_score  / total
        sell_prob = sell_score / total
        if buy_prob  > 0.5: return {"signal": "BUY",  "confidence": buy_prob}
        if sell_prob > 0.5: return {"signal": "SELL", "confidence": sell_prob}
        return {"signal": "HOLD", "confidence": max(buy_prob, sell_prob)}


# ════════════════════════════════════════════════════════════════════════════
#  META LEARNER
# ════════════════════════════════════════════════════════════════════════════
class MetaLearner:
    def __init__(self, feature_names: List[str], retrain_every: int = 30, min_samples: int = 100):
        self.feature_names = feature_names
        self.retrain_every = retrain_every
        self.min_samples   = min_samples
        self.X_buf  = deque(maxlen=500)
        self.y_buf  = deque(maxlen=500)
        self.model  = LogisticRegression(class_weight="balanced", max_iter=1000) if ML_AVAILABLE else None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self.update_counter = 0
        self._lock  = threading.Lock()
        self._load_model()

    def _load_model(self):
        if META_MODEL_PATH.exists() and META_SCALER_PATH.exists() and ML_AVAILABLE:
            try:
                self.model   = joblib.load(META_MODEL_PATH)
                self.scaler  = joblib.load(META_SCALER_PATH)
                self.trained = True
                log.info("[MetaLearner] Loaded saved model")
            except Exception as e:
                log.warning(f"[MetaLearner] Load failed: {e}")

    def _save_model(self):
        if self.model and self.scaler:
            try:
                joblib.dump(self.model,  META_MODEL_PATH)
                joblib.dump(self.scaler, META_SCALER_PATH)
            except Exception:
                pass

    def _extract_features(self, analysis_result: dict, ensemble_decision: dict,
                           agent_signals: list, lstm_agent=None, llm_sentiment: float = 0.5) -> List[float]:
        regime_map = {"TRENDING": 1, "RANGING": 2, "VOLATILE": 3}
        ri   = analysis_result.get("regime_info", {})
        ind  = analysis_result.get("ind", {})
        price = analysis_result.get("price", 1.0)
        buy_w = sell_w = 0.0
        for s in agent_signals:
            w = 1.0 / max(len(agent_signals), 1)
            if s.get("signal") == "BUY":   buy_w  += w * s.get("confidence", 0)
            elif s.get("signal") == "SELL": sell_w += w * s.get("confidence", 0)
        tot = buy_w + sell_w
        lstm_prob = lstm_agent.predict(ind, analysis_result.get("candles", [])) if lstm_agent else 0.5
        return [
            ensemble_decision.get("confidence", 0),
            buy_w  / (tot + 1e-9),
            sell_w / (tot + 1e-9),
            regime_map.get(ri.get("regime", "RANGING"), 2) / 3.0,
            ind.get("adx14", 0) / 100.0,
            min(ind.get("atr14", 0) / (price + 1e-9), 0.1) * 10,
            1 if analysis_result.get("btc_trend") == "up" else (-1 if analysis_result.get("btc_trend") == "down" else 0),
            1 if analysis_result.get("eth_trend") == "up" else (-1 if analysis_result.get("eth_trend") == "down" else 0),
            min(analysis_result.get("consecutive_losses", 0), 5) / 5.0,
            datetime.utcnow().hour / 24.0,
            np.mean([s.get("confidence", 0) for s in agent_signals]) if agent_signals else 0.5,
            sum(1 for s in agent_signals if s.get("signal") == "BUY")  / max(len(agent_signals), 1),
            sum(1 for s in agent_signals if s.get("signal") == "SELL") / max(len(agent_signals), 1),
            lstm_prob,
            llm_sentiment,
        ]

    def update(self, analysis_result, ensemble_decision, agent_signals, actual_win, lstm_agent=None, llm_sentiment=0.5):
        if not ML_AVAILABLE: return
        feat = self._extract_features(analysis_result, ensemble_decision, agent_signals, lstm_agent, llm_sentiment)
        with self._lock:
            self.X_buf.append(feat)
            self.y_buf.append(1 if actual_win else 0)
            self.update_counter += 1
        if self.update_counter >= self.retrain_every and len(self.X_buf) >= self.min_samples:
            self.train()

    def train(self):
        if not ML_AVAILABLE or len(self.X_buf) < self.min_samples: return
        with self._lock:
            X, y = np.array(list(self.X_buf)), np.array(list(self.y_buf))
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.trained = True
        self._save_model()
        log.info(f"[MetaLearner] Trained on {len(X)} samples  acc={self.model.score(Xs, y):.2f}")

    def predict(self, analysis_result, ensemble_decision, agent_signals, lstm_agent=None, llm_sentiment=0.5):
        if not ML_AVAILABLE or not self.trained:
            return None
        feat = np.array(self._extract_features(analysis_result, ensemble_decision, agent_signals, lstm_agent, llm_sentiment)).reshape(1, -1)
        prob = float(self.model.predict_proba(self.scaler.transform(feat))[0][1])
        thr  = META_CONFIDENCE_THRESHOLD
        if prob > thr:       return "BUY",  prob
        if prob < 1 - thr:   return "SELL", 1 - prob
        return None


# ════════════════════════════════════════════════════════════════════════════
#  SELECTIVE ENTRY GATE
# ════════════════════════════════════════════════════════════════════════════
class SelectiveEntryGate:
    def __init__(self, min_agreeing_agents=4, min_confidence=0.62, max_conflict_conf=0.55, min_regime_conf=0.55):
        self.min_agreeing   = min_agreeing_agents
        self.min_confidence = min_confidence
        self.max_conflict   = max_conflict_conf
        self.min_regime_conf = min_regime_conf
        self._stats = {"passed": 0, "blocked": 0, "reasons": {}}

    def _block(self, reason):
        self._stats["blocked"] += 1
        self._stats["reasons"][reason] = self._stats["reasons"].get(reason, 0) + 1

    def check(self, signals, decision, regime_info) -> Tuple[bool, str]:
        signal = decision.get("signal", "HOLD")
        conf   = decision.get("confidence", 0.0)
        if signal == "HOLD":
            return False, "hold_signal"
        if conf < self.min_confidence:
            self._block("low_confidence"); return False, f"conf_{conf:.2f}"
        agreeing = sum(1 for s in signals if s.get("signal") == signal and s.get("confidence", 0) > 0.55)
        if agreeing < self.min_agreeing:
            self._block("insufficient_agreement"); return False, f"{agreeing}_agree_need_{self.min_agreeing}"
        opposite = "SELL" if signal == "BUY" else "BUY"
        conflict  = [s.get("confidence", 0) for s in signals if s.get("signal") == opposite and s.get("confidence", 0) > self.max_conflict]
        if conflict:
            self._block("strong_conflict"); return False, f"conflict_{max(conflict):.2f}"
        if regime_info.get("confidence", 1.0) < self.min_regime_conf:
            self._block("low_regime_conf"); return False, "low_regime_conf"
        self._stats["passed"] += 1
        return True, "passed"

    def stats(self) -> dict:
        total = self._stats["passed"] + self._stats["blocked"]
        return {
            "total": total,
            "passed": self._stats["passed"],
            "blocked": self._stats["blocked"],
            "pass_rate": round(self._stats["passed"] / max(1, total), 3),
            "top_reasons": sorted(self._stats["reasons"].items(), key=lambda x: -x[1])[:5],
        }


# ════════════════════════════════════════════════════════════════════════════
#  SESSION FILTER
# ════════════════════════════════════════════════════════════════════════════
class SessionFilter:
    SESSIONS = [
        (7,  10, "london_open",    1.3),
        (10, 13, "london_midday",  1.0),
        (13, 17, "us_open",        1.4),
        (17, 20, "us_afternoon",   1.1),
        (20, 23, "us_close",       0.9),
        (23,  1, "overnight",      0.7),
        (1,   6, "asian_dead",     0.5),
        (6,   7, "pre_london",     0.8),
    ]
    BLOCK_BELOW = 0.70

    def get_current_session(self) -> Tuple[str, float]:
        h = pd.Timestamp.utcnow().hour
        for start, end, name, mult in self.SESSIONS:
            if start < end:
                if start <= h < end: return name, mult
            else:
                if h >= start or h < end: return name, mult
        return "unknown", 1.0

    def allows_entry(self, symbol: str = "") -> Tuple[bool, str, float]:
        session, mult = self.get_current_session()
        return mult >= self.BLOCK_BELOW, session, mult

    def adjust_confidence(self, conf: float) -> float:
        _, mult = self.get_current_session()
        return min(0.95, conf * mult)

    def get_max_hold_bars(self) -> int:
        _, mult = self.get_current_session()
        if mult >= 1.2: return MAX_HOLD_BARS_NORMAL
        if mult >= 0.9: return MAX_HOLD_BARS_VOLATILE
        return MAX_HOLD_BARS_NEWS


# ════════════════════════════════════════════════════════════════════════════
#  MARKET REGIME
# ════════════════════════════════════════════════════════════════════════════
def detect_regime(candles: list) -> str:
    if len(candles) < 14:
        return "RANGING"
    adx_val = adx(candles, 14)
    if adx_val > REGIME_ADX_TRENDING:
        return "TRENDING"
    atr_val = atr(candles, 14)
    if atr_val / (candles[-1]["close"] + 1e-9) > REGIME_VOLA_THRESHOLD:
        return "VOLATILE"
    return "RANGING"


# ════════════════════════════════════════════════════════════════════════════
#  RISK MANAGER
# ════════════════════════════════════════════════════════════════════════════
class RiskManager:
    def __init__(self):
        self.atr_stop_mult       = ATR_STOP_MULT
        self.atr_take_mult       = ATR_TAKE_MULT
        self.partial_take_mult   = PARTIAL_TAKE_MULT
        self.trailing_activation = TRAILING_ACTIVATION
        self.trailing_retreat    = TRAILING_RETREAT
        self.consecutive_losses  = 0
        self._consecutive_holds  = 0

    def position_size(self, balance_usdt: float, confidence: float, atr_val: float,
                      price: float, volume_ratio: float = 1.0) -> float:
        base        = balance_usdt * MAX_POSITION_PCT
        conf_scale  = 0.75 + 0.25 * (confidence - 0.55) / 0.45
        atr_pct     = atr_val / (price + 1e-9)
        atr_scale   = min(1.0, 0.015 / (atr_pct + 0.001))
        loss_scale  = 1.0 / (1 + self.consecutive_losses * 0.5)
        size        = base * conf_scale * atr_scale * loss_scale
        size        = min(size, balance_usdt * 0.20)
        return max(MIN_TRADE_USDT, size)

    def check_position(self, entry: float, current: float, direction: str,
                       rsi_val: float, atr_val: float, ema9: float,
                       partial_taken: bool = False, hold_bars: int = 0,
                       trade_confidence: float = 0.65, session_filter=None,
                       lowest_price=None, highest_price=None) -> dict:
        pct    = ((current - entry) / entry * 100) if direction == "BUY" else ((entry - current) / entry * 100)
        sl_pct = STOP_LOSS_PCT if STOP_LOSS_PCT else 1.5
        tp_pct = TAKE_PROFIT_PCT if TAKE_PROFIT_PCT else 3.5
        max_bars = session_filter.get_max_hold_bars() if session_filter else MAX_HOLD_BARS

        hard_stop = HARD_STOP_PCT_HIGH_CONF if trade_confidence >= 0.70 else HARD_STOP_PCT_MED_CONF if trade_confidence >= 0.55 else HARD_STOP_PCT_LOW_CONF
        if pct <= -hard_stop:
            return {"action": "CLOSE", "reason": "hard_stop", "pnl_pct": pct}

        if direction == "BUY" and lowest_price is not None:
            mae = (lowest_price - entry) / entry * 100
            if mae <= -MAX_ADVERSE_EXCURSION_PCT:
                return {"action": "CLOSE", "reason": "mae", "pnl_pct": pct}
        if direction == "SELL" and highest_price is not None:
            mae = (entry - highest_price) / entry * 100
            if mae <= -MAX_ADVERSE_EXCURSION_PCT:
                return {"action": "CLOSE", "reason": "mae", "pnl_pct": pct}

        stop_dist = self.atr_stop_mult * atr_val
        take_dist = self.atr_take_mult * atr_val
        if direction == "BUY":
            if current <= entry - stop_dist:   return {"action": "CLOSE",   "reason": "atr_stop", "pnl_pct": pct}
            if current >= entry + take_dist:    return {"action": "CLOSE",   "reason": "atr_take", "pnl_pct": pct}
            if not partial_taken and current >= entry + take_dist * self.partial_take_mult:
                return {"action": "PARTIAL", "reason": "partial", "pnl_pct": pct}
        else:
            if current >= entry + stop_dist:    return {"action": "CLOSE",   "reason": "atr_stop", "pnl_pct": pct}
            if current <= entry - take_dist:    return {"action": "CLOSE",   "reason": "atr_take", "pnl_pct": pct}
            if not partial_taken and current <= entry - take_dist * self.partial_take_mult:
                return {"action": "PARTIAL", "reason": "partial", "pnl_pct": pct}

        if pct <= -sl_pct: return {"action": "CLOSE", "reason": "pct_stop", "pnl_pct": pct}
        if pct >= tp_pct:  return {"action": "CLOSE", "reason": "pct_take", "pnl_pct": pct}

        if hold_bars >= max_bars:
            return {"action": "CLOSE", "reason": "timeout", "pnl_pct": pct}

        return {"action": "HOLD", "reason": "none", "pnl_pct": pct}


# ════════════════════════════════════════════════════════════════════════════
#  DATA FEED
# ════════════════════════════════════════════════════════════════════════════
def get_top_volume_pairs(limit=TOP_PAIRS_BY_VOLUME, min_volume_usdt=MIN_24H_VOLUME_USDT) -> List[str]:
    try:
        client  = Client(API_KEY, API_SECRET, testnet=True)
        tickers = client.futures_ticker()
        pairs   = [(t["symbol"], float(t["quoteVolume"])) for t in tickers
                   if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) >= min_volume_usdt]
        pairs.sort(key=lambda x: x[1], reverse=True)
        result = [p[0] for p in pairs[:limit]]
        log.info(f"Top {len(result)} pairs by volume")
        return result
    except Exception as e:
        log.error(f"Pair fetch failed: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


class DataFeed:
    LOT_STEPS = {
        "BTCUSDT": 0.001, "ETHUSDT": 0.01, "BNBUSDT": 0.1, "SOLUSDT": 0.1,
        "XRPUSDT": 1.0, "ADAUSDT": 1.0, "DOGEUSDT": 10.0, "LTCUSDT": 0.001,
    }

    def __init__(self, mode: str = "live"):
        self.mode      = mode
        self.simulated = True
        self._sim_prices = {}
        self._historical_data = {}
        self.client       = None
        self.trade_manager = None
        if mode == "live" and BINANCE_OK:
            try:
                self.client = Client(API_KEY, API_SECRET, testnet=True)
                self.client.futures_base_url = "https://testnet.binancefuture.com/fapi"
                self.client.futures_testnet  = True
                self.simulated = False
                log.info("Connected: Binance Futures Testnet")
                self._sync_time()
                self._prefetch_step_sizes()
            except Exception as e:
                log.warning(f"Binance connect failed ({e}) — simulation mode")

    def _sync_time(self):
        try:
            st = self.client.futures_time()
            self.client.timestamp_offset = st["serverTime"] - int(time.time() * 1000)
        except Exception:
            pass

    def _prefetch_step_sizes(self):
        try:
            info = self.client.futures_exchange_info()
            for sym in info["symbols"]:
                for f in sym["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        self.LOT_STEPS[sym["symbol"]] = float(f["stepSize"])
        except Exception as e:
            log.warning(f"Step size prefetch failed: {e}")

    def _round_qty(self, symbol: str, qty: float) -> float:
        step = self.LOT_STEPS.get(symbol, 0.01)
        if qty < step: return 0.0
        return round(round(qty / step) * step, 8)

    def get_klines(self, symbol: str, interval: str = "5m", limit: int = 200) -> list:
        if self.mode == "backtest":
            return self._historical_data.get(f"{symbol}_{interval}", [])
        if self.simulated:
            return self._simulate_klines(symbol, limit)
        try:
            raw = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            return [{"open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                     "close": float(k[4]), "volume": float(k[5])} for k in raw]
        except Exception as e:
            log.error(f"get_klines {symbol}: {e}")
            return []

    def _simulate_klines(self, symbol: str, limit: int = 200) -> list:
        import random
        price   = self._sim_prices.get(symbol, random.uniform(100, 50000))
        candles = []
        for _ in range(limit):
            chg = max(-0.05, min(0.05, random.gauss(0, 0.015)))
            o, c = price, price * (1 + chg)
            h = max(o, c) * (1 + abs(random.gauss(0, 0.004)))
            l = min(o, c) * (1 - abs(random.gauss(0, 0.004)))
            candles.append({"open": o, "high": h, "low": l, "close": c,
                            "volume": random.uniform(500, 5000) * (1 + abs(chg) * 10)})
            price = c
        self._sim_prices[symbol] = price
        return candles

    def get_account_balance(self) -> dict:
        if self.mode == "live" and not self.simulated:
            try:
                info = self.client.futures_account()
                return {b["asset"]: float(b["availableBalance"]) for b in info["assets"] if float(b["availableBalance"]) > 0}
            except Exception as e:
                log.error(f"Balance error: {e}")
                return {"USDT": 0}
        if self.trade_manager:
            eq = self.trade_manager.initial_capital + self.trade_manager.total_pnl
            return {"USDT": max(eq, 0.0)}
        return {"USDT": INITIAL_CAPITAL}

    def get_order_book_imbalance(self, symbol: str) -> Tuple[float, float]:
        if self.simulated: return 0.0, 0.0
        try:
            depth = self.client.futures_order_book(symbol=symbol, limit=20)
            bids  = sum(float(b[1]) for b in depth["bids"])
            asks  = sum(float(a[1]) for a in depth["asks"])
            return bids, asks
        except Exception:
            return 0.0, 0.0

    def get_funding_rate(self, symbol: str) -> float:
        if self.simulated: return 0.0
        try:
            info = self.client.futures_mark_price(symbol=symbol)
            return float(info.get("lastFundingRate", 0))
        except Exception:
            return 0.0

    def place_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False, retries: int = 3) -> Optional[dict]:
        if self.simulated:
            price = self._sim_prices.get(symbol, 100)
            tag   = "[CLOSE-SIM]" if reduce_only else "[OPEN-SIM]"
            log.info(f"{tag} {side} {quantity:.6f} {symbol} @ ${price:.4f}")
            return {"orderId": f"SIM_{int(time.time())}", "price": price}
        self._sync_time()
        for attempt in range(retries):
            try:
                params = {"symbol": symbol, "side": side, "type": "MARKET",
                          "quantity": quantity, "newOrderRespType": "RESULT"}
                if reduce_only:
                    params["reduceOnly"] = True
                order = self.client.futures_create_order(**params)
                log.info(f"[ORDER] placed: {order}")
                return order
            except BinanceAPIException as e:
                log.error(f"Order error {symbol} {side}: {e}")
                if "-1021" in str(e):
                    self._sync_time()
                    time.sleep(1)
                if attempt == retries - 1: return None
                time.sleep(2)
            except Exception as e:
                log.error(f"Unexpected order error {symbol}: {e}")
                if attempt == retries - 1: return None
                time.sleep(2)
        return None

    def load_historical(self, symbol: str, interval: str, start: str, end: str, timeout: int = 30) -> list:
        client = Client(API_KEY, API_SECRET, requests_params={"timeout": timeout})
        for attempt in range(3):
            try:
                klines = client.futures_historical_klines(symbol, interval, start, end)
                return [{"timestamp": k[0], "open": float(k[1]), "high": float(k[2]),
                         "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])} for k in klines]
            except Exception as e:
                log.warning(f"Historical {symbol} attempt {attempt+1}: {e}")
                if attempt == 2: return []
                time.sleep(5)
        return []

    def get_open_positions(self) -> Dict[str, dict]:
        if self.simulated: return {}
        try:
            return {p["symbol"]: {"qty": float(p["positionAmt"]), "entry_price": float(p["entryPrice"]), "unrealized_pnl": float(p["unRealizedProfit"])}
                    for p in self.client.futures_position_information() if abs(float(p["positionAmt"])) > 1e-8}
        except Exception as e:
            log.error(f"get_open_positions: {e}"); return {}


# ════════════════════════════════════════════════════════════════════════════
#  TRADE MANAGER
# ════════════════════════════════════════════════════════════════════════════
class TradeManager:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital   = initial_capital
        self.open_trades       = {}
        self.trade_history     = []
        self.total_pnl         = 0.0
        self.wins = self.losses = self.consecutive_losses = 0
        self.equity_curve      = [initial_capital]
        self.equity_timestamps = [datetime.utcnow().isoformat()]
        self.peak_equity       = initial_capital
        self.max_drawdown      = 0.0
        self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                self.open_trades       = d.get("open_trades", {})
                self.total_pnl         = d.get("total_pnl", 0.0)
                self.wins              = d.get("wins", 0)
                self.losses            = d.get("losses", 0)
                self.consecutive_losses = d.get("consecutive_losses", 0)
                self.peak_equity       = d.get("peak_equity", self.initial_capital)
                self.max_drawdown      = d.get("max_drawdown", 0.0)
                self.equity_curve      = d.get("equity_curve", [self.initial_capital])
                self.equity_timestamps = d.get("equity_timestamps", [datetime.utcnow().isoformat()])
                log.info(f"State loaded: {len(self.open_trades)} open trades")
            except Exception as e:
                log.warning(f"State load failed: {e}")

    def _save_state(self):
        try:
            STATE_FILE.write_text(json.dumps({
                "open_trades": self.open_trades, "total_pnl": self.total_pnl,
                "wins": self.wins, "losses": self.losses,
                "consecutive_losses": self.consecutive_losses,
                "peak_equity": self.peak_equity, "max_drawdown": self.max_drawdown,
                "equity_curve": self.equity_curve[-500:],
                "equity_timestamps": self.equity_timestamps[-500:],
                "saved_at": datetime.utcnow().isoformat(),
            }, indent=2))
        except Exception as e:
            log.error(f"State save: {e}")

    def open_trade(self, symbol, direction, entry_price, qty, order_id, signals,
                   regime_info=None, confidence=0.65, tp_override=None, sl_override=None):
        self.open_trades[symbol] = {
            "direction": direction, "entry_price": entry_price, "qty": qty,
            "order_id": order_id, "opened_at": datetime.utcnow().isoformat(),
            "partial_exits": [], "trailing_activated": False,
            "best_price": entry_price, "trailing_stop": None,
            "signals": signals, "regime_info": regime_info or {}, "hold_bars": 0,
            "confidence": confidence, "tp_override": tp_override, "sl_override": sl_override,
        }
        log.info(f"[OPEN] {symbol} {direction} @ {entry_price:.4f}  qty={qty:.6f}  conf={confidence:.2f}")
        self._save_state()

    def close_trade(self, symbol, exit_price, reason, qty=None,
                    ensemble=None, lstm_agents=None, meta_learner=None, analysis_result=None):
        if symbol not in self.open_trades:
            return None
        t     = self.open_trades[symbol]
        qty   = min(qty or t["qty"], t["qty"])
        entry = t["entry_price"]
        dir_  = t["direction"]
        gross = (exit_price - entry) * qty if dir_ == "BUY" else (entry - exit_price) * qty
        fee   = (entry * qty + exit_price * qty) * FEES
        slip  = abs(exit_price - entry) * qty * SLIPPAGE
        net   = gross - fee - slip
        self.total_pnl += net
        win   = net >= 0
        if win:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        if lstm_agents and symbol in lstm_agents:
            lstm_agents[symbol].record_outcome(win)
        if ensemble and t.get("signals"):
            for sig in t["signals"]:
                if sig["signal"] != "HOLD":
                    ensemble.update_weight(sig["agent"], sig["signal"] == dir_)
        if meta_learner and analysis_result is not None:
            meta_learner.update(analysis_result,
                                {"signal": dir_, "confidence": t.get("confidence", 0.65)},
                                t["signals"], win,
                                lstm_agents.get(symbol) if lstm_agents else None,
                                analysis_result.get("llm_sentiment", 0.5))
        record = {"symbol": symbol, "direction": dir_, "entry_price": entry, "exit_price": exit_price,
                  "qty": qty, "pnl": round(net, 4), "reason": reason,
                  "opened_at": t["opened_at"], "closed_at": datetime.utcnow().isoformat()}
        self.trade_history.append(record)
        eq = self.initial_capital + self.total_pnl
        self.equity_curve.append(eq)
        self.equity_timestamps.append(datetime.utcnow().isoformat())
        self.peak_equity  = max(self.peak_equity, eq)
        self.max_drawdown = max(self.max_drawdown, (self.peak_equity - eq) / self.peak_equity)
        try:
            with TRADES_FILE.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
        log.info(f"[CLOSE] {symbol}  PnL={net:+.4f} USDT  reason={reason}")
        Telegram.trade_closed(symbol, dir_, entry, exit_price, net, reason)
        t["qty"] -= qty
        if t["qty"] <= 1e-9:
            del self.open_trades[symbol]
        else:
            t["partial_exits"].append(record)
        self._save_state()
        return record

    def sync_with_exchange(self, exchange_positions: Dict[str, dict]):
        for sym, pos in exchange_positions.items():
            qty   = pos["qty"]
            entry = pos["entry_price"]
            dir_  = "BUY" if qty > 0 else "SELL"
            aq    = abs(qty)
            if sym in self.open_trades:
                self.open_trades[sym]["qty"]         = aq
                self.open_trades[sym]["entry_price"] = entry
            else:
                self.open_trades[sym] = {
                    "direction": dir_, "entry_price": entry, "qty": aq, "order_id": "external",
                    "opened_at": datetime.utcnow().isoformat(), "partial_exits": [],
                    "trailing_activated": False, "best_price": entry, "trailing_stop": None,
                    "signals": [], "regime_info": {}, "hold_bars": 0,
                    "confidence": 0.65, "tp_override": None, "sl_override": None,
                }
        for sym in list(self.open_trades.keys()):
            if sym not in exchange_positions:
                del self.open_trades[sym]
        self._save_state()

    def increment_hold_bars(self, symbol: str):
        if symbol in self.open_trades:
            self.open_trades[symbol]["hold_bars"] = self.open_trades[symbol].get("hold_bars", 0) + 1

    def update_mae_tracking(self, symbol: str, current_price: float):
        if symbol in self.open_trades:
            t = self.open_trades[symbol]
            if t["direction"] == "BUY":
                t["lowest_price"]  = min(t.get("lowest_price",  current_price), current_price)
            else:
                t["highest_price"] = max(t.get("highest_price", current_price), current_price)

    def summary(self) -> dict:
        total      = self.wins + self.losses
        win_rate   = self.wins / total * 100 if total else 0
        final_eq   = self.initial_capital + self.total_pnl
        total_ret  = (final_eq - self.initial_capital) / self.initial_capital * 100
        gw  = sum(t["pnl"] for t in self.trade_history if t["pnl"] > 0)
        gl  = abs(sum(t["pnl"] for t in self.trade_history if t["pnl"] < 0))
        pf  = gw / gl if gl else float("inf")
        return {"open": len(self.open_trades), "total_trades": total, "wins": self.wins,
                "losses": self.losses, "consecutive_losses": self.consecutive_losses,
                "win_rate": round(win_rate, 2), "total_pnl": round(self.total_pnl, 4),
                "total_return": round(total_ret, 2), "max_drawdown": round(self.max_drawdown * 100, 2),
                "profit_factor": round(pf, 3), "final_equity": round(final_eq, 2)}

    def daily_pnl(self) -> float:
        today = date.today().isoformat()
        return sum(t["pnl"] for t in self.trade_history if t.get("closed_at", "")[:10] == today)

    def compute_sharpe(self, risk_free_rate: float = 0.02) -> float:
        if len(self.equity_curve) < 5: return 0.0
        eq = pd.Series(self.equity_curve, dtype=float)
        dr = eq.pct_change().dropna()
        if dr.std() == 0 or len(dr) < 2: return 0.0
        return float((dr.mean() - risk_free_rate / 252) / dr.std() * math.sqrt(252))


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL LOGGER
# ════════════════════════════════════════════════════════════════════════════
class SignalLogger:
    _lock = threading.Lock()

    @staticmethod
    def log(symbol, signal, confidence, agent, regime, price):
        row = [datetime.utcnow().isoformat(), symbol, agent, signal,
               f"{confidence:.4f}", regime, f"{price:.6f}"]
        try:
            with SignalLogger._lock:
                hdr = not SIGNALS_CSV.exists()
                with SIGNALS_CSV.open("a", newline="") as f:
                    w = csv.writer(f)
                    if hdr:
                        w.writerow(["timestamp", "symbol", "agent", "signal", "confidence", "regime", "price"])
                    w.writerow(row)
        except PermissionError:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  FLASK DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def start_dashboard(bot):
    if not FLASK_OK or not ENABLE_DASHBOARD:
        return
    app = Flask("CryptoBotDashboard")

    @app.route("/")
    def home():
        return jsonify({"endpoints": ["/status", "/positions", "/trades", "/weights", "/gate_stats"]})

    @app.route("/status")
    def status():
        s = bot.trade_mgr.summary()
        return jsonify({**s, "daily_pnl": round(bot.trade_mgr.daily_pnl(), 4),
                        "circuit_open": bot.circuit_broken, "cycle": bot.cycle,
                        "mode": bot.mode, "sharpe": round(bot.trade_mgr.compute_sharpe(), 3)})

    @app.route("/positions")
    def positions(): return jsonify(bot.trade_mgr.open_trades)

    @app.route("/trades")
    def trades(): return jsonify(bot.trade_mgr.trade_history[-100:])

    @app.route("/weights")
    def weights(): return jsonify(bot.ensemble.weights)

    @app.route("/gate_stats")
    def gate_stats(): return jsonify(bot.entry_gate.stats())

    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False, use_reloader=False),
        daemon=True
    )
    t.start()
    log.info(f"Dashboard → http://localhost:{DASHBOARD_PORT}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN BOT
# ════════════════════════════════════════════════════════════════════════════
class CryptoBot:
    def __init__(self, mode: str = "live"):
        global TRADE_PAIRS
        if not TRADE_PAIRS:
            TRADE_PAIRS = get_top_volume_pairs(limit=TOP_PAIRS_BY_VOLUME, min_volume_usdt=MIN_24H_VOLUME_USDT)

        self.mode  = mode
        self.feed  = DataFeed(mode=mode)
        self.cycle = 0
        self.circuit_broken = False

        # Technical agents
        self.technical_agents: List[Any] = [
            TrendHunterAgent(),
            ADXAgent(threshold=REGIME_ADX_TRENDING),
            VolumeSentinelAgent(),
            StochRSIAgent(),
            IchimokuAgent(),
            OrderBookAgent(),
            VWAPAgent(),
            CrossAssetAgent(),
        ]
        if USE_MEAN_REVERSION_AGENT:
            self.technical_agents.append(MeanReversionAgent(
                MEAN_REVERSION_LOOKBACK, MEAN_REVERSION_ZSCORE_THRESH,
                MEAN_REVERSION_RSI_OVERSOLD, MEAN_REVERSION_RSI_OVERBOUGHT, MEAN_REVERSION_BB_POSITION))
        if USE_TREND_BREAKOUT_AGENT:
            self.technical_agents.append(TrendBreakoutAgent(
                TREND_BREAKOUT_LOOKBACK, TREND_BREAKOUT_LONG_MA, TREND_BREAKOUT_VOL_SPIKE,
                TREND_BREAKOUT_ADX_THRESH, TREND_BREAKOUT_USE_BOLLINGER, TREND_BREAKOUT_BB_STD, TREND_BREAKOUT_LONG_ONLY))
        if USE_NOISE_SCALPER:
            self.technical_agents.append(NoiseScalperAgent())

        # LLM sentiment (optional)
        self.llm = LLMSentimentAgent() if USE_LLM_SENTIMENT else None

        # ML agents
        self.ml_agents: Dict[str, dict] = {}
        self.ml_agents_enabled = USE_ML_AGENT and ML_AVAILABLE
        if self.ml_agents_enabled:
            for sym in TRADE_PAIRS:
                self.ml_agents[sym] = {"ElasticNet": ElasticNetAgent(sym, ML_TRAIN_INTERVAL, ML_FEATURE_WINDOW, ML_MIN_SAMPLES)}
                if XGB_AVAILABLE:
                    self.ml_agents[sym]["XGBoost"] = XGBoostAgent(sym, ML_TRAIN_INTERVAL, ML_FEATURE_WINDOW, ML_MIN_SAMPLES)
                if USE_RANDOMFOREST and RF_AVAILABLE:
                    self.ml_agents[sym]["RandomForest"] = RandomForestAgent(sym, ML_TRAIN_INTERVAL, ML_FEATURE_WINDOW, ML_MIN_SAMPLES)

        # Ensemble
        all_agents = self.technical_agents.copy()
        if self.llm:
            all_agents.append(self.llm)
        self.ensemble = AdaptiveEnsemble(all_agents, decay=WEIGHT_DECAY, window=RECENT_WINDOW, weight_floor=ENSEMBLE_WEIGHT_FLOOR)
        for name in ["ElasticNet", "XGBoost", "RandomForest", "LSTM"]:
            if name not in self.ensemble.weights:
                self.ensemble.weights[name]     = ENSEMBLE_WEIGHT_FLOOR
                self.ensemble.performance[name] = deque(maxlen=RECENT_WINDOW)

        self.risk      = RiskManager()
        self.trade_mgr = TradeManager(INITIAL_CAPITAL)
        self.feed.trade_manager = self.trade_mgr
        self._trade_lock = threading.Lock()

        self._price_history  = {sym: deque(maxlen=60) for sym in TRADE_PAIRS}
        self.candle_history  = {sym: deque(maxlen=ML_FEATURE_WINDOW + 1) for sym in TRADE_PAIRS}
        self.ml_update_count = {sym: 0 for sym in TRADE_PAIRS}

        self.entry_gate    = SelectiveEntryGate(MIN_AGREEING_AGENTS, GATE_MIN_CONFIDENCE, GATE_MAX_CONFLICT, GATE_MIN_REGIME_CONF)
        self.session_filter = SessionFilter() if USE_SESSION_FILTER else None
        self.order_flow    = OrderFlowAgent() if USE_ORDER_FLOW else None

        self.lstm_agents: Dict[str, LSTMAgent] = {}
        if USE_LSTM_AGENT:
            for sym in TRADE_PAIRS:
                self.lstm_agents[sym] = LSTMAgent(sym)

        self.brain = None
        if LLM_BRAIN_AVAILABLE and USE_LLM_BRAIN:
            try:
                self.brain = LLMBrainAgent(provider=BRAIN_PROVIDER, cache_ttl=BRAIN_CACHE_TTL)
            except Exception:
                pass

        self.meta_learner = None
        if USE_META_LEARNER and ML_AVAILABLE:
            self.meta_learner = MetaLearner(META_LEARNER_FEATURES, META_LEARNER_RETRAIN_EVERY, META_MIN_SAMPLES)
            log.info("[MetaLearner] Enabled")

        self._recently_closed = {}
        self._cooldown_secs   = 300

        if not self.feed.simulated:
            pos = self.feed.get_open_positions()
            if pos:
                self.trade_mgr.sync_with_exchange(pos)

    # ── Helpers ─────────────────────────────────────────────────────────
    def _macro_trend(self, symbol: str) -> str:
        try:
            candles = self.feed.get_klines(symbol, interval="15m", limit=30)
            if candles and len(candles) >= 21:
                prices = [c["close"] for c in candles]
                return "up" if ema(prices, 9) > ema(prices, 21) else "down"
        except Exception:
            pass
        return "flat"

    def _check_circuit_breaker(self) -> bool:
        if self.circuit_broken: return True
        daily  = self.trade_mgr.daily_pnl()
        limit  = -(self.trade_mgr.initial_capital * DAILY_LOSS_LIMIT_PCT / 100)
        if daily < limit:
            log.warning(f"Circuit breaker: daily PnL={daily:.2f} < limit={limit:.2f}")
            Telegram.circuit_breaker(daily)
            self.circuit_broken = True
        return self.circuit_broken

    def _market_filters(self, ind: dict) -> bool:
        if ENABLE_VOLUME_FILTER:
            if ind["volumes"][-1] / (ind["avg_vol20"] + 1e-9) < MIN_VOLUME_RATIO:
                return False
        if ENABLE_VOLATILITY_FILTER:
            if ind["atr14"] / (ind["prices"][-1] + 1e-9) * 100 < MIN_VOLATILITY_PCT:
                return False
        return True

    def _time_filter(self) -> bool:
        if not USE_TIME_FILTER: return True
        now = datetime.utcnow().time()
        return not (dt_time(0, 0) <= now <= dt_time(4, 0))

    def _funding_filter(self, symbol: str, direction: str) -> bool:
        if not ENABLE_FUNDING_FILTER or direction != "BUY": return True
        rate = self.feed.get_funding_rate(symbol)
        if rate > MAX_FUNDING_RATE:
            log.info(f"[SKIP] {symbol} funding={rate:.4%}")
            return False
        return True

    def _correlation_filter(self, symbol: str, direction: str) -> bool:
        same = [s for s, t in self.trade_mgr.open_trades.items() if t["direction"] == direction and s != symbol]
        if len(same) < MAX_CORR_TRADES: return True
        my = pd.Series(list(self._price_history.get(symbol, [])))
        for other in same:
            op = pd.Series(list(self._price_history.get(other, [])))
            if len(op) >= len(my) >= 10:
                corr = my.corr(op.iloc[-len(my):])
                if not math.isnan(corr) and corr > CORRELATION_THRESHOLD:
                    log.info(f"[SKIP] {symbol} correlated with {other}")
                    return False
        return True

    def _higher_tf_filter(self, symbol: str, direction: str) -> bool:
        candles = self.feed.get_klines(symbol, interval="1h", limit=50)
        if not candles or len(candles) < 21: return True
        prices = [c["close"] for c in candles]
        e9, e21 = ema(prices, 9), ema(prices, 21)
        htf_bull = e9 > e21 * 1.001
        htf_bear = e9 < e21 * 0.999
        if direction == "BUY"  and htf_bear: return False
        if direction == "SELL" and htf_bull: return False
        return True

    # ── Analyse a single pair ────────────────────────────────────────────
    def analyze_pair(self, symbol: str, injected_candles: Optional[list] = None) -> Optional[dict]:
        candles = injected_candles or self.feed.get_klines(symbol, interval="5m", limit=200)
        if len(candles) < 60 or symbol in self.trade_mgr.open_trades:
            return None

        ind = precompute_indicators(candles)
        self._price_history[symbol].append(ind["current_price"])

        if not self._market_filters(ind):
            return None

        bids, asks = self.feed.get_order_book_imbalance(symbol)
        regime     = detect_regime(candles)
        adx_val    = ind["adx14"]
        regime_conf = min(1.0, adx_val / 50.0)
        regime_info = {"regime": regime, "confidence": regime_conf, "adx": adx_val,
                       "volatility": ind["atr14"] / (ind["current_price"] + 1e-9)}

        signals = []
        for agent in self.technical_agents:
            if isinstance(agent, OrderBookAgent):
                sig = agent.analyze(ind, candles, bids, asks)
            elif isinstance(agent, CrossAssetAgent):
                sig = agent.analyze(ind, candles, symbol=symbol)
            else:
                sig = agent.analyze(ind, candles)

            if regime == "TRENDING" and agent.name in ("MeanReversion", "NoiseScalper"):
                sig = {"signal": "HOLD", "confidence": 0.0, "agent": agent.name}
            if regime == "RANGING"  and agent.name in ("TrendBreakout",):
                sig = {"signal": "HOLD", "confidence": 0.0, "agent": agent.name}
            boost_map = {"TRENDING": ["TrendHunter","ADX","TrendBreakout"],
                         "RANGING":  ["StochRSI","VWAP","MeanReversion"],
                         "VOLATILE": ["VolumeSentinel"]}
            if agent.name in boost_map.get(regime, []):
                sig["confidence"] = min(0.95, sig["confidence"] * 1.10)

            signals.append(sig)

        llm_sentiment = 0.5
        if self.llm:
            llm_sig   = self.llm.analyze(ind, candles, symbol=symbol)
            llm_sentiment = llm_sig.get("meta", {}).get("sentiment", 0.5)
            signals.append(llm_sig)

        # Note: ML agents are now trained on trade outcomes, not updated here in backtest.
        # In live mode, they still need periodic training, which happens in monitor_positions.
        if self.ml_agents_enabled and symbol in self.ml_agents:
            for ml_agent in self.ml_agents[symbol].values():
                signals.append(ml_agent.analyze(ind, candles))

        if self.order_flow:
            signals.append(self.order_flow.analyze(ind, candles, symbol=symbol))

        if symbol in self.lstm_agents:
            signals.append(self.lstm_agents[symbol].analyze(ind, candles))
            if len(candles) >= LSTM_SEQ_LEN + 2:
                prev = candles[:-1]
                self.lstm_agents[symbol].update(precompute_indicators(prev), prev, candles[-1])
                if self.lstm_agents[symbol]._update_count % LSTM_TRAIN_EVERY == 0:
                    threading.Thread(target=self.lstm_agents[symbol].train, daemon=True).start()

        decision = self.ensemble.vote(signals)

        if self.brain and decision["signal"] != "HOLD":
            mctx = {"price": ind["current_price"], "regime": regime, "rsi14": ind["rsi14"],
                    "adx14": adx_val, "btc_trend": self._macro_trend("BTCUSDT"),
                    "eth_trend": self._macro_trend("ETHUSDT"),
                    "consecutive_losses": self.trade_mgr.consecutive_losses}
            bd = self.brain.decide(symbol, mctx, signals)
            if bd["confidence"] >= BRAIN_MIN_CONF:
                decision = {"signal": bd["signal"], "confidence": min(decision["confidence"], bd["confidence"])}

        if decision["signal"] != "HOLD" and not self._higher_tf_filter(symbol, decision["signal"]):
            decision = {"signal": "HOLD", "confidence": 0.0}

        # Meta-learner override with gate bypass
        meta_override = False
        original_signal = decision["signal"]
        if self.meta_learner and decision["signal"] != "HOLD":
            ar = {"ind": ind, "candles": candles, "price": ind["current_price"],
                  "regime_info": regime_info, "consecutive_losses": self.trade_mgr.consecutive_losses,
                  "btc_trend": self._macro_trend("BTCUSDT"), "eth_trend": self._macro_trend("ETHUSDT")}
            meta_res = self.meta_learner.predict(ar, decision, signals, self.lstm_agents.get(symbol), llm_sentiment)
            if meta_res is not None:
                meta_sig, meta_conf = meta_res
                if meta_sig != "HOLD" and meta_conf >= 0.75:
                    meta_override = True
                    decision = {"signal": meta_sig, "confidence": meta_conf}
                    log.info(f"[Meta] {symbol}: {original_signal} → {meta_sig} ({meta_conf:.2f}) [OVERRIDE]")

        for sig in signals:
            SignalLogger.log(symbol, sig["signal"], sig["confidence"], sig["agent"], regime, ind["current_price"])

        return {"symbol": symbol, "price": ind["current_price"], "decision": decision,
                "atr": ind["atr14"], "signals": signals, "regime": regime,
                "regime_info": regime_info, "ind": ind, "candles": candles,
                "volume_ratio": ind["volumes"][-1] / (ind["avg_vol20"] + 1e-9) if ind["avg_vol20"] != 0 else 1.0,
                "llm_sentiment": llm_sentiment, "meta_override": meta_override}

    # ── Execute ──────────────────────────────────────────────────────────
    def execute_decision(self, analysis: dict):
        symbol   = analysis["symbol"]
        price    = analysis["price"]
        decision = analysis["decision"]
        signal   = decision["signal"]
        conf     = decision["confidence"]
        regime_info = analysis.get("regime_info", {})
        meta_override = analysis.get("meta_override", False)

        if signal == "HOLD" or self._check_circuit_breaker():
            return
        if not self._time_filter():
            return
        if self.session_filter:
            ok, session_name, _ = self.session_filter.allows_entry(symbol)
            if not ok:
                log.info(f"[SESSION] {symbol} blocked — {session_name}")
                return
            conf = self.session_filter.adjust_confidence(conf)
        if conf < MIN_CONFIDENCE:
            return

        gate_ok, gate_reason = self.entry_gate.check(analysis["signals"], decision, regime_info)
        if meta_override and not gate_ok:
            bypass_reasons = ["agree", "conflict", "conf_"]
            if any(r in gate_reason for r in bypass_reasons):
                gate_ok = True
                gate_reason = "meta_override"
                log.info(f"[Meta] {symbol}: gate block '{gate_reason}' overridden by MetaLearner")
        if not gate_ok:
            log.info(f"[GATE] {symbol} blocked: {gate_reason}")
            return
        if not self._funding_filter(symbol, signal):
            return
        if not self._correlation_filter(symbol, signal):
            return
        if self.trade_mgr.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            log.warning(f"[HALT] {self.trade_mgr.consecutive_losses} consecutive losses")
            return

        with self._trade_lock:
            if len(self.trade_mgr.open_trades) >= MAX_OPEN_TRADES:
                return
            if symbol in self.trade_mgr.open_trades:
                return
            now = time.time()
            if symbol in self._recently_closed and now - self._recently_closed[symbol] < self._cooldown_secs:
                return
            if not self.feed.simulated:
                pos = self.feed.get_open_positions()
                if symbol in pos:
                    return

            balance = self.feed.get_account_balance()
            usdt    = balance.get("USDT", 0)
            self.risk.consecutive_losses = self.trade_mgr.consecutive_losses
            size = self.risk.position_size(usdt, conf, analysis["atr"], price, analysis.get("volume_ratio", 1.0))

            fee_cost     = size * FEES * 2
            slip_cost    = size * SLIPPAGE * 2
            breakeven_pct = (fee_cost + slip_cost) / size * 100
            if breakeven_pct > STOP_LOSS_PCT * 0.5:
                log.warning(f"[SKIP] {symbol} trade too small: breakeven={breakeven_pct:.2f}% vs SL={STOP_LOSS_PCT}%")
                return

            qty = self.feed._round_qty(symbol, size / price)
            if qty <= 0:
                log.warning(f"[SKIP] {symbol} qty=0 after rounding")
                return

            order = self.feed.place_order(symbol, "BUY" if signal == "BUY" else "SELL", qty)
            if order:
                self.trade_mgr.open_trade(symbol, signal, price, qty,
                                          order.get("orderId", "unknown"),
                                          analysis["signals"], regime_info, confidence=conf)
                Telegram.trade_opened(symbol, signal, price, qty, conf)

    # ── Monitor open positions ───────────────────────────────────────────
    def monitor_positions(self):
        for symbol in list(self.trade_mgr.open_trades.keys()):
            try:
                candles = self.feed.get_klines(symbol, limit=60)
                if not candles: continue
                current = candles[-1]["close"]
                ind     = precompute_indicators(candles)
                trade   = self.trade_mgr.open_trades.get(symbol)
                if not trade: continue

                check = self.risk.check_position(
                    trade["entry_price"], current, trade["direction"],
                    ind["rsi14"], ind["atr14"], ind["ema9"],
                    partial_taken=bool(trade.get("partial_exits")),
                    hold_bars=trade.get("hold_bars", 0),
                    trade_confidence=trade.get("confidence", 0.65),
                    session_filter=self.session_filter,
                    lowest_price=trade.get("lowest_price"),
                    highest_price=trade.get("highest_price"),
                )
                close_side = "SELL" if trade["direction"] == "BUY" else "BUY"
                ar = {"ind": ind, "candles": candles, "price": current,
                      "regime_info": {"regime": detect_regime(candles)},
                      "consecutive_losses": self.trade_mgr.consecutive_losses,
                      "btc_trend": self._macro_trend("BTCUSDT"),
                      "eth_trend": self._macro_trend("ETHUSDT"),
                      "llm_sentiment": trade.get("llm_sentiment", 0.5)}

                if check["action"] == "CLOSE":
                    qty_r = self.feed._round_qty(symbol, trade["qty"])
                    if qty_r > 0:
                        self.feed.place_order(symbol, close_side, qty_r, reduce_only=True)
                        self.trade_mgr.close_trade(symbol, current, check["reason"],
                                                   ensemble=self.ensemble,
                                                   lstm_agents=self.lstm_agents,
                                                   meta_learner=self.meta_learner,
                                                   analysis_result=ar)
                elif check["action"] == "PARTIAL":
                    half = self.feed._round_qty(symbol, trade["qty"] / 2)
                    if half > 0:
                        self.feed.place_order(symbol, close_side, half, reduce_only=True)
                        self.trade_mgr.close_trade(symbol, current, check["reason"], qty=half,
                                                   ensemble=self.ensemble,
                                                   lstm_agents=self.lstm_agents,
                                                   meta_learner=self.meta_learner,
                                                   analysis_result=ar)
                else:
                    self._update_trailing(symbol, trade, current)
                    self.trade_mgr.update_mae_tracking(symbol, current)
                    pct = ((current - trade["entry_price"]) / trade["entry_price"] * 100
                           if trade["direction"] == "BUY"
                           else (trade["entry_price"] - current) / trade["entry_price"] * 100)
                    log.info(f"[HOLD] {symbol} {trade['direction']} PnL%={pct:+.2f}%  bars={trade.get('hold_bars',0)}")

            except Exception:
                log.error(f"Monitor error {symbol}: {traceback.format_exc()}")

    def _update_trailing(self, symbol: str, trade: dict, current: float):
        rm        = self.risk
        direction = trade["direction"]
        entry     = trade["entry_price"]
        pnl_pct   = ((current - entry) / entry * 100 if direction == "BUY"
                     else (entry - current) / entry * 100)
        retreat   = rm.trailing_retreat

        if direction == "BUY":
            if pnl_pct >= rm.trailing_activation:
                if not trade["trailing_activated"]:
                    trade["trailing_activated"] = True
                    trade["best_price"]   = current
                    trade["trailing_stop"] = current * (1 - retreat)
                elif current > trade["best_price"]:
                    trade["best_price"]   = current
                    trade["trailing_stop"] = current * (1 - retreat)
                if trade.get("trailing_stop") and current <= trade["trailing_stop"]:
                    qty = self.feed._round_qty(symbol, trade["qty"])
                    if qty > 0:
                        self.feed.place_order(symbol, "SELL", qty, reduce_only=True)
                        ar = {"ind": {}, "candles": [], "price": current,
                              "regime_info": {"regime": "UNKNOWN"},
                              "consecutive_losses": self.trade_mgr.consecutive_losses,
                              "btc_trend": "flat", "eth_trend": "flat", "llm_sentiment": 0.5}
                        self.trade_mgr.close_trade(symbol, current, "trailing_stop",
                                                   ensemble=self.ensemble, lstm_agents=self.lstm_agents,
                                                   meta_learner=self.meta_learner, analysis_result=ar)
        else:
            if pnl_pct >= rm.trailing_activation:
                if not trade["trailing_activated"]:
                    trade["trailing_activated"] = True
                    trade["best_price"]   = current
                    trade["trailing_stop"] = current * (1 + retreat)
                elif current < trade["best_price"]:
                    trade["best_price"]   = current
                    trade["trailing_stop"] = current * (1 + retreat)
                if trade.get("trailing_stop") and current >= trade["trailing_stop"]:
                    qty = self.feed._round_qty(symbol, trade["qty"])
                    if qty > 0:
                        self.feed.place_order(symbol, "BUY", qty, reduce_only=True)
                        ar = {"ind": {}, "candles": [], "price": current,
                              "regime_info": {"regime": "UNKNOWN"},
                              "consecutive_losses": self.trade_mgr.consecutive_losses,
                              "btc_trend": "flat", "eth_trend": "flat", "llm_sentiment": 0.5}
                        self.trade_mgr.close_trade(symbol, current, "trailing_stop",
                                                   ensemble=self.ensemble, lstm_agents=self.lstm_agents,
                                                   meta_learner=self.meta_learner, analysis_result=ar)

    def print_status(self):
        s = self.trade_mgr.summary()
        log.info("─" * 60)
        log.info(f"Cycle #{self.cycle}  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        log.info(f"Open={s['open']}/{MAX_OPEN_TRADES}  WR={s['win_rate']:.1f}%  Trades={s['total_trades']}")
        log.info(f"PnL={s['total_pnl']:+.2f} USDT  Return={s['total_return']:.1f}%  DD={s['max_drawdown']:.1f}%  PF={s['profit_factor']:.2f}")
        log.info(f"Daily={self.trade_mgr.daily_pnl():+.2f}  Circuit={'OPEN' if self.circuit_broken else 'OK'}  ConsecLoss={s['consecutive_losses']}")
        log.info(f"Gate: {self.entry_gate.stats()}")
        log.info("─" * 60)

    # ── Live loop ────────────────────────────────────────────────────────
    def run_live(self, interval: int = 60):
        log.info(f"LIVE loop — {len(TRADE_PAIRS)} pairs  interval={interval}s")
        log.info(f"Mode: {'SIMULATION' if self.feed.simulated else 'BINANCE TESTNET'}")
        last_day = date.today()
        while True:
            self.cycle += 1
            try:
                today = date.today()
                if today != last_day:
                    Telegram.daily_summary(self.trade_mgr.summary())
                    self.circuit_broken = False
                    last_day = today

                if self._check_circuit_breaker():
                    time.sleep(interval)
                    continue

                self.monitor_positions()

                for sym in list(self.trade_mgr.open_trades.keys()):
                    self.trade_mgr.increment_hold_bars(sym)

                if not self.feed.simulated:
                    prev = set(self.trade_mgr.open_trades.keys())
                    self.trade_mgr.sync_with_exchange(self.feed.get_open_positions())
                    for sym in prev - set(self.trade_mgr.open_trades.keys()):
                        self._recently_closed[sym] = time.time()

                free = [s for s in TRADE_PAIRS if s not in self.trade_mgr.open_trades]
                if not free:
                    time.sleep(interval)
                    continue

                log.info(f"[CYCLE {self.cycle}] Analysing {len(free)} pairs")
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futures = {ex.submit(self.analyze_pair, sym): sym for sym in free}
                    for f in as_completed(futures):
                        try:
                            res = f.result()
                            if res and res["decision"]["signal"] != "HOLD":
                                log.info(f"{res['symbol']:12s}  {res['decision']['signal']}({res['decision']['confidence']:.2f})  {res['regime']}")
                                self.execute_decision(res)
                        except Exception as e:
                            log.error(f"Analysis error: {e}")

                if self.cycle % 5 == 0:
                    self.print_status()

                time.sleep(interval)

            except KeyboardInterrupt:
                log.info("Stopped.")
                self.trade_mgr._save_state()
                break
            except Exception as e:
                log.error(f"Cycle error: {e}")
                time.sleep(10)

    # ── Backtest (CORRECTED) ─────────────────────────────────────────────────
    def run_backtest(self, start_date: str, end_date: str, interval: str = "1h") -> dict:
        global TRADE_PAIRS
        log.info(f"Backtest: {start_date} → {end_date} ({interval})")
        valid_pairs = []
        for sym in TRADE_PAIRS:
            candles = self.feed.load_historical(sym, interval, start_date, end_date)
            if len(candles) < 200:
                log.warning(f"Skip {sym}: only {len(candles)} candles")
                continue
            self.feed._historical_data[f"{sym}_{interval}"] = candles
            valid_pairs.append(sym)
            log.info(f"  {sym}: {len(candles)} candles")

        if not valid_pairs:
            log.error("No valid pairs for backtest")
            return {}

        original_pairs = TRADE_PAIRS[:]
        TRADE_PAIRS    = valid_pairs
        base_candles   = self.feed._historical_data[f"{valid_pairs[0]}_{interval}"]
        n              = len(base_candles)
        warmup         = 100

        for idx in range(warmup, n - 1):
            # --- Entries (use next bar's open) ---
            for sym in TRADE_PAIRS:
                history = self.feed._historical_data.get(f"{sym}_{interval}", [])
                if idx + 1 >= len(history): continue
                window = history[max(0, idx - 200):idx + 1]
                if len(window) < 60: continue

                res = self.analyze_pair(sym, injected_candles=window)
                if res and res["decision"]["signal"] != "HOLD":
                    next_candle = history[idx + 1]
                    res["price"] = next_candle["open"]
                    self.execute_decision(res)

            # --- Exits (intra‑bar simulation) ---
            for sym in list(self.trade_mgr.open_trades.keys()):
                history = self.feed._historical_data.get(f"{sym}_{interval}", [])
                if idx >= len(history): continue
                window = history[max(0, idx - 60):idx + 1]
                if not window: continue

                trade = self.trade_mgr.open_trades.get(sym)
                if not trade: continue

                current_candle = window[-1]
                entry = trade["entry_price"]
                direction = trade["direction"]
                hold_bars = trade.get("hold_bars", 0)

                ind = precompute_indicators(window)
                atr_val = ind["atr14"]

                exit_price = None
                reason = None
                if direction == "BUY":
                    stop_price = entry * (1 - STOP_LOSS_PCT / 100)
                    take_price = entry * (1 + TAKE_PROFIT_PCT / 100)
                    atr_stop = entry - ATR_STOP_MULT * atr_val
                    atr_take = entry + ATR_TAKE_MULT * atr_val
                    stop_price = max(stop_price, atr_stop)
                    take_price = min(take_price, atr_take)

                    if current_candle["low"] <= stop_price:
                        exit_price = stop_price
                        reason = "stop_loss"
                    elif current_candle["high"] >= take_price:
                        exit_price = take_price
                        reason = "take_profit"
                else:
                    stop_price = entry * (1 + STOP_LOSS_PCT / 100)
                    take_price = entry * (1 - TAKE_PROFIT_PCT / 100)
                    atr_stop = entry + ATR_STOP_MULT * atr_val
                    atr_take = entry - ATR_TAKE_MULT * atr_val
                    stop_price = min(stop_price, atr_stop)
                    take_price = max(take_price, atr_take)

                    if current_candle["high"] >= stop_price:
                        exit_price = stop_price
                        reason = "stop_loss"
                    elif current_candle["low"] <= take_price:
                        exit_price = take_price
                        reason = "take_profit"

                if exit_price is not None:
                    qty_r = self.feed._round_qty(sym, trade["qty"])
                    if qty_r > 0:
                        ar = {"ind": ind, "candles": window, "price": exit_price,
                              "regime_info": {"regime": detect_regime(window)},
                              "consecutive_losses": self.trade_mgr.consecutive_losses,
                              "btc_trend": "flat", "eth_trend": "flat", "llm_sentiment": 0.5}
                        record = self.trade_mgr.close_trade(sym, exit_price, reason, qty=qty_r,
                                                            ensemble=self.ensemble, lstm_agents=self.lstm_agents,
                                                            meta_learner=self.meta_learner, analysis_result=ar)
                        # Train ML agents on actual trade outcome
                        if record and self.ml_agents_enabled and sym in self.ml_agents:
                            entry_ind = precompute_indicators(history[max(0, trade.get("entry_idx", 0)-200):trade.get("entry_idx", 0)+1])
                            for ml_agent in self.ml_agents[sym].values():
                                ml_agent.update_with_trade_outcome(entry_ind, [], record['pnl'])
                    continue

                if hold_bars >= MAX_HOLD_BARS:
                    exit_price = current_candle["close"]
                    qty_r = self.feed._round_qty(sym, trade["qty"])
                    if qty_r > 0:
                        ar = {"ind": ind, "candles": window, "price": exit_price,
                              "regime_info": {"regime": detect_regime(window)},
                              "consecutive_losses": self.trade_mgr.consecutive_losses,
                              "btc_trend": "flat", "eth_trend": "flat", "llm_sentiment": 0.5}
                        record = self.trade_mgr.close_trade(sym, exit_price, "timeout", qty=qty_r,
                                                            ensemble=self.ensemble, lstm_agents=self.lstm_agents,
                                                            meta_learner=self.meta_learner, analysis_result=ar)
                        if record and self.ml_agents_enabled and sym in self.ml_agents:
                            entry_ind = precompute_indicators(history[max(0, trade.get("entry_idx", 0)-200):trade.get("entry_idx", 0)+1])
                            for ml_agent in self.ml_agents[sym].values():
                                ml_agent.update_with_trade_outcome(entry_ind, [], record['pnl'])
                    continue

                self.trade_mgr.update_mae_tracking(sym, current_candle["close"])
                self.trade_mgr.increment_hold_bars(sym)

            if idx % 500 == 0:
                closed = self.trade_mgr.wins + self.trade_mgr.losses
                log.info(f"  Progress: {idx}/{n}  trades={closed}  WR={self.trade_mgr.wins/(closed+1e-9)*100:.1f}%")

        s      = self.trade_mgr.summary()
        sharpe = self.trade_mgr.compute_sharpe()
        log.info("=" * 60)
        log.info("BACKTEST RESULTS")
        for k, v in s.items():
            log.info(f"  {k:22s}: {v}")
        log.info(f"  {'sharpe':22s}: {sharpe:.3f}")
        log.info("=" * 60)
        TRADE_PAIRS = original_pairs
        return {**s, "sharpe": round(sharpe, 3)}


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MODE = "live"   # change to "live" for live trading
    bot  = CryptoBot(mode=MODE)
    start_dashboard(bot)
    if MODE == "live":
        bot.run_live(interval=60)
    else:
        bot.run_backtest(BACKTEST_START, BACKTEST_END, interval=BACKTEST_INTERVAL)
