#!/usr/bin/env python3
"""
CryptoBot — 100‑Coin Ultra‑Fast + LLM Sentiment + $1 Profit Lock + ADAPTIVE RISK
================================================================================
- Auto‑fetches top pairs
- WebSocket order book & trade stream
- LLM sentiment always visible + entry gate
- $1 unrealised profit lock
- Manual trade intelligence
- Full backtest engine
- 4‑layer quant engine, ML veto, LSTM, Kelly, circuit breakers
- Adaptive Risk Engine: profiles selected by a composite score combining
  ATR volatility, order‑book imbalance, candle quality, ADX, regime, session
- ACTUAL FILL PRICES used for entry and exit → PnL matches exchange
- ADM‑Quad V2 neural network integrated as confidence multiplier
"""
import os, sys, time, logging, json, threading, math, traceback, asyncio, random, re
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Fast JSON
try:
    import ujson as json_fast
except ImportError:
    import json as json_fast

# WebSocket
try:
    import websockets
    WS_OK = True
except ImportError:
    WS_OK = False
    print("⚠️ websockets not installed. Install: pip install websockets")

# Binance
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_OK = True
except ImportError:
    BINANCE_OK = False

# ML
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    ML_OK = True
except ImportError:
    ML_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from flask import Flask, jsonify
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

# Import config
try:
    from config import *
except ImportError:
    print("ERROR: config.py not found. Using defaults.")
    API_KEY = ""
    API_SECRET = ""
    TRADE_PAIRS = []

# ─────────────────────────────── CONFIG DEFAULTS ───────────────────────────────
_CONFIG_DEFAULTS = {
    'MAX_HOLD_SECONDS': 14400,
    'TIMEOUT_STANDARD_SECONDS': 7200,
    'VWAP_MIN_DEVIATION_ATR': 0.4, 'VWAP_MAX_DEVIATION_ATR': 5.0,
    'OB_DEPTH_LEVELS': 20, 'OF_TRADE_LOOKBACK': 100,
    'OB_CACHE_TTL': 5, 'OF_CACHE_TTL': 10,
    'OB_IMBALANCE_SEED': 0.15, 'OF_IMBALANCE_SEED': 0.15,
    'WS_ORDER_BOOK_DEPTH': 20, 'WS_RECONNECT_DELAY': 5.0, 'WS_MAX_CONNECTIONS': 3,
    'KELLY_LOOKBACK': 40, 'KELLY_MAX_FRACTION': 0.05, 'KELLY_FLOOR': 0.02,
    'MAX_POSITION_PCT': 0.12,
    'WEIGHT_PRICE_MOMENTUM': 0.30, 'WEIGHT_MICROSTRUCTURE': 0.40,
    'WEIGHT_CONTEXT': 0.20, 'WEIGHT_ML_VETO': 0.10,
    'CALIBRATION_PERCENTILE': 65,
    'REGIME_ADX_TRENDING': 20, 'REGIME_ADX_STRONG': 35, 'REGIME_VOL_HIGH': 0.020,
    'CONTEXT_ASSETS': ["BTCUSDT","ETHUSDT","BNBUSDT"],
    'CONTEXT_INTERVAL': "15m", 'CONTEXT_EMA_FAST': 9, 'CONTEXT_EMA_SLOW': 21,
    'CONTEXT_CACHE_TTL': 45,
    'SESSION_SCORES': {
        "london_open":1.40,"london_us_overlap":1.50,"us_open":1.45,
        "us_afternoon":1.20,"london_midday":1.10,"us_close":0.95,
        "asian_session":0.80,"dead_hours":0.65
    },
    'MIN_SESSION_SCORE': 0.60,
    'TRADE_COOLDOWN_SECONDS': 90, 'LIVE_LOOP_INTERVAL': 30,
    'KLINE_INTERVAL': "5m", 'KLINE_LIMIT': 200,
    'ATR_STOP_MULT': 1.5, 'ATR_TAKE_MULT': 2.5, 'ATR_PARTIAL_MULT': 1.70,
    'TRAILING_ATR_MULT': 1.0, 'HARD_STOP_ATR_MULT': 1.5,
    'MAX_HOLD_BARS': 36, 'MIN_RR_RATIO': 1.5,
    'MAX_CONSECUTIVE_LOSSES': 14, 'DAILY_LOSS_LIMIT_PCT': 21.0,
    'WEEKLY_LOSS_LIMIT_PCT': 10.0,
    'ML_VETO_ENABLED': True, 'ML_VETO_MIN_SAMPLES': 80,
    'ML_VETO_LOSS_THRESHOLD': 0.65, 'ML_TRAIN_INTERVAL': 15,
    'ML_FEATURE_WINDOW': 600, 'ML_SAVE_PATH': "logs/models",
    'LSTM_ENABLED': True, 'LSTM_SEQ_LEN': 20, 'LSTM_HIDDEN_SIZE': 64,
    'LSTM_MIN_SAMPLES': 80, 'LSTM_TRAIN_EVERY': 20,
    'MIN_CONFIDENCE': 0.72, 'INITIAL_CAPITAL': 200.0,
    'MAX_OPEN_TRADES': 5, 'MIN_TRADE_USDT': 15.0,
    'FEES': 0.0004, 'SLIPPAGE': 0.0002,
    'TOP_PAIRS_BY_VOLUME': 10, 'MIN_24H_VOLUME_USDT': 50_000_000,
    'ENABLE_DASHBOARD': True, 'DASHBOARD_PORT': 5050, 'LOG_LEVEL': "INFO",
    'BACKTEST_START': "2023-06-11", 'BACKTEST_END': "2023-08-22",
    'BACKTEST_INTERVAL': "1h",
    'LLM_ENABLED': True, 'LLM_PROVIDER': "groq",
    'GROQ_API_KEY': "", 'GROQ_MODEL': "llama3-8b-8192",
    'HF_API_KEY': "", 'HF_MODEL': "deepseek-ai/DeepSeek-V3",
    'OLLAMA_BASE_URL': "http://localhost:11434", 'OLLAMA_MODEL': "llama3",
    'LLM_SYMBOL_CACHE_TTL': 90, 'LLM_MARKET_CACHE_TTL': 300,
    'LLM_MIN_CONFIDENCE': 0.50, 'LLM_MAX_TOKENS': 120, 'LLM_TEMPERATURE': 0.05,
    'LLM_WARMUP_TRADES': 10, 'LLM_OVERRIDE_WIN_RATE': 0.65,
    'LLM_OVERRIDE_CONFIDENCE': 0.82,
    'TAVILY_API_KEY': "",
    'LLM_MAX_PAIRS_PER_CYCLE': 5,
    'LLM_RETRY_MAX_ATTEMPTS': 3, 'LLM_RETRY_BASE_DELAY': 2,
    'LLM_INTER_SYMBOL_DELAY': 0.5,
    'VOLATILITY_LOOKBACK': 20,
    'MANUAL_TRADE_CONFIDENCE': 0.70,
    'MAX_HOLD_DAYS_MANUAL': 30,
    'LLM_CACHE_TTL': 90,
    'LLM_PREDICT_ENTRY_ENABLED': True,
    'LLM_PREDICT_ENTRY_CONFIDENCE': 0.60,
    # Adaptive Risk defaults
    'RISK_ATR_LOOKBACK': 100,
    'RISK_IMBALANCE_THRESHOLD': 0.25,
    'RISK_BODY_RATIO_THRESHOLD': 0.6,
    'RISK_ADX_STRONG': 40,
    'DYNAMIC_ATR_PROFILES': {
        "low_vol": {
            "stop":    1.0,
            "take":    2.5,
            "partial": 1.5,
            "trailing":0.8,
            "hard":    1.5,
        },
        "normal": {
            "stop":    1.5,
            "take":    3.0,
            "partial": 2.0,
            "trailing":1.0,
            "hard":    2.0,
        },
        "high_vol": {
            "stop":    2.0,
            "take":    4.0,
            "partial": 2.5,
            "trailing":1.2,
            "hard":    2.5,
        },
        "chaotic": {
            "stop":    1.8,
            "take":    3.5,
            "partial": 2.3,
            "trailing":1.1,
            "hard":    2.3,
        },
        "trending_strong": {
            "stop":    2.2,
            "take":    4.5,
            "partial": 2.8,
            "trailing":1.4,
            "hard":    2.8,
        },
    },
    # ADM defaults
    'ADM_ENABLED': True,
    'ADM_MODEL_PATH': "logs/adm_final/adm_quad_final.pt",
    'ADM_SCALER_PATH': "logs/adm_final/adm_quad_final_scaler.joblib",
    'ADM_CONFIDENCE_THRESHOLD': 0.55,
    'ADM_BOOST_FACTOR': 1.1,
    'ADM_DAMPEN_FACTOR': 0.9,
}
for _k, _v in _CONFIG_DEFAULTS.items():
    if _k not in globals():
        globals()[_k] = _v

# Paths
Path("logs").mkdir(exist_ok=True)
Path(ML_SAVE_PATH).mkdir(parents=True, exist_ok=True)
STATE_FILE = Path("logs/state.json")
TRADES_FILE = Path("logs/trades.jsonl")
VETO_MODEL_PATH = Path(f"{ML_SAVE_PATH}/veto_model.joblib")
VETO_SCALER_PATH = Path(f"{ML_SAVE_PATH}/veto_scaler.joblib")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("CryptoBot")

# ─────────────────────────────── TELEGRAM ───────────────────────────────
class Telegram:
    _url = (f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            if TELEGRAM_TOKEN else None)

    @staticmethod
    def send(text: str):
        if not Telegram._url: return
        try:
            requests.post(Telegram._url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=5)
        except Exception: pass

    @staticmethod
    def trade(symbol, direction, price, qty, conf, pnl=None, reason=None):
        if pnl is None:
            Telegram.send(f"<b>🟢 OPEN</b> {symbol} {direction} @ {price:.4f} "
                         f"qty={qty:.5f} conf={conf:.1%}")
        else:
            e = "✅" if pnl >= 0 else "❌"
            Telegram.send(f"{e} <b>CLOSE</b> {symbol} {direction} "
                         f"PnL={pnl:+.4f} USDT [{reason}]")

    @staticmethod
    def alert(msg: str):
        Telegram.send(f"🚨 <b>ALERT</b> {msg}")

# ─────────────────────────────── INDICATORS ───────────────────────────────
def _ema(prices: np.ndarray, span: int) -> float:
    if len(prices) < span: return float(prices[-1])
    alpha = 2.0 / (span + 1)
    val   = float(prices[0])
    for p in prices[1:]:
        val = alpha * float(p) + (1 - alpha) * val
    return val

def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1: return 50.0
    delta = np.diff(prices.astype(float))
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    ag = np.mean(gain[:period])
    al = np.mean(loss[:period])
    for i in range(period, len(gain)):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
    return float(100 - 100 / (1 + ag / (al + 1e-9)))

def _atr(candles: list, period: int = 14) -> float:
    if len(candles) < period + 1: return 0.0
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs: return 0.0
    atr_val = np.mean(trs[:period])
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return float(atr_val)

def _adx(candles: list, period: int = 14) -> Tuple[float, float, float]:
    if len(candles) < period * 2 + 1:
        return 0.0, 0.0, 0.0
    trs, pdms, ndms = [], [], []
    for i in range(1, len(candles)):
        h, l, ph, pl = (candles[i]["high"], candles[i]["low"],
                        candles[i-1]["high"], candles[i-1]["low"])
        pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdms.append(up if up > dn and up > 0 else 0)
        ndms.append(dn if dn > up and dn > 0 else 0)
    def wilder(arr):
        s = np.mean(arr[:period])
        res = [s]
        for v in arr[period:]:
            s = (s * (period - 1) + v) / period
            res.append(s)
        return np.array(res)
    str_ = wilder(trs)
    pdi  = 100 * wilder(pdms) / (str_ + 1e-9)
    ndi  = 100 * wilder(ndms) / (str_ + 1e-9)
    dx   = 100 * np.abs(pdi - ndi) / (pdi + ndi + 1e-9)
    adx_val = np.mean(dx[:period])
    for v in dx[period:]:
        adx_val = (adx_val * (period - 1) + v) / period
    return float(adx_val), float(pdi[-1]), float(ndi[-1])

def _stoch_rsi(prices: np.ndarray, rsi_p=14, stoch_p=14, k_p=3) -> Tuple[float,float]:
    if len(prices) < rsi_p + stoch_p + k_p: return 50.0, 50.0
    rsi_vals = []
    for i in range(rsi_p, len(prices)):
        rsi_vals.append(_rsi(prices[max(0,i-rsi_p-10):i+1], rsi_p))
    if len(rsi_vals) < stoch_p + k_p: return 50.0, 50.0
    rsi_arr = np.array(rsi_vals)
    k_vals = []
    for i in range(stoch_p - 1, len(rsi_arr)):
        window = rsi_arr[i-stoch_p+1:i+1]
        lo, hi = window.min(), window.max()
        k_vals.append(100 * (rsi_arr[i] - lo) / (hi - lo + 1e-9))
    if len(k_vals) < k_p: return 50.0, 50.0
    k = float(np.mean(k_vals[-k_p:]))
    d = float(np.mean(k_vals[-k_p*2:-k_p]) if len(k_vals) >= k_p*2 else k)
    return k, d

def _bollinger(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    if len(prices) < period:
        m = float(np.mean(prices))
        return m, m, m, 0.5
    win  = prices[-period:]
    mid  = float(np.mean(win))
    std  = float(np.std(win))
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pos   = (prices[-1] - lower) / (upper - lower + 1e-9)
    return float(upper), float(mid), float(lower), float(pos)

def _compute_all(candles: list) -> dict:
    closes  = np.array([c["close"]  for c in candles], dtype=float)
    highs   = np.array([c["high"]   for c in candles], dtype=float)
    lows    = np.array([c["low"]    for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)
    opens   = np.array([c["open"]   for c in candles], dtype=float)
    n       = len(closes)

    atr14   = _atr(candles, 14)
    adx14, pdi, ndi = _adx(candles, 14)
    rsi14   = _rsi(closes, 14)
    sk, sd  = _stoch_rsi(closes)
    bb_up, bb_mid, bb_low, bb_pos = _bollinger(closes, 20, 2.0)
    ema9    = _ema(closes, 9)
    ema12   = _ema(closes, 12)
    ema20   = _ema(closes, 20)
    ema21   = _ema(closes, 21)
    ema26   = _ema(closes, 26)
    ema50   = _ema(closes, 50)
    ema200  = _ema(closes, 200)

    avg_vol  = float(np.mean(volumes[-21:-1])) if n >= 21 else float(volumes[-1])
    vol_ratio = float(volumes[-1]) / (avg_vol + 1e-9)

    if n >= VOLATILITY_LOOKBACK + 1:
        rets = np.diff(np.log(closes[-VOLATILITY_LOOKBACK-1:]))
        realized_vol = float(np.std(rets) * np.sqrt(252 * 288))
    else:
        realized_vol = atr14 / (closes[-1] + 1e-9) * 10

    if n >= 20:
        win20 = closes[-20:]
        zscore = float((closes[-1] - win20.mean()) / (win20.std() + 1e-9))
    else:
        zscore = 0.0

    def ret(k):
        return float((closes[-1] - closes[-k]) / (closes[-k] + 1e-9)) if n >= k else 0.0

    if n >= 14:
        hi14 = highs[-14:].max()
        lo14 = lows[-14:].min()
        williams_r = float(-100 * (hi14 - closes[-1]) / (hi14 - lo14 + 1e-9))
    else:
        williams_r = -50.0

    last = candles[-1]
    body      = last["close"] - last["open"]
    candle_range = last["high"] - last["low"] + 1e-9
    body_pct  = abs(body) / candle_range
    upper_wick = last["high"] - max(last["open"], last["close"])
    lower_wick = min(last["open"], last["close"]) - last["low"]

    return {
        "closes": closes, "highs": highs, "lows": lows,
        "volumes": volumes, "opens": opens,
        "price":    float(closes[-1]),
        "atr14":    atr14,
        "adx14":    adx14, "pdi": pdi, "ndi": ndi,
        "rsi14":    rsi14,
        "stoch_k":  sk, "stoch_d": sd,
        "bb_upper": bb_up, "bb_mid": bb_mid, "bb_lower": bb_low,
        "bb_pos":   bb_pos,
        "ema9":     ema9, "ema12": ema12, "ema20": ema20,
        "ema21":    ema21, "ema26": ema26, "ema50": ema50, "ema200": ema200,
        "avg_vol":  avg_vol, "vol_ratio": vol_ratio,
        "zscore":   zscore,
        "realized_vol": realized_vol,
        "ret1":     ret(2),
        "ret3":     ret(4),
        "ret6":     ret(7),
        "ret12":    ret(13),
        "williams_r": williams_r,
        "body": body, "body_pct": body_pct,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
        "candle_range": candle_range,
    }

# ════════════════════════════════════ LAYER 1: PRICE MOMENTUM ════════════════════════════════════
class PriceMomentumLayer:
    def __init__(self):
        self._score_history = deque(maxlen=200)

    def _trend_score(self, ind: dict) -> float:
        sep = (ind["ema9"] - ind["ema21"]) / (ind["ema21"] + 1e-9)
        macd = (ind["ema12"] - ind["ema26"]) / (ind["ema26"] + 1e-9)
        return float(np.clip((sep + macd) / 0.010, -1.0, 1.0))

    def _adx_score(self, ind: dict) -> float:
        if ind["adx14"] < 15: return 0.0
        direction = 1.0 if ind["pdi"] > ind["ndi"] else -1.0
        strength  = min(1.0, (ind["adx14"] - 15) / 50)
        return direction * strength

    def _stoch_score(self, ind: dict) -> float:
        k = ind["stoch_k"]
        if k > 80: return -min(1.0, (k - 80) / 20)
        if k < 20: return  min(1.0, (20 - k) / 20)
        return (50 - k) / 50 * 0.3

    def _mean_reversion_score(self, ind: dict) -> float:
        z     = np.clip(ind["zscore"] / 3.0, -1.0, 1.0)
        bb    = (0.5 - ind["bb_pos"]) * 2
        rsi_s = (50 - ind["rsi14"]) / 50
        if z < 0 and bb > 0 and rsi_s > 0:
            return float(min(1.0, (abs(z) + bb + rsi_s) / 3))
        if z > 0 and bb < 0 and rsi_s < 0:
            return float(max(-1.0, -(abs(z) + abs(bb) + abs(rsi_s)) / 3))
        return 0.0

    def _breakout_score(self, ind: dict, candles: list) -> float:
        price = ind["price"]
        if ind["vol_ratio"] < 1.8 or ind["adx14"] < 20:
            return 0.0
        if price > ind["bb_upper"]:
            excess = (price - ind["bb_upper"]) / (ind["atr14"] + 1e-9)
            return float(min(1.0, excess))
        if price < ind["bb_lower"]:
            excess = (ind["bb_lower"] - price) / (ind["atr14"] + 1e-9)
            return float(max(-1.0, -excess))
        return 0.0

    def _ichimoku_score(self, ind: dict, candles: list) -> float:
        if len(candles) < 52: return 0.0
        highs = [c["high"] for c in candles]
        lows  = [c["low"]  for c in candles]
        tenkan  = (max(highs[-9:])  + min(lows[-9:]))  / 2
        kijun   = (max(highs[-26:]) + min(lows[-26:])) / 2
        span_a  = (tenkan + kijun) / 2
        span_b  = (max(highs[-52:]) + min(lows[-52:])) / 2
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        price = ind["price"]
        if price > cloud_top:
            dist = (price - cloud_top) / (ind["atr14"] + 1e-9)
            return float(min(1.0, dist * 0.5))
        if price < cloud_bot:
            dist = (cloud_bot - price) / (ind["atr14"] + 1e-9)
            return float(max(-1.0, -dist * 0.5))
        return 0.0

    def score(self, ind: dict, candles: list) -> dict:
        if len(candles) < 60:
            return {"raw_score": 0.0, "confidence": 0.0, "signal": "HOLD"}

        components = {
            "trend":    (self._trend_score(ind),             0.25),
            "adx":      (self._adx_score(ind),               0.20),
            "stoch":    (self._stoch_score(ind),              0.15),
            "mean_rev": (self._mean_reversion_score(ind),     0.20),
            "breakout": (self._breakout_score(ind, candles),  0.10),
            "ichimoku": (self._ichimoku_score(ind, candles),  0.10),
        }

        raw_score = sum(s * w for s, w in components.values())
        self._score_history.append(abs(raw_score))

        if len(self._score_history) >= 20:
            pct = np.percentile(list(self._score_history), CALIBRATION_PERCENTILE)
            confidence = min(1.0, abs(raw_score) / (pct + 1e-9)) * 0.85
        else:
            confidence = abs(raw_score) * 0.6

        direction = "BUY" if raw_score > 0 else "SELL"
        signal    = direction if confidence > 0.45 else "HOLD"

        return {
            "raw_score":  float(raw_score),
            "confidence": float(confidence),
            "signal":     signal,
            "components": {k: round(v[0], 3) for k, v in components.items()},
        }

# ════════════════════════════════════ LAYER 2: MICROSTRUCTURE ════════════════════════════════════
class SessionVWAP:
    def __init__(self):
        self._reset_date = None
        self._cum_tp_vol = 0.0
        self._cum_vol    = 0.0
        self._vwap       = None

    def update(self, candle: dict):
        today = datetime.utcnow().date()
        if self._reset_date != today:
            self._reset_date = today
            self._cum_tp_vol = 0.0
            self._cum_vol    = 0.0
            self._vwap       = None
        tp = (candle["high"] + candle["low"] + candle["close"]) / 3
        self._cum_tp_vol += tp * candle["volume"]
        self._cum_vol    += candle["volume"]
        self._vwap = self._cum_tp_vol / (self._cum_vol + 1e-9)

    def deviation(self, price: float, atr: float) -> float:
        if self._vwap is None or atr < 1e-9:
            return 0.0
        return (price - self._vwap) / atr

    @property
    def value(self):
        return self._vwap

class OrderBookAnalyzer:
    def __init__(self, depth=OB_DEPTH_LEVELS, cache_ttl=OB_CACHE_TTL):
        self.depth      = depth
        self.cache_ttl  = cache_ttl
        self._cache     = {}
        self._imb_hist  = deque(maxlen=200)

    def _fetch(self, symbol: str) -> Optional[dict]:
        now    = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached["ts"] < self.cache_ttl:
            return cached["data"]
        try:
            url = (f"https://fapi.binance.com/fapi/v1/depth"
                   f"?symbol={symbol}&limit={self.depth}")
            r   = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                self._cache[symbol] = {"data": data, "ts": now}
                return data
        except Exception:
            pass
        return None

    def analyze(self, symbol: str) -> dict:
        data = self._fetch(symbol)
        if not data:
            return {"signal": "HOLD", "confidence": 0.0,
                    "imbalance": 0.0, "spread_pct": 0.0}
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
        if not bids or not asks:
            return {"signal": "HOLD", "confidence": 0.0,
                    "imbalance": 0.0, "spread_pct": 0.0}
        mid   = (bids[0][0] + asks[0][0]) / 2
        spread_pct = (asks[0][0] - bids[0][0]) / (mid + 1e-9)
        bid_vol = sum(q * (1 - i * 0.03) for i, (p, q) in enumerate(bids[:10]))
        ask_vol = sum(q * (1 - i * 0.03) for i, (p, q) in enumerate(asks[:10]))
        total   = bid_vol + ask_vol + 1e-9
        imb     = (bid_vol - ask_vol) / total
        self._imb_hist.append(abs(imb))
        if len(self._imb_hist) >= 20:
            threshold = np.percentile(list(self._imb_hist), 70)
        else:
            threshold = OB_IMBALANCE_SEED
        if abs(imb) < threshold:
            return {"signal": "HOLD", "confidence": 0.0,
                    "imbalance": imb, "spread_pct": spread_pct}
        conf = min(0.90, abs(imb) / (threshold + 1e-9) * 0.6)
        sig  = "BUY" if imb > 0 else "SELL"
        return {"signal": sig, "confidence": conf,
                "imbalance": imb, "spread_pct": spread_pct}

class OrderFlowAnalyzer:
    def __init__(self, lookback=OF_TRADE_LOOKBACK, cache_ttl=OF_CACHE_TTL):
        self.lookback   = lookback
        self.cache_ttl  = cache_ttl
        self._cache     = {}
        self._imb_hist  = deque(maxlen=200)

    def _fetch(self, symbol: str) -> Optional[list]:
        now    = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached["ts"] < self.cache_ttl:
            return cached["data"]
        try:
            url = (f"https://fapi.binance.com/fapi/v1/trades"
                   f"?symbol={symbol}&limit={self.lookback}")
            r   = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                self._cache[symbol] = {"data": data, "ts": now}
                return data
        except Exception:
            pass
        return None

    def analyze(self, symbol: str) -> dict:
        trades = self._fetch(symbol)
        if not trades:
            return {"signal": "HOLD", "confidence": 0.0, "imbalance": 0.0}
        buy_notional  = sum(float(t["qty"]) * float(t["price"])
                           for t in trades if not t.get("isBuyerMaker", True))
        sell_notional = sum(float(t["qty"]) * float(t["price"])
                           for t in trades if t.get("isBuyerMaker", True))
        total = buy_notional + sell_notional + 1e-9
        imb   = (buy_notional - sell_notional) / total
        recent_half = trades[len(trades)//2:]
        buy_r  = sum(float(t["qty"]) for t in recent_half if not t.get("isBuyerMaker",True))
        sell_r = sum(float(t["qty"]) for t in recent_half if t.get("isBuyerMaker",True))
        accel  = (buy_r - sell_r) / (buy_r + sell_r + 1e-9)
        self._imb_hist.append(abs(imb))
        if len(self._imb_hist) >= 20:
            threshold = np.percentile(list(self._imb_hist), 70)
        else:
            threshold = OF_IMBALANCE_SEED
        if abs(imb) < threshold:
            return {"signal": "HOLD", "confidence": 0.0, "imbalance": imb}
        base_conf = min(0.88, abs(imb) / (threshold + 1e-9) * 0.55)
        if (imb > 0 and accel > 0.1) or (imb < 0 and accel < -0.1):
            base_conf = min(0.90, base_conf * 1.15)
        sig = "BUY" if imb > 0 else "SELL"
        return {"signal": sig, "confidence": base_conf, "imbalance": imb}

class MicrostructureLayer:
    def __init__(self):
        self.vwap_trackers: Dict[str, SessionVWAP] = {}
        self.ob    = OrderBookAnalyzer()
        self.flow  = OrderFlowAnalyzer()

    def _get_vwap(self, symbol: str) -> SessionVWAP:
        if symbol not in self.vwap_trackers:
            self.vwap_trackers[symbol] = SessionVWAP()
        return self.vwap_trackers[symbol]

    def update_vwap(self, symbol: str, candle: dict):
        self._get_vwap(symbol).update(candle)

    def analyze(self, symbol: str, ind: dict, candles: list,
                live_ob_imb: Optional[float] = None,
                live_of_imb: Optional[float] = None) -> dict:
        price = ind["price"]
        atr   = ind["atr14"]
        if candles:
            self.update_vwap(symbol, candles[-1])
        vwap_tracker = self._get_vwap(symbol)
        vwap_dev_atr = vwap_tracker.deviation(price, atr)
        vwap_sig  = "HOLD"
        vwap_conf = 0.0
        if vwap_dev_atr < -VWAP_MIN_DEVIATION_ATR and abs(vwap_dev_atr) < VWAP_MAX_DEVIATION_ATR:
            depth     = abs(vwap_dev_atr) - VWAP_MIN_DEVIATION_ATR
            vwap_conf = min(0.90, 0.55 + depth * 0.12)
            vwap_sig  = "BUY"
        elif vwap_dev_atr > VWAP_MIN_DEVIATION_ATR and abs(vwap_dev_atr) < VWAP_MAX_DEVIATION_ATR:
            depth     = vwap_dev_atr - VWAP_MIN_DEVIATION_ATR
            vwap_conf = min(0.90, 0.55 + depth * 0.12)
            vwap_sig  = "SELL"
        if live_ob_imb is not None and live_of_imb is not None:
            ob_imb = live_ob_imb
            flow_imb = live_of_imb
            ob_sig = "BUY" if ob_imb > 0.15 else "SELL" if ob_imb < -0.15 else "HOLD"
            ob_conf = min(0.88, abs(ob_imb) * 1.2)
            flow_sig = "BUY" if flow_imb > 0.15 else "SELL" if flow_imb < -0.15 else "HOLD"
            flow_conf = min(0.88, abs(flow_imb) * 1.2)
        else:
            ob_res = self.ob.analyze(symbol)
            ob_sig = ob_res["signal"]
            ob_conf = ob_res["confidence"]
            ob_imb = ob_res["imbalance"]
            flow_res = self.flow.analyze(symbol)
            flow_sig = flow_res["signal"]
            flow_conf = flow_res["confidence"]
            flow_imb = flow_res["imbalance"]
        signals = [
            (vwap_sig,  vwap_conf,  "VWAP"),
            (ob_sig,    ob_conf,    "OrderBook"),
            (flow_sig,  flow_conf,  "OrderFlow"),
        ]
        buy_confs  = [(c, n) for s, c, n in signals if s == "BUY"]
        sell_confs = [(c, n) for s, c, n in signals if s == "SELL"]
        direction = None
        if len(buy_confs)  >= 2: direction = "BUY"
        if len(sell_confs) >= 2: direction = "SELL"
        if direction:
            opposite = sell_confs if direction == "BUY" else buy_confs
            if opposite and max(c for c, _ in opposite) > 0.65:
                direction = None
        if not direction:
            return {
                "signal": "HOLD", "confidence": 0.0,
                "vwap_dev_atr": vwap_dev_atr,
                "vwap": vwap_tracker.value,
                "ob_imbalance": ob_imb,
                "flow_imbalance": flow_imb,
                "components": {"VWAP": vwap_sig, "OrderBook": ob_sig, "OrderFlow": flow_sig},
            }
        agreeing_confs = buy_confs if direction == "BUY" else sell_confs
        avg_conf = float(np.mean([c for c, _ in agreeing_confs]))
        if len(agreeing_confs) == 3:
            avg_conf = min(0.92, avg_conf * 1.12)
        return {
            "signal":        direction,
            "confidence":    avg_conf,
            "vwap_dev_atr":  vwap_dev_atr,
            "vwap":          vwap_tracker.value,
            "ob_imbalance":  ob_imb,
            "flow_imbalance": flow_imb,
            "agreeing":      len(agreeing_confs),
            "components":    {"VWAP": vwap_sig, "OrderBook": ob_sig, "OrderFlow": flow_sig},
        }

# ════════════════════════════════════ LAYER 3: CONTEXT ════════════════════════════════════
class ContextLayer:
    def __init__(self):
        self._asset_cache  = {}
        self._cache_ts     = {}
        self._regime_hist  = deque(maxlen=50)

    def _fetch_trend(self, symbol: str) -> str:
        now = time.time()
        if (symbol in self._cache_ts and
                now - self._cache_ts[symbol] < CONTEXT_CACHE_TTL):
            return self._asset_cache.get(symbol, "flat")
        try:
            url = (f"https://fapi.binance.com/fapi/v1/klines"
                   f"?symbol={symbol}&interval={CONTEXT_INTERVAL}&limit=30")
            r   = requests.get(url, timeout=3)
            if r.status_code == 200:
                closes = np.array([float(k[4]) for k in r.json()])
                fast   = _ema(closes, CONTEXT_EMA_FAST)
                slow   = _ema(closes, CONTEXT_EMA_SLOW)
                sep    = (fast - slow) / (slow + 1e-9)
                trend  = "up" if sep > 0.002 else "down" if sep < -0.002 else "flat"
                self._asset_cache[symbol] = trend
                self._cache_ts[symbol]    = now
                return trend
        except Exception:
            pass
        return "flat"

    def _session_score(self) -> Tuple[str, float]:
        h = datetime.utcnow().hour
        if   7  <= h < 10:  return "london_open",       SESSION_SCORES["london_open"]
        elif 13 <= h < 15:  return "london_us_overlap",  SESSION_SCORES["london_us_overlap"]
        elif 15 <= h < 17:  return "us_open",            SESSION_SCORES["us_open"]
        elif 17 <= h < 20:  return "us_afternoon",       SESSION_SCORES["us_afternoon"]
        elif 10 <= h < 13:  return "london_midday",      SESSION_SCORES["london_midday"]
        elif 20 <= h < 23:  return "us_close",           SESSION_SCORES["us_close"]
        elif 1  <= h < 6:   return "dead_hours",         SESSION_SCORES["dead_hours"]
        else:               return "asian_session",      SESSION_SCORES["asian_session"]

    def _detect_regime(self, ind: dict, candles: list) -> Tuple[str, float]:
        adx_val = ind["adx14"]
        vol_pct = ind["atr14"] / (ind["price"] + 1e-9)
        if adx_val >= REGIME_ADX_STRONG:
            return "TRENDING", min(1.0, (adx_val - REGIME_ADX_STRONG) / 30 + 0.7)
        if adx_val >= REGIME_ADX_TRENDING:
            return "TRENDING", min(0.7, (adx_val - REGIME_ADX_TRENDING) / 15 + 0.4)
        if vol_pct > REGIME_VOL_HIGH:
            return "VOLATILE", min(1.0, vol_pct / REGIME_VOL_HIGH - 1 + 0.5)
        return "RANGING", min(0.8, 1.0 - adx_val / REGIME_ADX_TRENDING)

    def analyze(self, symbol: str, ind: dict, candles: list) -> dict:
        trends    = {s: self._fetch_trend(s) for s in CONTEXT_ASSETS if s != symbol}
        up_count  = sum(1 for t in trends.values() if t == "up")
        dn_count  = sum(1 for t in trends.values() if t == "down")
        n_assets  = len(trends)
        if up_count >= n_assets * 0.7:
            macro_bias = "BUY";  macro_conf = up_count / n_assets
        elif dn_count >= n_assets * 0.7:
            macro_bias = "SELL"; macro_conf = dn_count / n_assets
        else:
            macro_bias = "NEUTRAL"; macro_conf = 0.5
        regime, regime_conf = self._detect_regime(ind, candles)
        self._regime_hist.append(regime)
        session_name, session_score = self._session_score()
        context_conf = (macro_conf * 0.4 + regime_conf * 0.3
                        + min(session_score, 1.0) * 0.3)
        return {
            "macro_bias":    macro_bias,
            "macro_conf":    macro_conf,
            "regime":        regime,
            "regime_conf":   regime_conf,
            "session_name":  session_name,
            "session_score": session_score,
            "context_conf":  context_conf,
            "asset_trends":  trends,
        }

# ════════════════════════════════════ LAYER 4: ML VETO ════════════════════════════════════
class MLVetoLayer:
    FEATURE_NAMES = [
        "pm_score", "pm_confidence",
        "vwap_dev_atr", "ob_imbalance", "flow_imbalance", "micro_conf",
        "session_score", "regime_encoded", "macro_encoded",
        "atr_pct", "vol_ratio", "rsi14", "adx14",
        "realized_vol", "spread_pct",
        "consecutive_losses", "hour_sin", "hour_cos",
    ]
    def __init__(self):
        self._X_buf  = deque(maxlen=ML_FEATURE_WINDOW)
        self._y_buf  = deque(maxlen=ML_FEATURE_WINDOW)
        self._model  = None
        self._scaler = RobustScaler() if ML_OK else None
        self._trained = False
        self._lock   = threading.Lock()
        self._update_count = 0
        self._try_load()

    def _try_load(self):
        if VETO_MODEL_PATH.exists() and VETO_SCALER_PATH.exists() and ML_OK:
            try:
                self._model  = joblib.load(VETO_MODEL_PATH)
                self._scaler = joblib.load(VETO_SCALER_PATH)
                self._trained = True
                log.info("[ML] Loaded saved veto model")
            except Exception as e:
                log.warning(f"[ML] Load failed: {e}")

    def _save(self):
        try:
            joblib.dump(self._model,  VETO_MODEL_PATH)
            joblib.dump(self._scaler, VETO_SCALER_PATH)
        except Exception: pass

    def extract_features(self, pm_result: dict, micro_result: dict,
                          ctx_result: dict, ind: dict,
                          consecutive_losses: int,
                          spread_pct: float = 0.0) -> np.ndarray:
        h    = datetime.utcnow().hour
        regime_enc = {"TRENDING": 1.0, "RANGING": 0.0, "VOLATILE": 0.5}.get(
            ctx_result.get("regime", "RANGING"), 0.0)
        macro_enc  = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0}.get(
            ctx_result.get("macro_bias", "NEUTRAL"), 0.0)
        return np.array([
            pm_result.get("raw_score", 0.0),
            pm_result.get("confidence", 0.0),
            micro_result.get("vwap_dev_atr", 0.0),
            micro_result.get("ob_imbalance", 0.0),
            micro_result.get("flow_imbalance", 0.0),
            micro_result.get("confidence", 0.0),
            ctx_result.get("session_score", 1.0),
            regime_enc,
            macro_enc,
            ind["atr14"] / (ind["price"] + 1e-9),
            ind["vol_ratio"],
            ind["rsi14"] / 100.0,
            ind["adx14"] / 100.0,
            min(ind.get("realized_vol", 0.5), 2.0),
            spread_pct * 100,
            min(consecutive_losses, 6) / 6.0,
            math.sin(2 * math.pi * h / 24),
            math.cos(2 * math.pi * h / 24),
        ], dtype=float)

    def record_outcome(self, features: np.ndarray, trade_won: bool):
        with self._lock:
            self._X_buf.append(features)
            self._y_buf.append(1 if trade_won else 0)
            self._update_count += 1
        if (self._update_count % ML_TRAIN_INTERVAL == 0 and
                len(self._X_buf) >= ML_VETO_MIN_SAMPLES):
            threading.Thread(target=self.train, daemon=True).start()

    def train(self):
        if not ML_OK: return
        with self._lock:
            if len(self._X_buf) < ML_VETO_MIN_SAMPLES: return
            X = np.array(list(self._X_buf))
            y = np.array(list(self._y_buf))
        if y.mean() < 0.1 or y.mean() > 0.9:
            log.warning(f"[ML] Imbalanced labels: {y.mean():.2f}")
        Xs = self._scaler.fit_transform(X)
        if XGB_OK:
           base = xgb.XGBClassifier(
           n_estimators=200, max_depth=4, learning_rate=0.05,
           subsample=0.8, colsample_bytree=0.8,
           min_child_weight=5, gamma=1.0,
           eval_metric="logloss",
           random_state=42)
        else:
            base = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42)
        model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        try:
            model.fit(Xs, y)
            with self._lock:
                self._model   = model
                self._trained = True
            self._save()
            wr = y.mean()
            log.info(f"[ML] Trained: {len(X)} samples  "
                     f"historical_WR={wr:.2f}  "
                     f"model_acc={model.score(Xs, y):.2f}")
        except Exception as e:
            log.error(f"[ML] Train error: {e}")

    def should_veto(self, features: np.ndarray) -> Tuple[bool, float]:
        if not ML_VETO_ENABLED:
            return False, 0.5
        if not ML_OK or not self._trained or self._model is None:
            return False, 0.5
        if len(self._X_buf) < ML_VETO_MIN_SAMPLES:
            return False, 0.5
        try:
            Xs = self._scaler.transform(features.reshape(1, -1))
            proba = self._model.predict_proba(Xs)[0]
            p_loss = float(proba[0]) if len(proba) > 1 else float(1 - proba[0])
            veto   = p_loss > ML_VETO_LOSS_THRESHOLD
            return veto, p_loss
        except Exception as e:
            log.warning(f"[ML] predict error: {e}")
            return False, 0.5

# ════════════════════════ CONFIDENCE CALIBRATOR ════════════════════════
class ConfidenceCalibrator:
    N_BINS = 10
    def __init__(self):
        self._bins = [deque(maxlen=100) for _ in range(self.N_BINS)]
        self._threshold = MIN_CONFIDENCE
    def _bin_idx(self, conf: float) -> int:
        return min(int(conf * self.N_BINS), self.N_BINS - 1)
    def record(self, entry_conf: float, won: bool):
        idx = self._bin_idx(entry_conf)
        self._bins[idx].append(1 if won else 0)
        self._update_threshold()
    def _update_threshold(self):
        for i in range(self.N_BINS - 1, -1, -1):
            if len(self._bins[i]) >= 8:
                wr = np.mean(list(self._bins[i]))
                if wr >= 0.60:
                    self._threshold = max(0.50, i / self.N_BINS)
                    return
        self._threshold = MIN_CONFIDENCE
    @property
    def threshold(self) -> float:
        return self._threshold
    def summary(self) -> dict:
        result = {}
        for i, b in enumerate(self._bins):
            if len(b) >= 5:
                low  = i / self.N_BINS
                high = (i + 1) / self.N_BINS
                result[f"{low:.1f}-{high:.1f}"] = round(float(np.mean(list(b))), 3)
        return result

# ════════════════════════ KELLY POSITION SIZER ════════════════════════
class KellyPositionSizer:
    def __init__(self, lookback=KELLY_LOOKBACK):
        self.lookback = lookback
        self._outcomes = deque(maxlen=lookback)
    def record(self, pnl_pct: float, won: bool):
        self._outcomes.append((pnl_pct, won))
    def compute_fraction(self) -> float:
        if len(self._outcomes) < 20:
            return KELLY_FLOOR
        wins   = [(p, w) for p, w in self._outcomes if w]
        losses = [(p, w) for p, w in self._outcomes if not w]
        if not wins or not losses:
            return KELLY_FLOOR
        win_rate  = len(wins) / len(self._outcomes)
        avg_win   = float(np.mean([abs(p) for p, _ in wins]))
        avg_loss  = float(np.mean([abs(p) for p, _ in losses]))
        if avg_loss < 1e-9: return KELLY_MAX_FRACTION
        b       = avg_win / avg_loss
        kelly_f = (b * win_rate - (1 - win_rate)) / b
        kelly_f = max(0, kelly_f) * 0.5
        return float(np.clip(kelly_f, KELLY_FLOOR, KELLY_MAX_FRACTION))
    def position_size(self, equity: float, entry_price: float,
                      atr: float, confidence: float) -> float:
        fraction    = self.compute_fraction()
        conf_scale  = 0.7 + 0.3 * confidence
        atr_scale   = min(1.0, 0.015 / (atr / (entry_price + 1e-9) + 1e-6))
        raw_size    = equity * fraction * conf_scale * atr_scale
        return max(MIN_TRADE_USDT, min(raw_size, equity * MAX_POSITION_PCT))

# ════════════════════════ LSTM AGENT ════════════════════════
if TORCH_OK:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size=10, hidden=64, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers,
                                batch_first=True, dropout=dropout)
            self.drop = nn.Dropout(dropout)
            self.fc   = nn.Linear(hidden, 1)
            self.sig  = nn.Sigmoid()
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.sig(self.fc(self.drop(o[:, -1, :]))).squeeze(-1)

    class LSTMAgent:
        name = "LSTM"
        SEQ  = LSTM_SEQ_LEN
        FEAT = 10
        def __init__(self, symbol: str):
            self.symbol = symbol
            sp = Path(ML_SAVE_PATH)
            sp.mkdir(parents=True, exist_ok=True)
            self._mfile   = sp / f"LSTM_{symbol}.pt"
            self._X       = deque(maxlen=500)
            self._y       = deque(maxlen=500)
            self._model   = _LSTMNet(self.FEAT)
            self._trained = False
            self._n_upd   = 0
            self._lock    = threading.Lock()
            self._load()

        def _load(self):
            if self._mfile.exists():
                try:
                    self._model.load_state_dict(
                        torch.load(self._mfile, map_location="cpu"))
                    self._model.eval()
                    self._trained = True
                except Exception: pass

        def _feats(self, candles: list) -> Optional[np.ndarray]:
            if len(candles) < self.SEQ + 2: return None
            win    = candles[-(self.SEQ + 1):]
            c_arr  = np.array([c["close"]  for c in win], dtype=float)
            v_arr  = np.array([c["volume"] for c in win], dtype=float)
            e9     = pd.Series(c_arr).ewm(span=9,  adjust=False).mean().values
            e21    = pd.Series(c_arr).ewm(span=21, adjust=False).mean().values
            vm     = pd.Series(v_arr).rolling(10, min_periods=1).mean().values
            rows   = []
            for i in range(1, len(win)):
                p, pp = c_arr[i], c_arr[i-1]
                ret   = (p - pp) / (pp + 1e-9)
                rows.append([
                    ret,
                    (win[i]["high"] - win[i]["low"]) / (pp + 1e-9),
                    (win[i]["close"] - win[i]["open"]) / (pp + 1e-9),
                    (p - e9[i])  / (e9[i]  + 1e-9),
                    (p - e21[i]) / (e21[i] + 1e-9),
                    v_arr[i] / (vm[i] + 1e-9) - 1.0,
                    np.clip(ret * 50, -3, 3),
                    (e9[i] - e21[i]) / (p + 1e-9),
                    (win[i]["high"] - max(win[i]["open"], win[i]["close"])) / (pp + 1e-9),
                    (min(win[i]["open"], win[i]["close"]) - win[i]["low"]) / (pp + 1e-9),
                ])
            arr = np.nan_to_num(np.array(rows, dtype=np.float32), 0, 1, -1)
            return np.clip(arr, -5, 5) if arr.shape[0] == self.SEQ else None

        def update(self, candles: list, next_close: float):
            f = self._feats(candles)
            if f is None: return
            with self._lock:
                self._X.append(f)
                self._y.append(1.0 if next_close > candles[-1]["close"] else 0.0)
                self._n_upd += 1

        def train(self):
            with self._lock:
                if len(self._X) < LSTM_MIN_SAMPLES: return
                X = np.stack(list(self._X))
                y = np.array(list(self._y), dtype=np.float32)
            Xt, yt  = torch.tensor(X), torch.tensor(y)
            model   = _LSTMNet(self.FEAT)
            opt     = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
            lf      = nn.BCELoss()
            model.train()
            for _ in range(30):
                idx = torch.randperm(len(Xt))
                for s in range(0, len(Xt), 32):
                    b = idx[s:s+32]
                    opt.zero_grad()
                    lf(model(Xt[b]), yt[b]).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
            model.eval()
            with self._lock:
                self._model   = model
                self._trained = True
            try: torch.save(model.state_dict(), self._mfile)
            except Exception: pass

        def predict(self, candles: list) -> float:
            if not self._trained: return 0.5
            f = self._feats(candles)
            if f is None: return 0.5
            with torch.no_grad():
                return float(self._model(torch.tensor(f).unsqueeze(0)).item())

else:
    class LSTMAgent:
        name = "LSTM"
        def __init__(self, symbol: str):
            self.symbol = symbol
            self._n_upd = 0
        def update(self, candles, next_close): self._n_upd += 1
        def train(self): pass
        def predict(self, candles) -> float: return 0.5

# ════════════════════════ ADM ENGINE ════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class PatternEngine(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.bn1 = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.drop1(out)
        residual = out
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.drop2(out)
        return out + residual

class LogicValidator(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden); self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        out = F.relu(self.bn(self.fc(x)))
        out = self.drop(out)
        return torch.sigmoid(self.out(out)).squeeze(-1)

class MainPredictor(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden); self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        out = F.relu(self.bn(self.fc(x)))
        out = self.drop(out)
        return torch.sigmoid(self.out(out)).squeeze(-1)

class CurveDrawer(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden); self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 3)
    def forward(self, x):
        out = F.relu(self.bn(self.fc(x)))
        out = self.drop(out)
        return torch.sigmoid(self.out(out))

class RegimeDetector(nn.Module):
    def __init__(self, in_dim, classes, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, 128); self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, classes)
    def forward(self, x):
        out = F.relu(self.fc(x))
        out = self.drop(out)
        return F.softmax(self.out(out), dim=-1)

class ADMQuadV2(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, cfg["lstm_hidden"], cfg["lstm_layers"],
                            batch_first=True, dropout=cfg["dropout"] if cfg["lstm_layers"]>1 else 0)
        self.attention = MultiHeadAttention(cfg["lstm_hidden"], cfg["attention_heads"])
        self.shared = nn.Sequential(nn.Linear(cfg["lstm_hidden"], cfg["shared_dim"]),
                                    nn.ReLU(), nn.Dropout(cfg["dropout"]))
        self.pattern = PatternEngine(cfg["shared_dim"], cfg["pattern_units"], cfg["dropout"])
        combined_dim = cfg["shared_dim"] + cfg["pattern_units"]
        self.logic = LogicValidator(combined_dim, cfg["logic_units"], cfg["dropout"])
        self.predict = MainPredictor(combined_dim, cfg["predict_units"], cfg["dropout"])
        self.curve = CurveDrawer(combined_dim, cfg["curve_units"], cfg["dropout"])
        self.regime = RegimeDetector(combined_dim, cfg["regime_classes"], cfg["dropout"])
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        last = attn_out[:, -1, :]
        shared = self.shared(last)
        pattern = self.pattern(shared)
        combined = torch.cat([shared, pattern], dim=1)
        gate = self.logic(combined)
        prob = self.predict(combined)
        curves = self.curve(combined)
        regime = self.regime(combined)
        return gate, prob, curves, regime

class ADMEngine:
    """Loads the ADM‑Quad V2 neural network and provides confidence multiplier."""
    def __init__(self):
        self.model = None
        self.scaler = None
        self._enabled = False
        if not ADM_ENABLED:
            return
        model_path = Path(ADM_MODEL_PATH)
        scaler_path = Path(ADM_SCALER_PATH)
        if not model_path.exists() or not scaler_path.exists():
            log.warning(f"[ADM] Model or scaler not found at {model_path} / {scaler_path}")
            return
        try:
            cfg = {
                "lstm_hidden": 128, "lstm_layers": 2, "attention_heads": 4,
                "shared_dim": 64, "pattern_units": 1000, "predict_units": 500,
                "logic_units": 250, "curve_units": 250, "regime_classes": 3,
                "dropout": 0.35,
            }
            self.model = ADMQuadV2(9, cfg).to('cpu')
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            self.scaler = joblib.load(scaler_path)
            self._enabled = True
            log.info("[ADM] ADM‑Quad V2 loaded successfully")
        except Exception as e:
            log.error(f"[ADM] Failed to load ADM model: {e}")

    FEATURE_NAMES = [
        "body_ratio", "upper_wick", "lower_wick", "vol_ratio",
        "volatility", "rsi", "adx", "atr_pct", "spread_approx"
    ]

    def _extract_features(self, ind: dict) -> np.ndarray:
        price = ind["price"]
        body   = abs(ind["body"]) / (ind["candle_range"] + 1e-9)
        upper_w = ind["upper_wick"] / (ind["candle_range"] + 1e-9)
        lower_w = ind["lower_wick"] / (ind["candle_range"] + 1e-9)
        vol_rat = ind["vol_ratio"]
        vol     = ind["candle_range"] / (price + 1e-9)
        rsi    = ind["rsi14"]
        adx    = ind["adx14"]
        atr_p  = ind["atr14"] / (price + 1e-9)
        spread = ind["candle_range"] / (price + 1e-9)
        return np.array([body, upper_w, lower_w, vol_rat, vol, rsi, adx, atr_p, spread], dtype=float)

    def predict_probability(self, ind: dict) -> float:
        if not self._enabled:
            return 0.5
        feats = self._extract_features(ind).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        feat_tensor = torch.tensor(feats_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(feat_tensor)[1].item()
        return prob

    def get_confidence_multiplier(self, ind: dict) -> float:
        if not self._enabled:
            return 1.0
        prob = self.predict_probability(ind)
        if prob >= ADM_CONFIDENCE_THRESHOLD:
            return ADM_BOOST_FACTOR
        else:
            return ADM_DAMPEN_FACTOR

# ════════════════════════ DECISION ENGINE ════════════════════════
class LayeredDecisionEngine:
    def __init__(self):
        self.calibrator = ConfidenceCalibrator()

    def decide(self, pm: dict, micro: dict, ctx: dict, veto_result: Tuple[bool, float]) -> dict:
        pm_sig    = pm.get("signal", "HOLD")
        pm_conf   = pm.get("confidence", 0.0)
        micro_sig = micro.get("signal", "HOLD")
        micro_conf= micro.get("confidence", 0.0)
        if micro_sig != "HOLD" and micro_sig != pm_sig:
            return {"signal": "HOLD", "confidence": 0.0,
                    "blocked_by": "microstructure_conflict",
                    "pm_signal": pm_sig, "micro_signal": micro_sig}
        final_dir = pm_sig if pm_sig != "HOLD" else micro_sig
        if final_dir == "HOLD":
            return {"signal": "HOLD", "confidence": 0.0, "blocked_by": "no_signal"}
        session_score = ctx.get("session_score", 1.0)
        if session_score < MIN_SESSION_SCORE:
            return {"signal": "HOLD", "confidence": 0.0,
                    "blocked_by": "session",
                    "session": ctx.get("session_name")}
        macro_bias = ctx.get("macro_bias", "NEUTRAL")
        macro_mult = 1.0
        if macro_bias != "NEUTRAL" and macro_bias != final_dir:
            macro_mult = 0.75
        w_pm    = WEIGHT_PRICE_MOMENTUM
        w_micro = WEIGHT_MICROSTRUCTURE
        w_ctx   = WEIGHT_CONTEXT
        ctx_conf = ctx.get("context_conf", 0.5)
        if macro_bias == final_dir:
            ctx_conf = min(0.95, ctx_conf * 1.2)
        if micro_sig == "HOLD" and micro_conf == 0.0:
            w_pm_eff  = w_pm  + w_micro * 0.5
            w_ctx_eff = w_ctx + w_micro * 0.5
            eff_micro_conf = 0.0
            combined_conf = (pm_conf  * w_pm_eff + ctx_conf * w_ctx_eff) * macro_mult
        else:
            eff_micro_conf = micro_conf if micro_sig == final_dir else micro_conf * 0.3
            combined_conf = (pm_conf * w_pm + eff_micro_conf * w_micro + ctx_conf * w_ctx) * macro_mult
        combined_conf = min(0.95, combined_conf * min(session_score, 1.3))
        vetoed, p_loss = veto_result
        if vetoed:
            return {"signal": "HOLD", "confidence": 0.0,
                    "blocked_by": "ml_veto", "p_loss": p_loss}
        threshold = self.calibrator.threshold
        if combined_conf < threshold:
            return {"signal": "HOLD", "confidence": combined_conf,
                    "blocked_by": f"below_threshold_{threshold:.2f}"}
        return {
            "signal":         final_dir,
            "confidence":     round(combined_conf, 4),
            "pm_signal":      pm_sig,
            "pm_conf":        round(pm_conf, 4),
            "micro_signal":   micro_sig,
            "micro_conf":     round(micro_conf, 4),
            "session_score":  session_score,
            "macro_bias":     macro_bias,
            "p_loss_ml":      round(p_loss, 4),
            "blocked_by":     None,
        }

# ════════════════════════ ULTRA-FAST WEBSOCKET ════════════════════════
class LiveOrderBook:
    def __init__(self, symbol: str, depth: int = WS_ORDER_BOOK_DEPTH):
        self.symbol = symbol.lower()
        self.depth = depth
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_id = 0
        self._initialized = False
        self._buffered_events: List[dict] = []

    async def initialize(self):
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol.upper()}&limit=1000"
        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: requests.get(url, timeout=5))
            if resp.status_code != 200:
                raise Exception(f"Snapshot HTTP {resp.status_code}")
            data = resp.json()
            self.bids = {float(p): float(q) for p, q in data['bids'][:self.depth]}
            self.asks = {float(p): float(q) for p, q in data['asks'][:self.depth]}
            self.last_update_id = data['lastUpdateId']
            self._initialized = True
            for ev in self._buffered_events:
                if ev.get('u', 0) > self.last_update_id:
                    self._apply_delta(ev)
            self._buffered_events.clear()
            log.info(f"[OB] {self.symbol} initialized")
        except Exception as e:
            log.error(f"[OB] {self.symbol} init failed: {e}")
            raise

    def apply_delta(self, data: dict):
        if not self._initialized:
            if len(self._buffered_events) < 500:
                self._buffered_events.append(data)
            return
        self._apply_delta(data)

    def _apply_delta(self, data: dict):
        U = data.get('U')
        u = data.get('u')
        if U is None or u is None: return
        if u <= self.last_update_id: return
        if U <= self.last_update_id + 1 or self.last_update_id == 0:
            for p, q in data.get('b', []):
                price, qty = float(p), float(q)
                if qty == 0: self.bids.pop(price, None)
                else: self.bids[price] = qty
            for p, q in data.get('a', []):
                price, qty = float(p), float(q)
                if qty == 0: self.asks.pop(price, None)
                else: self.asks[price] = qty
            self.last_update_id = u
            if len(self.bids) > self.depth:
                self.bids = dict(sorted(self.bids.items(), reverse=True)[:self.depth])
            if len(self.asks) > self.depth:
                self.asks = dict(sorted(self.asks.items())[:self.depth])
        else:
            log.warning(f"[OB] {self.symbol} sequence gap, reinitializing...")
            self._initialized = False
            asyncio.create_task(self.initialize())

    def get_imbalance(self, levels: int = 10) -> float:
        bids_sorted = sorted(self.bids.items(), reverse=True)[:levels]
        asks_sorted = sorted(self.asks.items())[:levels]
        bid_vol = sum(q for _, q in bids_sorted)
        ask_vol = sum(q for _, q in asks_sorted)
        total = bid_vol + ask_vol + 1e-9
        return (bid_vol - ask_vol) / total

    def get_spread_bps(self) -> float:
        if not self.bids or not self.asks: return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        mid = (best_bid + best_ask) / 2
        return ((best_ask - best_bid) / mid) * 10000

class LiveOrderFlow:
    def __init__(self, symbol: str, lookback: int = 100):
        self.symbol = symbol.lower()
        self.lookback = lookback
        self._trades = deque(maxlen=lookback)

    def add_trade(self, data: dict):
        try:
            price = float(data['p'])
            qty = float(data['q'])
            is_buy = not data.get('m', False)
            self._trades.append((price, qty, is_buy))
        except Exception: pass

    def get_imbalance(self) -> float:
        if not self._trades: return 0.0
        buy_vol = sum(q for _, q, is_buy in self._trades if is_buy)
        sell_vol = sum(q for _, q, is_buy in self._trades if not is_buy)
        total = buy_vol + sell_vol + 1e-9
        return (buy_vol - sell_vol) / total

    def get_acceleration(self) -> float:
        if len(self._trades) < 20: return 0.0
        trades_list = list(self._trades)
        recent = trades_list[-20:]
        older = trades_list[-40:-20]
        recent_buy = sum(q for _, q, is_buy in recent if is_buy)
        recent_sell = sum(q for _, q, is_buy in recent if not is_buy)
        recent_imb = (recent_buy - recent_sell) / (recent_buy + recent_sell + 1e-9)
        older_buy = sum(q for _, q, is_buy in older if is_buy)
        older_sell = sum(q for _, q, is_buy in older if not is_buy)
        older_imb = (older_buy - older_sell) / (older_buy + older_sell + 1e-9)
        return recent_imb - older_imb

class WebSocketManager:
    def __init__(self, symbols: List[str], max_streams_per_conn: int = 50):
        self.symbols = [s.lower() for s in symbols]
        self.max_streams_per_conn = max_streams_per_conn
        self.order_books: Dict[str, LiveOrderBook] = {}
        self.flow_analyzers: Dict[str, LiveOrderFlow] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        log.info(f"[WS] Initializing {len(self.symbols)} symbols...")
        for sym in self.symbols:
            self.order_books[sym] = LiveOrderBook(sym)
            self.flow_analyzers[sym] = LiveOrderFlow(sym)
        init_tasks = [ob.initialize() for ob in self.order_books.values()]
        await asyncio.gather(*init_tasks)
        n_symbols = len(self.symbols)
        streams_per_conn = min(self.max_streams_per_conn, n_symbols // WS_MAX_CONNECTIONS + 1)
        groups = [self.symbols[i:i+streams_per_conn] for i in range(0, n_symbols, streams_per_conn)]
        self._running = True
        for g in groups:
            task = asyncio.create_task(self._run_group(g))
            self._tasks.append(task)
        log.info(f"[WS] Started {len(groups)} connections (max streams/conn={streams_per_conn})")

    async def stop(self):
        self._running = False
        for t in self._tasks: t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("[WS] Stopped")

    async def _run_group(self, symbols_group: List[str]):
        while self._running:
            try:
                await self._connect_and_listen(symbols_group)
            except Exception as e:
                log.error(f"[WS] Connection group error: {e}")
                if self._running:
                    await asyncio.sleep(WS_RECONNECT_DELAY)

    async def _connect_and_listen(self, symbols: List[str]):
        streams = []
        for sym in symbols:
            streams.append(f"{sym}@depth20@100ms")
            streams.append(f"{sym}@trade")
        stream_path = "/".join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={stream_path}"
        log.info(f"[WS] Connecting group ({len(symbols)} symbols)...")
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            log.info(f"[WS] Group connected")
            async for message in ws:
                if not self._running: break
                try:
                    data = json_fast.loads(message)
                    await self._handle_message(data)
                except Exception as e:
                    log.error(f"[WS] Message error: {e}")

    async def _handle_message(self, data: dict):
        if 'stream' not in data: return
        stream = data['stream']
        parts = stream.split('@')
        if len(parts) < 2: return
        symbol = parts[0]
        event = parts[1]
        payload = data.get('data', {})
        if not isinstance(payload, dict): return
        if event.startswith('depth'):
            ob = self.order_books.get(symbol)
            if ob: ob.apply_delta(payload)
        elif event == 'trade':
            flow = self.flow_analyzers.get(symbol)
            if flow: flow.add_trade(payload)

    async def get_microstructure(self, symbol: str) -> dict:
        sym = symbol.lower()
        ob = self.order_books.get(sym)
        flow = self.flow_analyzers.get(sym)
        return {"ob_imbalance": ob.get_imbalance(10) if ob else 0.0,
                "flow_imbalance": flow.get_imbalance() if flow else 0.0,
                "flow_acceleration": flow.get_acceleration() if flow else 0.0,
                "spread_bps": ob.get_spread_bps() if ob else 0.0}

# ════════════════════════ TAVILY NEWS ════════════════════════
def fetch_tavily_news(symbol: str) -> str:
    if not TAVILY_API_KEY: return ""
    try:
        query = f"{symbol} crypto news"
        resp = requests.post("https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic",
                  "include_images": False, "include_answer": False, "max_results": 3},
            timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        headlines = [r["title"] for r in results if "title" in r]
        return "; ".join(headlines[:3]) if headlines else ""
    except Exception as e:
        log.warning(f"[News] Tavily error: {e}")
        return ""

# ════════════════════════ ROBUST LLM SENTIMENT AGENT ════════════════════════
class LLMSentimentAgent:
    def __init__(self, provider: str = None, cache_ttl: int = None):
        self.provider  = provider or LLM_PROVIDER
        self.cache_ttl = cache_ttl or LLM_CACHE_TTL
        self._cache: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _extract_number(self, raw: str) -> float:
        if not raw: return 0.5
        match = re.search(r"([0-9]*\.?[0-9]+)", raw)
        if match:
            try: return float(match.group(1))
            except ValueError: return 0.5
        return 0.5

    def _call_api(self, prompt: str) -> Optional[str]:
        for attempt in range(LLM_RETRY_MAX_ATTEMPTS):
            try:
                if self.provider == "groq":
                    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                    payload = {
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a trading assistant. Output only a number."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": LLM_MAX_TOKENS,
                    }
                    resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                         headers=headers, json=payload, timeout=10)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
                elif self.provider == "huggingface":
                    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
                    payload = {"inputs": prompt, "parameters": {"max_new_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE, "return_full_text": False}}
                    resp = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                                         headers=headers, json=payload, timeout=20)
                    resp.raise_for_status()
                    return resp.json()[0]["generated_text"].strip()
                elif self.provider == "ollama":
                    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                                         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                                               "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}},
                                         timeout=20)
                    resp.raise_for_status()
                    return resp.json()["response"].strip()
                else:
                    return None
            except Exception as e:
                log.warning(f"[LLM] attempt {attempt+1} failed: {e}")
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))
        return None

    def get_sentiment(self, symbol: str, candles: list, news: str = "") -> float:
        now = time.time()
        with self._lock:
            if symbol in self._cache and now - self._cache[symbol][1] < self.cache_ttl:
                return self._cache[symbol][0]
        last_n = 10
        recent = candles[-last_n:] if len(candles) >= last_n else candles
        price_change = (recent[-1]["close"] - recent[0]["open"]) / recent[0]["open"] * 100 if recent else 0
        prompt = (
            f"Rate sentiment for {symbol} (0=bearish, 0.5=neutral, 1=bullish). "
            f"Data: last {len(recent)} candles, change {price_change:.2f}%. "
            f"News: {news[:200] if news else 'none'}. "
            f"Output ONLY a number."
        )
        raw = self._call_api(prompt)
        log.info(f"[LLM RAW] {symbol} = {raw!r}")
        sentiment = self._extract_number(raw)
        sentiment = max(0.0, min(1.0, sentiment))
        with self._lock:
            self._cache[symbol] = (sentiment, now)
        label = "BULLISH" if sentiment > 0.65 else ("BEARISH" if sentiment < 0.35 else "NEUTRAL")
        log.info(f"[LLM] {symbol} sentiment={sentiment:.2f} → {label}")
        return sentiment

    def predict_reach_target(self, symbol: str, current_price: float,
                            take_price: float, stop_price: float,
                            atr: float, candles: list,
                            direction: str, news: str = "") -> float:
        if not LLM_PREDICT_ENTRY_ENABLED: return 0.5
        prompt = (
            f"SYMBOL: {symbol}. Current price: {current_price:.4f}. "
            f"If a {direction} trade is taken now with take-profit at {take_price:.4f} "
            f"and stop-loss at {stop_price:.4f}, ATR: {atr:.4f}. "
            f"News: {news[:200] if news else 'none'}. "
            f"What is the probability (0-1) that the price will reach the take-profit "
            f"BEFORE hitting the stop-loss within the next 6 hours? Output ONLY a number."
        )
        raw = self._call_api(prompt)
        log.info(f"[LLM RAW ENTRY] {symbol} {direction} = {raw!r}")
        prob = self._extract_number(raw)
        prob = max(0.0, min(1.0, prob))
        log.info(f"[LLM] {symbol} {direction} entry gate probability={prob:.2f}")
        return prob

# ════════════════════════ DATA FEED ════════════════════════
def get_top_volume_pairs(limit=TOP_PAIRS_BY_VOLUME, min_vol=MIN_24H_VOLUME_USDT) -> List[str]:
    try:
        client = Client(API_KEY, API_SECRET, testnet=True)
        tickers = client.futures_ticker()
        pairs = [(t["symbol"], float(t["quoteVolume"])) for t in tickers
                 if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) >= min_vol]
        pairs.sort(key=lambda x: x[1], reverse=True)
        cleaned = [p[0] for p in pairs if re.match(r'^[A-Z0-9]+USDT$', p[0])]
        return cleaned[:limit]
    except Exception as e:
        log.error(f"Pair fetch failed: {e}")
        return [p for p in TRADE_PAIRS if re.match(r'^[A-Z0-9]+USDT$', p)]

class DataFeed:
    LOT_STEPS = {
        "BTCUSDT": 0.001, "ETHUSDT": 0.01, "BNBUSDT": 0.1, "SOLUSDT": 0.1,
        "XRPUSDT": 1.0, "ADAUSDT": 1.0, "DOGEUSDT": 10.0, "LTCUSDT": 0.001,
        "AVAXUSDT": 0.1, "MATICUSDT": 1.0,
    }
    def __init__(self, mode="live"):
        self.mode = mode
        self.simulated = True
        self._sim = {}
        self._hist = {}
        self.client = None
        self.trade_mgr = None
        if mode == "live" and BINANCE_OK and API_KEY:
            try:
                self.client = Client(API_KEY, API_SECRET, testnet=True)
                self.client.futures_base_url = "https://testnet.binancefuture.com/fapi"
                self.simulated = False
                log.info("[Feed] Connected to Binance Testnet")
                self._sync_time()
                self._prefetch_lots()
            except Exception as e:
                log.warning(f"[Feed] Binance init failed: {e} → simulation")

    def _sync_time(self):
        try:
            st = self.client.futures_time()
            self.client.timestamp_offset = st["serverTime"] - int(time.time() * 1000)
        except Exception: pass

    def _prefetch_lots(self):
        try:
            info = self.client.futures_exchange_info()
            for sym in info["symbols"]:
                for f in sym["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        self.LOT_STEPS[sym["symbol"]] = float(f["stepSize"])
        except Exception: pass

    def round_qty(self, symbol: str, qty: float) -> float:
        step = self.LOT_STEPS.get(symbol, 0.01)
        if qty < step: return 0.0
        return round(round(qty / step) * step, 8)

    def get_klines(self, symbol: str, interval=None, limit=None) -> list:
        interval = interval or KLINE_INTERVAL
        limit = limit or KLINE_LIMIT
        if self.mode == "backtest":
            return self._hist.get(f"{symbol}_{interval}", [])
        if self.simulated:
            import random as rand
            price = self._sim.get(symbol, 50000.0)
            out = []
            for _ in range(limit):
                chg = max(-0.03, min(0.03, rand.gauss(0, 0.008)))
                o, c = price, price * (1 + chg)
                h = max(o, c) * (1 + abs(rand.gauss(0, 0.002)))
                l = min(o, c) * (1 - abs(rand.gauss(0, 0.002)))
                out.append({"open": o, "high": h, "low": l, "close": c, "volume": rand.uniform(500, 5000)})
                price = c
            self._sim[symbol] = price
            return out
        try:
            raw = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            return [{"open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                     "close": float(k[4]), "volume": float(k[5])} for k in raw]
        except Exception as e:
            log.error(f"[Feed] Klines error {symbol}: {e}")
            return []

    def balance(self) -> float:
        if self.mode == "live" and not self.simulated:
            try:
                info = self.client.futures_account()
                for asset in info["assets"]:
                    if asset["asset"] == "USDT":
                        return float(asset["availableBalance"])
            except Exception: pass
        if self.trade_mgr:
            return max(0.0, self.trade_mgr.initial_capital + self.trade_mgr.total_pnl)
        return INITIAL_CAPITAL

    def place_order(self, symbol: str, side: str, qty: float, reduce_only=False) -> Optional[dict]:
        if self.simulated:
            price = self._sim.get(symbol, 50000)
            log.info(f"[SIM] {side} {qty:.6f} {symbol} @ {price:.4f}")
            return {"orderId": f"SIM_{int(time.time())}", "price": price}
        self._sync_time()
        try:
            params = {"symbol": symbol, "side": side, "type": "MARKET",
                      "quantity": qty, "newOrderRespType": "RESULT"}
            if reduce_only: params["reduceOnly"] = True
            order = self.client.futures_create_order(**params)
            log.info(f"[ORDER] {order}")
            return order
        except BinanceAPIException as e:
            log.error(f"[Feed] Order error {symbol}: {e}")
            return None

    def get_open_positions(self) -> dict:
        if self.simulated: return {}
        try:
            pos = self.client.futures_position_information()
            return {p["symbol"]: {"qty": float(p["positionAmt"]),
                                  "entry_price": float(p["entryPrice"]),
                                  "unrealized_pnl": float(p["unRealizedProfit"])}
                    for p in pos if abs(float(p["positionAmt"])) > 1e-8}
        except Exception: return {}

    def load_historical(self, symbol: str, interval: str, start: str, end: str) -> list:
        client = Client(API_KEY, API_SECRET, requests_params={"timeout": 30})
        try:
            klines = client.futures_historical_klines(symbol, interval, start, end)
            return [{"timestamp": k[0], "open": float(k[1]), "high": float(k[2]),
                     "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
                    for k in klines]
        except Exception as e:
            log.error(f"[Feed] Historical error {symbol}: {e}")
            return []

# ════════════════════════ TRADE MANAGER ════════════════════════
class TradeManager:
    def __init__(self, initial_capital=INITIAL_CAPITAL, mode="live"):
        self.initial_capital = initial_capital
        self.mode = mode
        self.open_trades = {}
        self.trade_history = []
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.equity_curve = [initial_capital]
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        if mode != "backtest":
            self._load()
        else:
            log.info("[Backtest] Starting with fresh state")

    def _load(self):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                self.open_trades = d.get("open_trades", {})
                self.total_pnl = d.get("total_pnl", 0.0)
                self.wins = d.get("wins", 0)
                self.losses = d.get("losses", 0)
                self.consecutive_losses = d.get("consecutive_losses", 0)
                self.peak_equity = d.get("peak_equity", self.initial_capital)
                self.max_drawdown = d.get("max_drawdown", 0.0)
                self.equity_curve = d.get("equity_curve", [self.initial_capital])
            except Exception as e:
                log.warning(f"[State] Load error: {e}")

    def _save(self):
        try:
            STATE_FILE.write_text(json.dumps({
                "open_trades": self.open_trades,
                "total_pnl": self.total_pnl,
                "wins": self.wins,
                "losses": self.losses,
                "consecutive_losses": self.consecutive_losses,
                "peak_equity": self.peak_equity,
                "max_drawdown": self.max_drawdown,
                "equity_curve": self.equity_curve[-500:],
            }, indent=2))
        except Exception: pass

    def open_trade(self, symbol, direction, entry_price, qty, confidence, atr,
                   entry_features=None, pm_result=None, micro_result=None,
                   llm_signal=None, is_manual=False, atr_multipliers=None):
        self.open_trades[symbol] = {
            "direction": direction,
            "entry_price": entry_price,
            "qty": qty,
            "opened_at": datetime.utcnow().isoformat(),
            "confidence": confidence,
            "atr_at_entry": atr,
            "hold_bars": 0,
            "partial_taken": False,
            "trailing_activated": False,
            "best_price": entry_price,
            "trailing_stop": None,
            "lowest_price": entry_price,
            "highest_price": entry_price,
            "entry_features": entry_features.tolist() if entry_features is not None else None,
            "pm_signal": pm_result.get("signal", "?") if pm_result else "?",
            "micro_signal": micro_result.get("signal", "?") if micro_result else "?",
            "llm_signal": llm_signal,
            "is_manual": is_manual,
            "atr_multipliers": atr_multipliers,
        }
        log.info(f"[OPEN] {symbol} {direction} @ {entry_price:.4f} qty={qty:.5f} conf={confidence:.2%}")
        if self.mode != "backtest":
            Telegram.trade(symbol, direction, entry_price, qty, confidence)
        self._save()

    def close_trade(self, symbol, exit_price, reason, qty=None,
                    veto_layer=None, kelly=None, calibrator=None,
                    llm_tracker=None):
        t = self.open_trades.get(symbol)
        if not t: return None
        qty = min(qty or t["qty"], t["qty"])
        entry = t["entry_price"]
        dir_  = t["direction"]
        gross = (exit_price - entry) * qty if dir_ == "BUY" else (entry - exit_price) * qty
        fee   = (entry + exit_price) * qty * FEES
        slip  = abs(exit_price - entry) * qty * SLIPPAGE
        net   = gross - fee - slip
        pnl_pct = net / (entry * qty + 1e-9) * 100

        self.total_pnl += net
        won = net >= 0
        if won:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1

        if veto_layer and t.get("entry_features") is not None:
            veto_layer.record_outcome(np.array(t["entry_features"]), won)
        if kelly:
            kelly.record(pnl_pct, won)
        if calibrator:
            calibrator.record(t["confidence"], won)
        if llm_tracker and t.get("llm_signal"):
            llm_tracker.record_prediction(t["llm_signal"], won)

        record = {
            "symbol": symbol, "direction": dir_,
            "entry_price": entry, "exit_price": exit_price,
            "qty": qty, "pnl": round(net, 4), "pnl_pct": round(pnl_pct, 3),
            "reason": reason,
            "opened_at": t["opened_at"],
            "closed_at": datetime.utcnow().isoformat(),
            "confidence": t["confidence"],
            "hold_bars": t.get("hold_bars", 0),
            "llm_signal": t.get("llm_signal", ""),
        }
        self.trade_history.append(record)
        eq = self.initial_capital + self.total_pnl
        self.equity_curve.append(eq)
        self.peak_equity = max(self.peak_equity, eq)
        self.max_drawdown = max(self.max_drawdown, (self.peak_equity - eq) / self.peak_equity)
        try:
            with TRADES_FILE.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception: pass
        log.info(f"[CLOSE] {symbol} {dir_} PnL={net:+.4f} USDT ({pnl_pct:+.2f}%) [{reason}]")
        if self.mode != "backtest":
            Telegram.trade(symbol, dir_, entry, qty, t["confidence"], pnl=net, reason=reason)
        t["qty"] -= qty
        if t["qty"] <= 1e-9:
            del self.open_trades[symbol]
        self._save()
        return record

    def increment_hold_bars(self, symbol):
        if symbol in self.open_trades:
            self.open_trades[symbol]["hold_bars"] += 1

    def update_tracking(self, symbol, current):
        if symbol in self.open_trades:
            t = self.open_trades[symbol]
            t["lowest_price"]  = min(t["lowest_price"], current)
            t["highest_price"] = max(t["highest_price"], current)

    def daily_pnl(self) -> float:
        today = date.today().isoformat()
        return sum(t["pnl"] for t in self.trade_history if t.get("closed_at", "")[:10] == today)

    def summary(self) -> dict:
        n  = self.wins + self.losses
        wr = self.wins / n * 100 if n else 0
        eq = self.initial_capital + self.total_pnl
        gw = sum(t["pnl"] for t in self.trade_history if t["pnl"] > 0)
        gl = abs(sum(t["pnl"] for t in self.trade_history if t["pnl"] < 0))
        return {
            "open": len(self.open_trades),
            "total_trades": n,
            "win_rate": round(wr, 2),
            "total_pnl": round(self.total_pnl, 4),
            "total_return_pct": round((eq - self.initial_capital) / self.initial_capital * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "profit_factor": round(gw / gl, 3) if gl else float("inf"),
            "consecutive_losses": self.consecutive_losses,
            "equity": round(eq, 2),
        }

    def compute_sharpe(self, risk_free_rate: float = 0.0, periods_per_year: int = 252 * 24) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        eq_series = pd.Series(self.equity_curve)
        returns = eq_series.pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        excess_ret = returns.mean() - (risk_free_rate / periods_per_year)
        std_ret = returns.std()
        if std_ret == 0 or np.isnan(std_ret):
            return 0.0
        sharpe_period = excess_ret / std_ret
        sharpe_annual = sharpe_period * np.sqrt(periods_per_year)
        return float(sharpe_annual)

# ════════════════════════ EXIT MANAGER ════════════════════════
class ExitManager:
    @staticmethod
    def check(trade: dict, current: float, current_atr: float, is_manual: bool = False) -> dict:
        entry = trade["entry_price"]
        direction = trade["direction"]
        atr_entry = trade.get("atr_at_entry", current_atr)

        mults = trade.get("atr_multipliers", {})
        stop_mult = mults.get("stop", ATR_STOP_MULT)
        take_mult = mults.get("take", ATR_TAKE_MULT)
        part_mult = mults.get("partial", ATR_PARTIAL_MULT)
        trail_mult = mults.get("trailing", TRAILING_ATR_MULT)
        hard_mult = mults.get("hard", HARD_STOP_ATR_MULT)

        pnl_pct = ((current - entry) / entry * 100 if direction == "BUY"
                   else (entry - current) / entry * 100)

        if is_manual:
            opened_at = datetime.fromisoformat(trade["opened_at"])
            if pnl_pct > 0.5:
                return {"action": "CLOSE", "reason": "profit_target_manual", "pnl_pct": pnl_pct}
            if (datetime.utcnow() - opened_at).days >= MAX_HOLD_DAYS_MANUAL:
                return {"action": "CLOSE", "reason": "timeout_manual", "pnl_pct": pnl_pct}
            return {"action": "HOLD", "reason": "await_profit", "pnl_pct": pnl_pct}

        stop_dist = stop_mult * atr_entry
        take_dist = take_mult * atr_entry
        part_dist = part_mult * atr_entry
        hard_dist = hard_mult * atr_entry

        if direction == "BUY":
            if current <= entry - hard_dist:
                return {"action": "CLOSE", "reason": "hard_stop", "pnl_pct": pnl_pct}
        else:
            if current >= entry + hard_dist:
                return {"action": "CLOSE", "reason": "hard_stop", "pnl_pct": pnl_pct}

        if direction == "BUY":
            if current <= entry - stop_dist:
                return {"action": "CLOSE", "reason": "atr_stop", "pnl_pct": pnl_pct}
            if current >= entry + take_dist:
                return {"action": "CLOSE", "reason": "atr_take", "pnl_pct": pnl_pct}
            if not trade.get("partial_taken", False) and current >= entry + part_dist:
                return {"action": "PARTIAL", "reason": "partial", "pnl_pct": pnl_pct}
        else:
            if current >= entry + stop_dist:
                return {"action": "CLOSE", "reason": "atr_stop", "pnl_pct": pnl_pct}
            if current <= entry - take_dist:
                return {"action": "CLOSE", "reason": "atr_take", "pnl_pct": pnl_pct}
            if not trade.get("partial_taken", False) and current <= entry - part_dist:
                return {"action": "PARTIAL", "reason": "partial", "pnl_pct": pnl_pct}

        trail_dist = trail_mult * atr_entry
        if direction == "BUY":
            if pnl_pct >= part_mult * (atr_entry / entry * 100):
                if not trade.get("trailing_activated", False):
                    trade["trailing_activated"] = True
                    trade["best_price"] = current
                    trade["trailing_stop"] = current - trail_dist
                elif current > trade.get("best_price", current):
                    trade["best_price"] = current
                    trade["trailing_stop"] = current - trail_dist
                if trade.get("trailing_stop") and current <= trade["trailing_stop"]:
                    return {"action": "CLOSE", "reason": "trailing", "pnl_pct": pnl_pct}
        else:
            if pnl_pct >= part_mult * (atr_entry / entry * 100):
                if not trade.get("trailing_activated", False):
                    trade["trailing_activated"] = True
                    trade["best_price"] = current
                    trade["trailing_stop"] = current + trail_dist
                elif current < trade.get("best_price", current):
                    trade["best_price"] = current
                    trade["trailing_stop"] = current + trail_dist
                if trade.get("trailing_stop") and current >= trade["trailing_stop"]:
                    return {"action": "CLOSE", "reason": "trailing", "pnl_pct": pnl_pct}

        opened_at = trade.get("opened_at", datetime.utcnow().isoformat())
        try:
            elapsed_s = (datetime.utcnow() - datetime.fromisoformat(opened_at)).total_seconds()
        except Exception:
            elapsed_s = trade.get("hold_bars", 0) * 300

        if elapsed_s >= MAX_HOLD_SECONDS:
            return {"action": "CLOSE", "reason": "timeout_max", "pnl_pct": pnl_pct}

        return {"action": "HOLD", "reason": "none", "pnl_pct": pnl_pct}

# ════════════════════════ LLM OVERRIDE TRACKER ════════════════════════
class LLMOverrideTracker:
    def __init__(self):
        self._predictions = deque(maxlen=LLM_WARMUP_TRADES*2)
        self._lock = threading.Lock()

    def record_prediction(self, prediction_signal: str, actual_won: bool):
        with self._lock:
            self._predictions.append((prediction_signal, actual_won))

    def can_override(self) -> bool:
        with self._lock:
            if len(self._predictions) < LLM_WARMUP_TRADES:
                return False
            trades = [(s, w) for s, w in self._predictions if s in ("BUY", "SELL")]
            if len(trades) < LLM_WARMUP_TRADES:
                return False
            wins = sum(1 for _, w in trades if w)
            wr = wins / len(trades)
            return wr >= LLM_OVERRIDE_WIN_RATE

# ════════════════════════ DASHBOARD ════════════════════════
def start_dashboard(bot):
    if not FLASK_OK or not ENABLE_DASHBOARD: return
    app = Flask("CryptoBot")

    @app.route("/")
    def idx():
        return jsonify({"routes": ["/status", "/positions", "/trades", "/calibration", "/kelly", "/veto", "/llm"]})

    @app.route("/status")
    def status():
        s = bot.trade_mgr.summary()
        return jsonify({**s, "daily_pnl": round(bot.trade_mgr.daily_pnl(), 4),
                        "circuit": bot.circuit_broken, "cycle": bot.cycle})

    @app.route("/positions")
    def positions(): return jsonify(bot.trade_mgr.open_trades)

    @app.route("/trades")
    def trades(): return jsonify(bot.trade_mgr.trade_history[-100:])

    @app.route("/calibration")
    def calib(): return jsonify(bot.engine.calibrator.summary())

    @app.route("/kelly")
    def kelly():
        return jsonify({sym: round(bot.kelly_sizers[sym].compute_fraction(), 4)
                        for sym in list(bot.kelly_sizers.keys())[:10]})

    @app.route("/veto")
    def veto():
        return jsonify({"trained": bot.veto_layer._trained, "samples": len(bot.veto_layer._X_buf)})

    @app.route("/llm")
    def llm():
        return jsonify({"enabled": LLM_ENABLED, "provider": LLM_PROVIDER,
                        "override_win_rate": LLM_OVERRIDE_WIN_RATE,
                        "can_override": bot.llm_tracker.can_override()})

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False, use_reloader=False),
        daemon=True
    ).start()
    log.info(f"[Dashboard] http://localhost:{DASHBOARD_PORT}")

# ════════════════════════ ADAPTIVE RISK ENGINE ════════════════════════
class AdaptiveRiskEngine:
    def __init__(self, atr_lookback=RISK_ATR_LOOKBACK):
        self.atr_lookback = atr_lookback

    def get_profile(self, ind: dict, candles: list, micro_data: dict,
                    session_score: float, regime: str) -> dict:
        if len(candles) >= self.atr_lookback:
            atr_values = []
            for i in range(self.atr_lookback, len(candles)+1):
                atr_values.append(_atr(candles[max(0,i-15):i], 14))
            atr_pct70 = np.percentile(atr_values, 70) if atr_values else ind["atr14"]
            vol_ratio = ind["atr14"] / (atr_pct70 + 1e-9)
        else:
            vol_ratio = 1.0

        ob_imb = micro_data.get("ob_imbalance", 0.0)
        chaotic = abs(ob_imb) > RISK_IMBALANCE_THRESHOLD

        body_pct = ind["body_pct"]
        indecision = body_pct < RISK_BODY_RATIO_THRESHOLD

        adx = ind["adx14"]
        strong_trend = adx >= RISK_ADX_STRONG

        if chaotic:
            profile_name = "chaotic"
        elif strong_trend:
            profile_name = "trending_strong"
        elif vol_ratio < 0.7:
            profile_name = "low_vol"
        elif vol_ratio > 1.3:
            profile_name = "high_vol"
        else:
            profile_name = "normal"

        if regime == "RANGING" and indecision:
            profile_name = "low_vol"

        profile = DYNAMIC_ATR_PROFILES.get(profile_name, DYNAMIC_ATR_PROFILES["normal"])
        log.info(f"[AdaptiveRisk] vol={vol_ratio:.2f} ADX={adx:.1f} ob_imb={ob_imb:.3f} "
                 f"body_pct={body_pct:.2f} regime={regime} → {profile_name} "
                 f"stop={profile['stop']} take={profile['take']}")
        return profile

# ════════════════════════ MAIN BOT ════════════════════════
class CryptoBot:
    def __init__(self, mode="live"):
        global TRADE_PAIRS
        if not TRADE_PAIRS:
            TRADE_PAIRS = get_top_volume_pairs()
        TRADE_PAIRS = [p for p in TRADE_PAIRS if re.match(r'^[A-Z0-9]+USDT$', p)]
        log.info(f"Trading pairs after cleanup: {len(TRADE_PAIRS)} pairs")

        self.mode = mode
        self.cycle = 0
        self.circuit_broken = False

        self.feed = DataFeed(mode=mode)
        self.trade_mgr = TradeManager(mode=mode)
        self.feed.trade_mgr = self.trade_mgr
        self._lock = threading.Lock()

        self.pm_layer = PriceMomentumLayer()
        self.micro_layer = MicrostructureLayer()
        self.ctx_layer = ContextLayer()
        self.veto_layer = MLVetoLayer()
        self.engine = LayeredDecisionEngine()
        self.exit_mgr = ExitManager()
        self.llm_tracker = LLMOverrideTracker()
        self.llm = LLMSentimentAgent() if LLM_ENABLED else None
        self.risk_engine = AdaptiveRiskEngine()

        # ADM Engine
        self.adm_engine = ADMEngine()

        self.lstm_agents: Dict[str, LSTMAgent] = {}
        self.kelly_sizers: Dict[str, KellyPositionSizer] = {}
        self._recently_closed: Dict[str, float] = {}

        for sym in TRADE_PAIRS:
            self.lstm_agents[sym] = LSTMAgent(sym)
            self.kelly_sizers[sym] = KellyPositionSizer()

        # Validate pairs (ensure they exist on exchange)
        if mode != "backtest" and not self.feed.simulated:
            valid_pairs = []
            for p in TRADE_PAIRS:
                try:
                    _ = self.feed.client.futures_klines(symbol=p, interval="1m", limit=1)
                    valid_pairs.append(p)
                except Exception:
                    continue
            TRADE_PAIRS = valid_pairs
            log.info(f"Validated {len(TRADE_PAIRS)} pairs with active data")

        self.ws_manager = None
        self._executor = ThreadPoolExecutor(max_workers=4)

        if mode != "backtest" and not self.feed.simulated:
            self._sync_initial_positions()
        elif mode == "backtest":
            interval_seconds = {"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400}
            secs_per_bar = interval_seconds.get(BACKTEST_INTERVAL, 3600)
            self.TRADE_COOLDOWN_BARS = max(1, int(TRADE_COOLDOWN_SECONDS / secs_per_bar))

    def _get_kelly(self, symbol: str) -> KellyPositionSizer:
        if symbol not in self.kelly_sizers:
            self.kelly_sizers[symbol] = KellyPositionSizer()
        return self.kelly_sizers[symbol]

    def _sync_initial_positions(self):
        pos = self.feed.get_open_positions()
        for sym, p in pos.items():
            if sym not in self.trade_mgr.open_trades:
                candles = self.feed.get_klines(sym)
                atr = _atr(candles, 14) if candles else 0
                self.trade_mgr.open_trade(
                    sym, "BUY" if p["qty"] > 0 else "SELL",
                    p["entry_price"], abs(p["qty"]),
                    MANUAL_TRADE_CONFIDENCE, atr,
                    is_manual=True
                )

    def _check_circuits(self) -> bool:
        if self.mode == "backtest": return False
        if self.circuit_broken: return True
        daily = self.trade_mgr.daily_pnl()
        if daily < -(self.trade_mgr.initial_capital * DAILY_LOSS_LIMIT_PCT / 100):
            self.circuit_broken = True
            Telegram.alert(f"Circuit breaker: daily PnL={daily:.2f}")
            return True
        if self.trade_mgr.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.circuit_broken = True
            return True
        return False

    def analyze_pair(self, symbol: str, injected=None) -> Optional[dict]:
        candles = injected or self.feed.get_klines(symbol)
        if len(candles) < 60: return None

        ind = _compute_all(candles)
        pm_result = self.pm_layer.score(ind, candles)
        micro_result = self.micro_layer.analyze(symbol, ind, candles)
        ctx_result = self.ctx_layer.analyze(symbol, ind, candles)

        veto_feats = self.veto_layer.extract_features(
            pm_result, micro_result, ctx_result, ind,
            self.trade_mgr.consecutive_losses)
        veto_result = self.veto_layer.should_veto(veto_feats)

        decision = self.engine.decide(pm_result, micro_result, ctx_result, veto_result)

        lstm_prob = self.lstm_agents.get(symbol, LSTMAgent(symbol)).predict(candles)

        llm_sentiment = 0.5
        llm_signal = None
        if self.llm:
            news = fetch_tavily_news(symbol) if self.mode != "backtest" else ""
            llm_sentiment = self.llm.get_sentiment(symbol, candles, news)
            if llm_sentiment > 0.65:
                llm_signal = "BUY"
            elif llm_sentiment < 0.35:
                llm_signal = "SELL"
            else:
                llm_signal = "HOLD"

            if decision["signal"] == "HOLD":
                if llm_sentiment > 0.82 and self.llm_tracker.can_override():
                    decision["signal"] = "BUY"
                    decision["confidence"] = llm_sentiment
                    log.info(f"[LLM] Override {symbol}: HOLD→BUY @ conf={llm_sentiment:.2f}")
                elif llm_sentiment < 0.18 and self.llm_tracker.can_override():
                    decision["signal"] = "SELL"
                    decision["confidence"] = 1.0 - llm_sentiment
                    log.info(f"[LLM] Override {symbol}: HOLD→SELL @ conf={1.0-llm_sentiment:.2f}")

        if decision["signal"] != "HOLD" and lstm_prob < 0.4:
            decision["confidence"] *= 0.9

        # ADM probability and multiplier
        adm_mult = self.adm_engine.get_confidence_multiplier(ind)
        if self.adm_engine._enabled and decision["signal"] != "HOLD":
            decision["confidence"] *= adm_mult
            log.info(f"[ADM] {symbol} adm_prob={self.adm_engine.predict_probability(ind):.3f} "
                     f"multiplier={adm_mult:.2f} → final_conf={decision['confidence']:.4f}")

        return {
            "symbol": symbol, "price": ind["price"], "atr": ind["atr14"],
            "decision": decision, "pm_result": pm_result,
            "micro_result": micro_result, "ctx_result": ctx_result,
            "veto_feats": veto_feats, "ind": ind, "candles": candles,
            "lstm_prob": lstm_prob, "llm_signal": llm_signal,
        }

    async def run_live(self):
        log.info(f"LIVE MODE – {len(TRADE_PAIRS)} pairs")
        if WS_OK:
            self.ws_manager = WebSocketManager(TRADE_PAIRS)
            await self.ws_manager.start()
        else:
            log.warning("WebSocket not available")
        asyncio.create_task(self._periodic_status())
        while True:
            try:
                if self._check_circuits():
                    await asyncio.sleep(30)
                    continue
                await self._monitor_async()
                free = [s for s in TRADE_PAIRS if s not in self.trade_mgr.open_trades]
                if free:
                    random.shuffle(free)
                    for sym in free[:LLM_MAX_PAIRS_PER_CYCLE]:
                        res = await self._analyze_pair_async(sym)
                        if res and res["decision"]["signal"] != "HOLD":
                            await self._execute_async(res)
                await asyncio.sleep(LIVE_LOOP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[Main] Error: {e}")
                await asyncio.sleep(5)

    async def _analyze_pair_async(self, symbol: str) -> Optional[dict]:
        candles = await asyncio.get_event_loop().run_in_executor(
            self._executor, self.feed.get_klines, symbol)
        if len(candles) < 60: return None
        return self.analyze_pair(symbol, injected=candles)

    async def _get_micro_data(self, symbol: str) -> dict:
        if self.ws_manager:
            try:
                return await self.ws_manager.get_microstructure(symbol)
            except Exception:
                pass
        return {}

    async def _execute_async(self, analysis: dict):
        decision = analysis["decision"]
        symbol = analysis["symbol"]
        price = analysis["price"]
        signal = decision["signal"]
        conf = decision["confidence"]

        with self._lock:
            if len(self.trade_mgr.open_trades) >= MAX_OPEN_TRADES:
                return
            equity = self.feed.balance()
            kelly = self._get_kelly(symbol)
            atr = analysis["atr"]

            micro_data = await self._get_micro_data(symbol)

            risk_profile = self.risk_engine.get_profile(
                ind=analysis["ind"],
                candles=analysis["candles"],
                micro_data=micro_data,
                session_score=analysis["ctx_result"]["session_score"],
                regime=analysis["ctx_result"]["regime"]
            )

            if self.llm and LLM_PREDICT_ENTRY_ENABLED:
                stop_price = price - (risk_profile["stop"] * atr) if signal == "BUY" else price + (risk_profile["stop"] * atr)
                take_price = price + (risk_profile["take"] * atr) if signal == "BUY" else price - (risk_profile["take"] * atr)
                news = fetch_tavily_news(symbol) if self.mode != "backtest" else ""
                prob = self.llm.predict_reach_target(
                    symbol, price, take_price, stop_price, atr,
                    analysis["candles"], signal, news)
                if prob < LLM_PREDICT_ENTRY_CONFIDENCE:
                    log.info(f"[LLM] {symbol} {signal} rejected – prob {prob:.2f} < {LLM_PREDICT_ENTRY_CONFIDENCE}")
                    return
                conf = min(conf, prob)

            size = kelly.position_size(equity, price, atr, conf)
            qty = self.feed.round_qty(symbol, size / price)
            if qty <= 0: return
            order = await asyncio.get_event_loop().run_in_executor(
                self._executor, self.feed.place_order, symbol,
                "BUY" if signal == "BUY" else "SELL", qty)
            if order:
                actual_entry_price = float(order.get("avgPrice", price))
                self.trade_mgr.open_trade(
                    symbol, signal, actual_entry_price, qty, conf, atr,
                    entry_features=analysis["veto_feats"],
                    pm_result=analysis["pm_result"],
                    micro_result=analysis["micro_result"],
                    llm_signal=analysis.get("llm_signal"),
                    atr_multipliers=risk_profile
                )

    async def _monitor_async(self):
        for symbol in list(self.trade_mgr.open_trades.keys()):
            try:
                trade = self.trade_mgr.open_trades.get(symbol)
                if not trade: continue
                candles = await asyncio.get_event_loop().run_in_executor(
                    self._executor, self.feed.get_klines, symbol, "5m", 60)
                if not candles: continue
                current = candles[-1]["close"]
                ind = await asyncio.get_event_loop().run_in_executor(
                    self._executor, _compute_all, candles)
                is_manual = trade.get("is_manual", False)
                check = ExitManager.check(trade, current, ind["atr14"], is_manual)
                close_side = "SELL" if trade["direction"] == "BUY" else "BUY"

                dir_mult = 1 if trade["direction"] == "BUY" else -1
                unreal_pnl = (current - trade["entry_price"]) * trade["qty"] * dir_mult
                if unreal_pnl >= 1.0 and check["action"] != "CLOSE":
                    check = {"action": "CLOSE", "reason": f"custom_$1_profit ({unreal_pnl:.2f})", "pnl_pct": 0}

                if check["action"] == "CLOSE":
                    exchange_pos = self.feed.get_open_positions()
                    if exchange_pos is None:
                        exchange_pos = {}
                    if symbol in exchange_pos:
                        exchange_qty = abs(exchange_pos[symbol]["qty"])
                        if exchange_qty <= 0:
                            log.warning(f"[MONITOR] No exchange position for {symbol} – removing")
                            del self.trade_mgr.open_trades[symbol]
                            continue
                        qty_r = self.feed.round_qty(symbol, min(trade["qty"], exchange_qty))
                    else:
                        log.warning(f"[MONITOR] {symbol} not on exchange – removing")
                        del self.trade_mgr.open_trades[symbol]
                        continue
                    if qty_r > 0:
                        order = await asyncio.get_event_loop().run_in_executor(
                            self._executor, self.feed.place_order, symbol, close_side, qty_r, True)
                        if order:
                            actual_exit_price = float(order.get("avgPrice", current))
                            self.trade_mgr.close_trade(symbol, actual_exit_price, check["reason"],
                                                       veto_layer=self.veto_layer,
                                                       kelly=self._get_kelly(symbol),
                                                       calibrator=self.engine.calibrator,
                                                       llm_tracker=self.llm_tracker)
                elif check["action"] == "PARTIAL":
                    exchange_pos = self.feed.get_open_positions()
                    if exchange_pos is None:
                        exchange_pos = {}
                    if symbol in exchange_pos:
                        exchange_qty = abs(exchange_pos[symbol]["qty"])
                        half = min(trade["qty"] / 2, exchange_qty)
                    else:
                        half = 0
                    qty_r = self.feed.round_qty(symbol, half)
                    if qty_r > 0:
                        order = await asyncio.get_event_loop().run_in_executor(
                            self._executor, self.feed.place_order, symbol, close_side, qty_r, True)
                        if order:
                            actual_exit_price = float(order.get("avgPrice", current))
                            trade["partial_taken"] = True
                            self.trade_mgr.close_trade(symbol, actual_exit_price, "partial", qty=qty_r)
            except Exception as e:
                log.error(f"[MONITOR] Skipping {symbol} due to error: {e}")
                continue

    async def _periodic_status(self):
        while True:
            await asyncio.sleep(60)
            self.cycle += 1
            if self.cycle % 20 == 0:
                s = self.trade_mgr.summary()
                log.info(f"Cycle {self.cycle} | Trades={s['total_trades']} WR={s['win_rate']:.1f}% "
                         f"PnL={s['total_pnl']:+.2f} Equity={s['equity']:.2f}")

    # ─── Backtest ─────────────────────────────────────────────────
    def run_backtest(self, start=BACKTEST_START, end=BACKTEST_END,
                     interval=BACKTEST_INTERVAL) -> dict:
        global TRADE_PAIRS
        TRADE_PAIRS = [p for p in TRADE_PAIRS if re.match(r'^[A-Z0-9]+USDT$', p)]
        log.info(f"═══ BACKTEST START ═══")
        log.info(f"Period: {start} → {end} ({interval})")
        valid = []
        for sym in TRADE_PAIRS:
            c = self.feed.load_historical(sym, interval, start, end)
            if len(c) < 200:
                log.warning(f"Skip {sym}: {len(c)} candles")
                continue
            self.feed._hist[f"{sym}_{interval}"] = c
            valid.append(sym)
            log.info(f"  ✓ {sym}: {len(c)} candles loaded")
        if not valid:
            log.error("No valid pairs. Exiting.")
            return {}
        orig = TRADE_PAIRS[:]
        TRADE_PAIRS = valid
        n = len(self.feed._hist[f"{valid[0]}_{interval}"])
        warmup = 60

        for idx in range(warmup, n - 1):
            for sym in list(self.trade_mgr.open_trades.keys()):
                hist = self.feed._hist.get(f"{sym}_{interval}", [])
                if idx >= len(hist): continue
                bar = hist[idx]
                trade = self.trade_mgr.open_trades.get(sym)
                if not trade: continue
                entry = trade["entry_price"]
                dir_ = trade["direction"]
                atr_entry = trade.get("atr_at_entry", _atr(hist[max(0,idx-14):idx+1], 14) if idx>14 else 0)
                is_manual = trade.get("is_manual", False)
                check = ExitManager.check(trade, bar["close"], atr_entry, is_manual)
                close_side = "SELL" if dir_ == "BUY" else "BUY"
                dir_mult = 1 if dir_ == "BUY" else -1
                unreal = (bar["close"] - entry) * trade["qty"] * dir_mult
                if unreal >= 1.0 and check["action"] != "CLOSE":
                    check = {"action": "CLOSE", "reason": f"$1_profit ({unreal:.2f})", "pnl_pct": 0}
                if check["action"] == "CLOSE":
                    qty_r = self.feed.round_qty(sym, trade["qty"])
                    if qty_r > 0:
                        self.trade_mgr.close_trade(sym, bar["close"], check["reason"],
                                                   veto_layer=self.veto_layer,
                                                   kelly=self._get_kelly(sym),
                                                   calibrator=self.engine.calibrator,
                                                   llm_tracker=self.llm_tracker)
                        self._recently_closed[sym] = float(idx)
                elif check["action"] == "PARTIAL":
                    half = self.feed.round_qty(sym, trade["qty"] / 2)
                    if half > 0:
                        self.trade_mgr.close_trade(sym, bar["close"], "partial", qty=half,
                                                   veto_layer=self.veto_layer,
                                                   kelly=self._get_kelly(sym),
                                                   calibrator=self.engine.calibrator,
                                                   llm_tracker=self.llm_tracker)
                        trade["partial_taken"] = True
                else:
                    if trade["hold_bars"] >= MAX_HOLD_BARS:
                        self.trade_mgr.close_trade(sym, bar["close"], "timeout",
                                                   veto_layer=self.veto_layer,
                                                   kelly=self._get_kelly(sym),
                                                   calibrator=self.engine.calibrator,
                                                   llm_tracker=self.llm_tracker)
                        self._recently_closed[sym] = float(idx)
                    else:
                        self.trade_mgr.update_tracking(sym, bar["close"])
                        self.trade_mgr.increment_hold_bars(sym)

            for sym in TRADE_PAIRS:
                hist = self.feed._hist.get(f"{sym}_{interval}", [])
                if idx + 1 >= len(hist): continue
                window = hist[max(0, idx - 200):idx + 1]
                if len(window) < 60: continue
                self.micro_layer.update_vwap(sym, window[-1])
                if sym not in self.lstm_agents:
                    self.lstm_agents[sym] = LSTMAgent(sym)
                lstm_agent = self.lstm_agents[sym]
                if len(window) >= LSTM_SEQ_LEN + 2:
                    lstm_agent.update(window[:-1], window[-1]["close"])
                    if lstm_agent._n_upd % LSTM_TRAIN_EVERY == 0:
                        lstm_agent.train()
                res = self.analyze_pair(sym, injected=window)
                if res and res["decision"]["signal"] != "HOLD":
                    last_close = self._recently_closed.get(sym)
                    if last_close is not None and (idx - last_close) < self.TRADE_COOLDOWN_BARS:
                        continue
                    next_open = hist[idx + 1]["open"]
                    res["price"] = next_open
                    self.execute_decision(res, bar_idx=idx+1)

            if idx % 1000 == 0:
                closed = self.trade_mgr.wins + self.trade_mgr.losses
                wr = (self.trade_mgr.wins / closed * 100) if closed else 0
                log.info(f"  Progress: {idx}/{n} | Trades={closed} | WR={wr:.1f}%")

        s = self.trade_mgr.summary()
        sharpe = self.trade_mgr.compute_sharpe()
        log.info("═" * 70)
        log.info("BACKTEST COMPLETE")
        log.info(f"  Total Trades:     {s['total_trades']}")
        log.info(f"  Win Rate:         {s['win_rate']:.2f}%")
        log.info(f"  Total PnL:        {s['total_pnl']:+.4f} USDT")
        log.info(f"  Return:           {s['total_return_pct']:+.2f}%")
        log.info(f"  Profit Factor:    {s['profit_factor']:.3f}")
        log.info(f"  Max Drawdown:     {s['max_drawdown_pct']:.2f}%")
        log.info(f"  Sharpe Ratio:     {sharpe:.3f}")
        log.info("═" * 70)
        TRADE_PAIRS = orig
        return {**s, "sharpe": round(sharpe, 3)}

    def execute_decision(self, analysis: dict, bar_idx: Optional[int] = None):
        decision = analysis["decision"]
        symbol = analysis["symbol"]
        price = analysis["price"]
        signal = decision["signal"]
        conf = decision["confidence"]

        if signal == "HOLD" or conf <= 0: return
        if self._check_circuits(): return
        with self._lock:
            if len(self.trade_mgr.open_trades) >= MAX_OPEN_TRADES: return
            if symbol in self.trade_mgr.open_trades: return
            if self.mode == "backtest" and bar_idx is not None:
                last_close = self._recently_closed.get(symbol)
                if last_close is not None and (bar_idx - last_close) < self.TRADE_COOLDOWN_BARS:
                    return
            else:
                now = time.time()
                if symbol in self._recently_closed and now - self._recently_closed[symbol] < TRADE_COOLDOWN_SECONDS:
                    return

            atr = analysis["atr"]
            risk_profile = self.risk_engine.get_profile(
                ind=analysis["ind"],
                candles=analysis["candles"],
                micro_data={},
                session_score=analysis["ctx_result"]["session_score"],
                regime=analysis["ctx_result"]["regime"]
            )

            if self.llm and LLM_PREDICT_ENTRY_ENABLED:
                stop_price = price - (risk_profile["stop"] * atr) if signal == "BUY" else price + (risk_profile["stop"] * atr)
                take_price = price + (risk_profile["take"] * atr) if signal == "BUY" else price - (risk_profile["take"] * atr)
                news = fetch_tavily_news(symbol) if self.mode != "backtest" else ""
                prob = self.llm.predict_reach_target(
                    symbol, price, take_price, stop_price, atr,
                    analysis["candles"], signal, news)
                if prob < LLM_PREDICT_ENTRY_CONFIDENCE:
                    log.info(f"[LLM] {symbol} {signal} rejected – prob {prob:.2f} < {LLM_PREDICT_ENTRY_CONFIDENCE}")
                    return
                conf = min(conf, prob)

            equity = self.feed.balance()
            kelly = self._get_kelly(symbol)
            size = kelly.position_size(equity, price, atr, conf)
            qty = self.feed.round_qty(symbol, size / price)
            if qty <= 0: return
            self.trade_mgr.open_trade(
                symbol, signal, price, qty, conf, analysis["atr"],
                entry_features=analysis["veto_feats"],
                pm_result=analysis["pm_result"],
                micro_result=analysis["micro_result"],
                llm_signal=analysis.get("llm_signal"),
                atr_multipliers=risk_profile
            )

# ════════════════════════ ENTRY POINT ════════════════════════
async def main():
    MODE = "backtest"
    bot = CryptoBot(mode=MODE)
    start_dashboard(bot)
    if MODE == "live":
        await bot.run_live()
    else:
        bot.run_backtest(BACKTEST_START, BACKTEST_END, BACKTEST_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())