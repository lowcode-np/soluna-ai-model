# -*- coding: utf-8 -*-
# Soluna AI: Signal Server with Training Config Support and Emoji Logging

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import json
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import load_model

from flask import Flask, jsonify, request
from werkzeug.serving import make_server
import logging

# --- Splash Screen Class ---
class SplashScreen:
    def __init__(self, root, duration_seconds=5):
        self.root = root
        self.splash = tk.Toplevel(root)
        self.splash.overrideredirect(True)
        self.splash.configure(bg="#0f0f1e")

        self.duration_ms = duration_seconds * 1000 
        self.update_interval_ms = 50
        self.total_steps = self.duration_ms // self.update_interval_ms
        self.progress_increment = 100 / self.total_steps
        self.current_step = 0

        width, height = 600, 350
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.splash.geometry(f'{width}x{height}+{x}+{y}')
        
        main_frame = tk.Frame(self.splash, bg="#1a1a2e", highlightbackground="#00ff88", highlightthickness=1)
        main_frame.pack(fill='both', expand=True, padx=1, pady=1)
        
        tk.Label(main_frame, text="SOLUNA AI", bg="#1a1a2e", fg="#00ff88", 
                 font=("Segoe UI", 48, "bold")).pack(pady=(60, 0))
        tk.Label(main_frame, text="Real-time AI Trading Signal Platform", bg="#1a1a2e", fg="#CCCCCC", 
                 font=("Segoe UI", 12)).pack(pady=(0, 40))

        self.status_label = tk.Label(main_frame, text="Initializing...", bg="#1a1a2e", fg="#888888",
                                     font=("Segoe UI", 10))
        self.status_label.pack(pady=(20, 5))

        s = ttk.Style()
        s.theme_use('clam')
        s.configure("green.Horizontal.TProgressbar", foreground='#00ff88', background='#00ff88', troughcolor='#2a2a3e', bordercolor="#1a1a2e", lightcolor="#1a1a2e", darkcolor="#1a1a2e")
        self.progress = ttk.Progressbar(main_frame, style="green.Horizontal.TProgressbar", orient="horizontal", 
                                        length=400, mode='determinate')
        self.progress.pack(pady=(0, 20))

    def _animate(self):
        if self.current_step <= self.total_steps:
            # อัปเดต Progress Bar
            self.progress['value'] = self.current_step * self.progress_increment
            
            # อัปเดตข้อความตามความคืบหน้า
            progress_percent = (self.current_step / self.total_steps) * 100
            if progress_percent < 40:
                self.status_label.config(text="Initializing components...")
            elif progress_percent < 80:
                self.status_label.config(text="Loading models...")
            else:
                self.status_label.config(text="Finalizing...")
            
            self.current_step += 1
            self.splash.after(self.update_interval_ms, self._animate)
        else:
            self.close()

    def close(self):
        self.splash.destroy()
        self.root.deiconify()

    def start(self):
        self.splash.after(0, self._animate)

# --- Configuration ---
class ServerConfig:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_PATH, '.models')
    DEFAULT_HOST = '127.0.0.1'
    DEFAULT_PORT = 5000
    NUM_CLASSES = 3

# --- Global State ---
app_state = {
    "latest_signal": {"status": "Waiting for data...", "signal": "NEUTRAL"},
    "is_server_running": False,
    "server_instance": None,
    "models": {},
    "scaler": None,
    "training_config": None,
    "model_paths": {
        "xgb": "",
        "lr": "",
        "lstm": "",
        "scaler": "",
        "config": ""
    }
}

global_log_text = None

# --- Emoji Mapping ---
EMOJI_MAP = {
    "INFO": "ℹ️",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "SIGNAL": "📡"
}

# --- Logging ---
def log_terminal(message, status="INFO", header=False):
    timestamp = datetime.now().strftime("%H:%M:%S")
    output = ""
    
    if header:
        separator = "=" * 60
        output = f"\n{separator}\n{message.center(60)}\n{separator}\n"
    else:
        emoji = EMOJI_MAP.get(status, "🔹") 
        output = f"{emoji} [{timestamp}] [{status:<7}] {message}\n"
        
    print(output, end="", flush=True)
    
    if global_log_text:
        try:
            global_log_text.configure(state='normal')
            global_log_text.insert(tk.END, output)
            global_log_text.see(tk.END)
            global_log_text.configure(state='disabled')
        except:
            pass

# --- Technical Indicators (Using Training Config) ---
def get_atr(high, low, close, n=14):
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def get_adx(high, low, close, n=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    sum_di = plus_di + minus_di
    dx = pd.Series(0.0, index=close.index) 
    valid_mask = (sum_di != 0)
    dx[valid_mask] = 100 * (abs(plus_di[valid_mask] - minus_di[valid_mask]) / sum_di[valid_mask])
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di

def get_rsi(close, n=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_bollinger_bands(close, n=20, std=2):
    sma = close.rolling(window=n).mean()
    std_dev = close.rolling(window=n).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

# --- Feature Creation (Using Training Config) ---
def create_gold_features_from_config(df_raw, config):
    """Create features using the same parameters as training"""
    df = df_raw.copy()
    open, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    # Moving Averages (use training config)
    df['SMA_10'] = close.rolling(window=config['SMA_SHORT']).mean()
    df['SMA_50'] = close.rolling(window=config['SMA_MEDIUM']).mean()
    df['SMA_200'] = close.rolling(window=config['SMA_LONG']).mean()
    df['EMA_12'] = close.ewm(span=config['EMA_FAST']).mean()
    df['EMA_26'] = close.ewm(span=config['EMA_SLOW']).mean()
    
    # Directional Indicators
    df['adx'], df['plus_di'], df['minus_di'] = get_adx(high, low, close, n=config['ADX_PERIOD'])

    # Ichimoku Cloud
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(52)
    df['price_above_kumo'] = np.where((close > senkou_span_a) & (close > senkou_span_b), 1, 0)
    df['price_below_kumo'] = np.where((close < senkou_span_a) & (close < senkou_span_b), 1, 0)
    df['tenkan_cross_kijun'] = np.sign(tenkan_sen - kijun_sen)

    # Momentum Indicators (use training config)
    df['rsi'] = get_rsi(close, n=config['RSI_PERIOD'])
    df['macd'], df['macd_signal'] = get_macd(close, fast=config['MACD_FAST'], 
                                              slow=config['MACD_SLOW'], 
                                              signal=config['MACD_SIGNAL'])
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    df['Momentum_5'] = close - close.shift(5)

    # Volatility Indicators (use training config)
    atr = get_atr(high, low, close, n=config['ATR_PERIOD'])
    df['atr'] = atr
    bb_upper, bb_middle, bb_lower = get_bollinger_bands(close, n=config['BB_PERIOD'], 
                                                         std=config['BB_STD'])
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

    # Volume Analysis
    df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20'].replace(0, np.nan)

    # Candlestick Patterns
    candle_range = high - low
    body_size = abs(close - open)
    df['body_to_range_ratio'] = body_size / candle_range
    df['upper_wick_ratio'] = (high - np.maximum(open, close)) / candle_range
    df['lower_wick_ratio'] = (np.minimum(open, close) - low) / candle_range
    
    df['is_bullish_pinbar'] = ((df['lower_wick_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.2)).astype(int)
    df['is_bearish_pinbar'] = ((df['upper_wick_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.2)).astype(int)

    prev_open, prev_close = open.shift(1), close.shift(1)
    df['is_bullish_engulfing'] = ((close > open) & (prev_close < prev_open) & 
                                  (close > prev_open) & (open < prev_close)).astype(int)
    df['is_bearish_engulfing'] = ((close < open) & (prev_close > prev_open) & 
                                  (close < prev_open) & (open > prev_close)).astype(int)
    
    # Price Distance Features
    df['dist_from_20h_high'] = (close - high.rolling(window=20).max()) / atr
    df['dist_from_20h_low'] = (close - low.rolling(window=20).min()) / atr

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Use feature names from training config
    feature_cols = config['FEATURE_NAMES']
    
    return df, feature_cols

# --- Model Loading ---
def load_all_models():
    try:
        paths = app_state["model_paths"]
        
        # Check all required files
        for key in ["xgb", "lr", "lstm", "scaler", "config"]:
            if not paths[key] or not os.path.exists(paths[key]):
                raise FileNotFoundError(f"Missing {key.upper()} file")
        
        # Load training configuration first
        with open(paths["config"], 'r') as f:
            app_state["training_config"] = json.load(f)
        log_terminal(f"Training config loaded", status="SUCCESS")
        log_terminal(f"  → RSI Period: {app_state['training_config']['RSI_PERIOD']}", status="INFO")
        log_terminal(f"  → SMA Periods: {app_state['training_config']['SMA_SHORT']}, " +
                    f"{app_state['training_config']['SMA_MEDIUM']}, " +
                    f"{app_state['training_config']['SMA_LONG']}", status="INFO")
        log_terminal(f"  → LSTM Sequence: {app_state['training_config']['LSTM_SEQ_LEN']}", status="INFO")
        
        # Load models
        app_state["scaler"] = joblib.load(paths["scaler"])
        app_state["models"]["xgb"] = joblib.load(paths["xgb"])
        app_state["models"]["lr"] = joblib.load(paths["lr"])
        
        with tf.device('/cpu:0'):
            app_state["models"]["lstm"] = load_model(paths["lstm"])
        
        log_terminal("All models loaded successfully", status="SUCCESS")
        return True
        
    except Exception as e:
        log_terminal(f"Model loading failed: {e}", status="ERROR")
        traceback.print_exc()
        return False

# --- Signal Generation ---
def generate_signal(historical_data):
    try:
        if not app_state["models"] or app_state["scaler"] is None or app_state["training_config"] is None:
            return {"error": "Models or config not loaded"}
        
        config = app_state["training_config"]
        
        df = pd.DataFrame(historical_data)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Use config-based feature creation
        df_features, feature_cols = create_gold_features_from_config(df, config)
        
        lstm_seq_len = config['LSTM_SEQ_LEN']
        
        if len(df_features) < lstm_seq_len:
            return {"error": f"Need at least {lstm_seq_len} candles"}
        
        X_new = df_features[feature_cols].iloc[-1:].values
        X_new_scaled = app_state["scaler"].transform(X_new)
        
        pred_xgb = app_state["models"]["xgb"].predict(X_new_scaled)[0]
        pred_lr = app_state["models"]["lr"].predict(X_new_scaled)[0]
        
        X_lstm_seq = app_state["scaler"].transform(
            df_features[feature_cols].iloc[-lstm_seq_len:].values
        )
        X_lstm_seq = X_lstm_seq.reshape(1, lstm_seq_len, -1)
        
        with tf.device('/cpu:0'):
            pred_lstm_proba = app_state["models"]["lstm"].predict(X_lstm_seq, verbose=0)
        pred_lstm = np.argmax(pred_lstm_proba[0])
        
        votes = [pred_xgb, pred_lr, pred_lstm]
        final_prediction = max(set(votes), key=votes.count)
        
        signal_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}
        
        result = {
            "timestamp": str(df.index[-1]),
            "signal": signal_map[final_prediction],
            "models": {
                "xgb": signal_map[pred_xgb],
                "lr": signal_map[pred_lr],
                "lstm": signal_map[pred_lstm]
            },
            "confidence": f"{votes.count(final_prediction)/3:.0%}",
            "price": float(df['Close'].iloc[-1]),
            "config_used": {
                "rsi_period": config['RSI_PERIOD'],
                "timeframe": config['TIMEFRAME']
            }
        }
        
        app_state["latest_signal"] = result
        log_terminal(f"Signal: {result['signal']} ({result['confidence']}) @ {result['price']:.2f}", 
                    status="SIGNAL")
        
        return result
        
    except Exception as e:
        log_terminal(f"Signal generation error: {e}", status="ERROR")
        traceback.print_exc()
        return {"error": str(e)}

# --- Flask Server ---
flask_app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

@flask_app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running" if app_state["is_server_running"] else "stopped",
        "models_loaded": len(app_state["models"]) > 0,
        "config_loaded": app_state["training_config"] is not None
    })

@flask_app.route('/signal', methods=['POST'])
def get_signal_endpoint():
    try:
        data = request.get_json()
        
        if not data or 'candles' not in data:
            return jsonify({"error": "Invalid format"}), 400
        
        candles = data['candles']
        
        min_candles = app_state["training_config"]['LSTM_SEQ_LEN'] + 200 if app_state["training_config"] else 300
        
        if len(candles) < min_candles:
            return jsonify({
                "error": f"Need minimum {min_candles} candles, received {len(candles)}"
            }), 400
        
        signal = generate_signal(candles)
        
        if "error" in signal:
            return jsonify(signal), 500
        
        return jsonify(signal)
        
    except Exception as e:
        log_terminal(f"API error: {e}", status="ERROR")
        return jsonify({"error": str(e)}), 500

# --- Server Thread ---
class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        threading.Thread.__init__(self, daemon=True)
        self.host = host
        self.port = port
        self.server = make_server(self.host, self.port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

# --- UI Application ---
class SignalServerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soluna AI Signal Server")
        self.root.geometry("900x650")
        self.root.configure(bg="#0f0f1e")
        self.root.resizable(False, False)
        
        self.host = tk.StringVar(value=ServerConfig.DEFAULT_HOST)
        self.port = tk.StringVar(value=str(ServerConfig.DEFAULT_PORT))
        
        self.model_xgb = tk.StringVar(value="Not selected")
        self.model_lr = tk.StringVar(value="Not selected")
        self.model_lstm = tk.StringVar(value="Not selected")
        self.model_scaler = tk.StringVar(value="Not selected")
        self.model_config = tk.StringVar(value="Not selected")
        
        self.setup_ui()
    
    def setup_ui(self):
        global global_log_text
        
        container = tk.Frame(self.root, bg="#0f0f1e")
        container.pack(fill='both', expand=True)
        
        # Banner Frame
        banner_frame = tk.Frame(container, bg="#1a1a2e", width=120)
        banner_frame.pack(side='left', fill='y')
        banner_frame.pack_propagate(False)
        banner_img = self.create_server_banner(width=120, height=650)
        banner_photo = ImageTk.PhotoImage(banner_img)
        banner_label = tk.Label(banner_frame, image=banner_photo, bg="#1a1a2e")
        banner_label.image = banner_photo
        banner_label.pack(fill='both', expand=True)
        
        main_frame = tk.Frame(container, bg="#0f0f1e")
        main_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        header_frame = tk.Frame(main_frame, bg="#0f0f1e")
        header_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(header_frame, text="SOLUNA SIGNAL SERVER", bg="#0f0f1e", fg="#00ff88",
                font=("Segoe UI", 14, "bold")).pack(anchor='w')
        tk.Label(header_frame, text="Real-time AI Trading Signal Provider", 
                bg="#0f0f1e", fg="#888888", font=("Segoe UI", 7)).pack(anchor='w')
        
        tk.Frame(main_frame, bg="#00ff88", height=2).pack(fill='x', pady=(0, 5))
        
        # Model Configuration Card
        model_card = self.create_card(main_frame, "⚙️Model Configuration", expand=False)
        model_inner = tk.Frame(model_card, bg="#1a1a2e")
        model_inner.pack(fill='x', padx=5, pady=4)
        
        models = [
            ("XGB Model", self.model_xgb, "xgb", ".pkl"),
            ("LR Model", self.model_lr, "lr", ".pkl"),
            ("LSTM Model", self.model_lstm, "lstm", ".h5"),
            ("Scaler", self.model_scaler, "scaler", ".pkl"),
            ("Config", self.model_config, "config", ".json")
        ]
        
        for i in range(0, len(models), 3):
            row = tk.Frame(model_inner, bg="#1a1a2e")
            row.pack(fill='x', pady=1)
            
            for j in range(3):
                if i + j < len(models):
                    label, var, key, ext = models[i + j]
                    self.create_compact_selector(row, label, var, key, ext)
        
        # Server Settings Card
        server_card = self.create_card(main_frame, "📡Server Settings", expand=False)
        server_inner = tk.Frame(server_card, bg="#1a1a2e")
        server_inner.pack(fill='x', padx=5, pady=4)
        
        settings_row = tk.Frame(server_inner, bg="#1a1a2e")
        settings_row.pack(fill='x', pady=1)
        
        for label, var in [("Host", self.host), ("Port", self.port)]:
            col = tk.Frame(settings_row, bg="#1a1a2e")
            col.pack(side='left', fill='x', expand=True, padx=1)
            
            tk.Label(col, text=label, bg="#1a1a2e", fg="#CCCCCC",
                    font=("Segoe UI", 7, "bold"), anchor='w').pack(fill='x')
            entry_bg = tk.Frame(col, bg="#2a2a3e")
            entry_bg.pack(fill='x')
            tk.Entry(entry_bg, textvariable=var, bg="#2a2a3e", fg="#FFFFFF",
                    font=("Segoe UI", 7), relief="flat",
                    insertbackground="#00ff88").pack(fill='x', padx=4, pady=2)
        
        # API Info Card
        info_card = self.create_card(main_frame, "🔗API Endpoints", expand=False)
        info_inner = tk.Frame(info_card, bg="#1a1a2e")
        info_inner.pack(fill='x', padx=5, pady=4)
        
        info_text = """POST /signal - Generate trading signal
Body: {"candles": [{"time": "2025-01-01 00:00", "open": 2050.0, 
      "high": 2055.0, "low": 2048.0, "close": 2052.0, "volume": 1000}, ...]}
Response: {"signal": "BUY|SELL|NEUTRAL", "confidence": "67%", 
           "price": 2650.50, "config_used": {...}}

GET /health - Check server status
Response: {"status": "running", "models_loaded": true, "config_loaded": true}"""
        
        info_display = tk.Text(info_inner, bg="#0a0a14", fg="#00ff88",
                              font=("Consolas", 8), height=7,
                              relief="flat", borderwidth=0, wrap=tk.WORD)
        info_display.insert("1.0", info_text)
        info_display.configure(state='disabled')
        info_display.pack(fill='x')
        
        # Status
        self.status_label = tk.Label(main_frame, text="⚫ Server Offline", 
                                    bg="#0f0f1e", fg="#BF616A",
                                    font=("Segoe UI", 9, "bold"))
        self.status_label.pack(fill='x', pady=(4, 4))
        
        # Control Buttons
        btn_frame = tk.Frame(main_frame, bg="#0f0f1e")
        btn_frame.pack(fill='x', pady=(0, 4))
        
        self.start_btn = tk.Button(btn_frame, text="▶️START SERVER",
                                   command=self.start_server,
                                   bg="#00ff88", fg="#0f0f1e",
                                   font=("Segoe UI", 9, "bold"),
                                   relief="flat", cursor="hand2",
                                   padx=15, pady=5)
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0, 2))
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️STOP SERVER",
                                  command=self.stop_server,
                                  bg="#BF616A", fg="white",
                                  font=("Segoe UI", 9, "bold"),
                                  relief="flat", cursor="hand2",
                                  padx=15, pady=5, state=tk.DISABLED)
        self.stop_btn.pack(side='right', fill='x', expand=True, padx=(2, 0))
        
        # Log Console
        log_card = self.create_card(main_frame, "🪧Server Log", expand=True)
        log_text = scrolledtext.ScrolledText(log_card, state='disabled', 
                                            bg="#0a0a14", fg="#00ff88", 
                                            font=("Consolas", 8), 
                                            relief="flat", borderwidth=0, wrap=tk.WORD, height=10)
        log_text.pack(fill='both', expand=True, padx=4, pady=4)
        global_log_text = log_text
    
    def create_card(self, parent, title, expand=True):
        card = tk.Frame(parent, bg="#1a1a2e", relief="flat", bd=0)
        card.pack(fill='both' if expand else 'x', expand=expand, pady=(0, 4))
        
        header = tk.Frame(card, bg="#2a2a3e")
        header.pack(fill='x')
        
        tk.Label(header, text=title, bg="#2a2a3e", fg="#00ff88",
                font=("Segoe UI", 8, "bold")).pack(anchor='w', padx=6, pady=3)
        
        return card
    
    def create_compact_selector(self, parent, label, var, key, ext):
        col = tk.Frame(parent, bg="#1a1a2e")
        col.pack(side='left', fill='x', expand=True, padx=1)
        
        tk.Label(col, text=label, bg="#1a1a2e", fg="#CCCCCC",
                font=("Segoe UI", 7, "bold"), anchor='w').pack(fill='x')
        
        selector_frame = tk.Frame(col, bg="#2a2a3e")
        selector_frame.pack(fill='x')
        
        tk.Button(selector_frame, text="SELECT",
                 command=lambda: self.select_model(var, key, ext),
                 bg="#4A90E2", fg="white", font=("Segoe UI", 6, "bold"),
                 relief="flat", cursor="hand2", padx=6, pady=2).pack(side='left', padx=(2,4), pady=2)

        filename_label = tk.Label(selector_frame, textvariable=var, bg="#2a2a3e", fg="#FFFFFF",
                                 font=("Segoe UI", 7), anchor='w', wraplength=150, justify='left')
        filename_label.pack(side='left', fill='x', expand=True, padx=(0,2), pady=2)
    
    def select_model(self, var, key, ext):
        file = filedialog.askopenfilename(
            title=f"Select {key.upper()} File",
            filetypes=((f"{ext} files", f"*{ext}"), ("All files", "*.*"))
        )
        if file:
            var.set(os.path.basename(file))
            app_state["model_paths"][key] = file
            log_terminal(f"Selected {key}: {os.path.basename(file)}", status="INFO")
    
    def start_server(self):
        for key in ["xgb", "lr", "lstm", "scaler", "config"]:
            if not app_state["model_paths"][key]:
                messagebox.showerror("Error", f"Please select {key.upper()} file")
                return
        
        log_terminal("Loading models and configuration...", status="INFO")
        if not load_all_models():
            messagebox.showerror("Error", "Failed to load models or config")
            return
        
        try:
            host = self.host.get()
            port = int(self.port.get())
            
            server = ServerThread(flask_app, host, port)
            server.start()
            
            app_state["server_instance"] = server
            app_state["is_server_running"] = True
            
            log_terminal(f"SERVER ONLINE at http://{host}:{port}", header=True)
            
            self.status_label.config(text=f"🟢 Server Online - {host}:{port}", fg="#00ff88")
            self.start_btn.config(state=tk.DISABLED, bg="#555555")
            self.stop_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", f"Server running at http://{host}:{port}\n\n" +
                              f"Using training config:\n" +
                              f"- RSI: {app_state['training_config']['RSI_PERIOD']}\n" +
                              f"- Timeframe: {app_state['training_config']['TIMEFRAME']}")
            
        except Exception as e:
            log_terminal(f"Server start failed: {e}", status="ERROR")
            messagebox.showerror("Error", str(e))
    
    def stop_server(self):
        if app_state["server_instance"]:
            app_state["server_instance"].shutdown()
            app_state["is_server_running"] = False
            log_terminal("SERVER STOPPED", header=True)
            
            self.status_label.config(text="⚫ Server Offline", fg="#BF616A")
            self.start_btn.config(state=tk.NORMAL, bg="#00ff88")
            self.stop_btn.config(state=tk.DISABLED)
            
            messagebox.showinfo("Stopped", "Server stopped successfully")
    
    def create_server_banner(self, width=120, height=650):
        """Creates server banner image."""
        img = Image.new('RGB', (width, height), '#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Gradient
        for y in range(height):
            ratio = y / height
            r = int(26 + (46 - 26) * ratio)
            g = int(26 + (139 - 26) * ratio)
            b = int(46 + (87 - 46) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Server nodes - centered
        center_start = 260
        for i in range(3):
            y = center_start + i * 60
            draw.rectangle([30, y, 90, y + 25], outline='#00ff88', width=2)
            draw.ellipse([56, y + 8, 64, y + 16], fill='#FFD700')
        
        # Connections
        for i in range(2):
            y1 = center_start + 12 + i * 60
            y2 = y1 + 60
            draw.line([(60, y1 + 25), (60, y2)], fill='#4A90E2', width=2)
        
        # Text
        try:
            title_font = ImageFont.truetype("arial.ttf", 20)
            sub_font = ImageFont.truetype("arial.ttf", 10)
        except:
            title_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
        
        draw.text((width // 2, 35), "SOLUNA", fill='#00ff88', font=title_font, anchor="mm")
        draw.text((width // 2, 60), "SERVER", fill='#FFFFFF', font=title_font, anchor="mm")
        draw.text((width // 2, height - 35), "Signal API", fill='#CCCCCC', font=sub_font, anchor="mm")
        draw.text((width // 2, height - 15), "Real-time Trading", fill='#CCCCCC', font=sub_font, anchor="mm")
        
        return img

# =============================================================================
# 7. APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    
    try:
        root.iconbitmap(".icon/server.ico")
    except:
        pass
    app = SignalServerApp(root)
    
    splash = SplashScreen(root, duration_seconds=5)
    splash.start()
    
    root.mainloop()
