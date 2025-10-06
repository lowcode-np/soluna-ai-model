# -*- coding: utf-8 -*-
# Soluna AI: File-Based Signal Server

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
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

# =============== CONFIGURATION ===============
# MT4/MT5 Common Files path - แก้ตาม username ของคุณ
MT4_COMMON_PATH = os.path.expanduser("~/AppData/Roaming/MetaQuotes/Terminal/Common/Files")
REQUEST_DIR = os.path.join(MT4_COMMON_PATH, "SolunaBridge", "requests")
RESPONSE_DIR = os.path.join(MT4_COMMON_PATH, "SolunaBridge", "responses")
CONFIG_FILE = os.path.join(MT4_COMMON_PATH, "SolunaBridge", "server_config.txt")

BRIDGE_CHECK_INTERVAL = 0.5

# สร้างโฟลเดอร์
os.makedirs(REQUEST_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

# เขียนไฟล์ config บอก path
with open(CONFIG_FILE, 'w') as f:
    f.write(f"{MT4_COMMON_PATH}\n")
    f.write(f"SolunaBridge/requests\n")
    f.write(f"SolunaBridge/responses\n")

# --- Splash Screen ---
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
        tk.Label(main_frame, text="File-Based Signal Server", bg="#1a1a2e", fg="#CCCCCC", 
                 font=("Segoe UI", 12)).pack(pady=(0, 40))

        self.status_label = tk.Label(main_frame, text="Initializing...", bg="#1a1a2e", fg="#888888",
                                     font=("Segoe UI", 10))
        self.status_label.pack(pady=(20, 5))

        s = ttk.Style()
        s.theme_use('clam')
        s.configure("green.Horizontal.TProgressbar", foreground='#00ff88', background='#00ff88', 
                   troughcolor='#2a2a3e', bordercolor="#1a1a2e", lightcolor="#1a1a2e", darkcolor="#1a1a2e")
        self.progress = ttk.Progressbar(main_frame, style="green.Horizontal.TProgressbar", 
                                       orient="horizontal", length=400, mode='determinate')
        self.progress.pack(pady=(0, 20))

    def _animate(self):
        if self.current_step <= self.total_steps:
            self.progress['value'] = self.current_step * self.progress_increment
            
            progress_percent = (self.current_step / self.total_steps) * 100
            if progress_percent < 40:
                self.status_label.config(text="Initializing...")
            elif progress_percent < 80:
                self.status_label.config(text="Loading models...")
            else:
                self.status_label.config(text="Ready...")
            
            self.current_step += 1
            self.splash.after(self.update_interval_ms, self._animate)
        else:
            self.close()

    def close(self):
        self.splash.destroy()
        self.root.deiconify()

    def start(self):
        self.splash.after(0, self._animate)

# --- Global State ---
app_state = {
    "is_server_running": False,
    "bridge_instance": None,
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
    "SIGNAL": "📡",
    "FILE": "📂"
}

def log_terminal(message, status="INFO", header=False):
    timestamp = datetime.now().strftime("%H:%M:%S")
    
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

# --- Technical Indicators (เหมือนเดิม) ---
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

def create_gold_features_from_config(df_raw, config):
    df = df_raw.copy()
    open, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    df['SMA_10'] = close.rolling(window=config['SMA_SHORT']).mean()
    df['SMA_50'] = close.rolling(window=config['SMA_MEDIUM']).mean()
    df['SMA_200'] = close.rolling(window=config['SMA_LONG']).mean()
    df['EMA_12'] = close.ewm(span=config['EMA_FAST']).mean()
    df['EMA_26'] = close.ewm(span=config['EMA_SLOW']).mean()
    
    df['adx'], df['plus_di'], df['minus_di'] = get_adx(high, low, close, n=config['ADX_PERIOD'])

    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(52)
    df['price_above_kumo'] = np.where((close > senkou_span_a) & (close > senkou_span_b), 1, 0)
    df['price_below_kumo'] = np.where((close < senkou_span_a) & (close < senkou_span_b), 1, 0)
    df['tenkan_cross_kijun'] = np.sign(tenkan_sen - kijun_sen)

    df['rsi'] = get_rsi(close, n=config['RSI_PERIOD'])
    df['macd'], df['macd_signal'] = get_macd(close, fast=config['MACD_FAST'], 
                                              slow=config['MACD_SLOW'], 
                                              signal=config['MACD_SIGNAL'])
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    df['Momentum_5'] = close - close.shift(5)

    atr = get_atr(high, low, close, n=config['ATR_PERIOD'])
    df['atr'] = atr
    bb_upper, bb_middle, bb_lower = get_bollinger_bands(close, n=config['BB_PERIOD'], 
                                                         std=config['BB_STD'])
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

    df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20'].replace(0, np.nan)

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
    
    df['dist_from_20h_high'] = (close - high.rolling(window=20).max()) / atr
    df['dist_from_20h_low'] = (close - low.rolling(window=20).min()) / atr

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df, config['FEATURE_NAMES']

def load_all_models():
    try:
        paths = app_state["model_paths"]
        
        for key in ["xgb", "lr", "lstm", "scaler", "config"]:
            if not paths[key] or not os.path.exists(paths[key]):
                raise FileNotFoundError(f"Missing {key.upper()} file")
        
        with open(paths["config"], 'r') as f:
            app_state["training_config"] = json.load(f)
        log_terminal(f"Training config loaded", status="SUCCESS")
        
        app_state["scaler"] = joblib.load(paths["scaler"])
        app_state["models"]["xgb"] = joblib.load(paths["xgb"])
        app_state["models"]["lr"] = joblib.load(paths["lr"])
        
        with tf.device('/cpu:0'):
            app_state["models"]["lstm"] = load_model(paths["lstm"])
        
        log_terminal("All models loaded", status="SUCCESS")
        return True
        
    except Exception as e:
        log_terminal(f"Model loading failed: {e}")
        traceback.print_exc()
        return False

def generate_signal(historical_data):
    try:
        if not app_state["models"] or app_state["scaler"] is None or app_state["training_config"] is None:
            return {"error": "Models not loaded"}
        
        config = app_state["training_config"]
        
        df = pd.DataFrame(historical_data)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
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
            "price": float(df['Close'].iloc[-1])
        }
        
        log_terminal(f"Signal: {result['signal']} ({result['confidence']}) @ {result['price']:.2f}", 
                    status="SIGNAL")
        
        return result
        
    except Exception as e:
        log_terminal(f"Signal error: {e}", status="ERROR")
        traceback.print_exc()
        return {"error": str(e)}

# =============== FILE BRIDGE ===============
def file_bridge_worker():
    log_terminal("File Bridge started (Multi-Terminal Support)", status="FILE")
    log_terminal(f"Monitoring: {REQUEST_DIR}", status="FILE")
    processed = {}
    processing = set()
    
    while app_state["is_server_running"]:
        try:
            if not os.path.exists(REQUEST_DIR):
                time.sleep(1)
                continue
            
            for item in os.listdir(REQUEST_DIR):
                terminal_path = os.path.join(REQUEST_DIR, item)
                
                if not os.path.isdir(terminal_path):
                    continue
                
                terminal_id = item
                terminal_res_dir = os.path.join(RESPONSE_DIR, terminal_id)
                
                os.makedirs(terminal_res_dir, exist_ok=True)
                
                if terminal_id not in processed:
                    processed[terminal_id] = set()
                
                try:
                    files = os.listdir(terminal_path)
                except:
                    continue
                
                for filename in files:
                    if not filename.endswith('.json'):
                        continue
                    
                    file_key = f"{terminal_id}/{filename}"
                    
                    if filename in processed[terminal_id] or file_key in processing:
                        continue
                    
                    filepath = os.path.join(terminal_path, filename)
                    
                    try:
                        processing.add(file_key)
                        time.sleep(0.1)
                        
                        if not os.path.exists(filepath):
                            processing.discard(file_key)
                            continue
                        
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        request_id = f"[{terminal_id}] {filename.replace('.json', '')}"
                        log_terminal(f"Request: {request_id}", status="FILE")
                        
                        result = generate_signal(data['candles'])
                        
                        response_file = os.path.join(terminal_res_dir, filename)
                        with open(response_file, 'w') as f:
                            json.dump(result, f)
                        
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        
                        processed[terminal_id].add(filename)
                        processing.discard(file_key)
                        
                        log_terminal(f"Response: {result.get('signal', 'ERROR')} -> {terminal_id}", 
                                   status="FILE")
                        
                    except Exception as e:
                        log_terminal(f"Error processing {file_key}: {e}", status="ERROR")
                        processing.discard(file_key)
                        try:
                            error_data = {"error": str(e)}
                            response_file = os.path.join(terminal_res_dir, filename)
                            with open(response_file, 'w') as f:
                                json.dump(error_data, f)
                            if os.path.exists(filepath):
                                os.remove(filepath)
                        except:
                            pass
                
                if len(processed[terminal_id]) > 100:
                    processed[terminal_id].clear()
            
            if len(processing) > 200:
                processing.clear()
            
            time.sleep(BRIDGE_CHECK_INTERVAL)
            
        except Exception as e:
            log_terminal(f"Bridge error: {e}", status="ERROR")
            traceback.print_exc()
            time.sleep(2)
    
    log_terminal("File Bridge stopped", status="FILE")
    
# --- UI ---
class SignalServerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soluna AI File Server")
        self.root.geometry("900x600")
        self.root.configure(bg="#0f0f1e")
        self.root.resizable(False, False)
        
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
        
        # Banner
        banner_frame = tk.Frame(container, bg="#1a1a2e", width=120)
        banner_frame.pack(side='left', fill='y')
        banner_frame.pack_propagate(False)
        banner_img = self.create_server_banner(width=120, height=600)
        banner_photo = ImageTk.PhotoImage(banner_img)
        banner_label = tk.Label(banner_frame, image=banner_photo, bg="#1a1a2e")
        banner_label.image = banner_photo
        banner_label.pack(fill='both', expand=True)
        
        main_frame = tk.Frame(container, bg="#0f0f1e")
        main_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(main_frame, text="SOLUNA FILE SERVER", bg="#0f0f1e", fg="#00ff88",
                font=("Segoe UI", 16, "bold")).pack(pady=(0, 5))
        
        tk.Label(main_frame, text="File-Based Communication - No WebRequest needed", 
                bg="#0f0f1e", fg="#888888", font=("Segoe UI", 9)).pack()
        
        tk.Frame(main_frame, bg="#00ff88", height=2).pack(fill='x', pady=10)
        
        # Models
        model_frame = tk.LabelFrame(main_frame, text="⚙️ Models", bg="#1a1a2e", fg="#00ff88",
                                    font=("Segoe UI", 10, "bold"))
        model_frame.pack(fill='x', pady=5)
        
        models = [
            ("XGB", self.model_xgb, "xgb", ".pkl"),
            ("LR", self.model_lr, "lr", ".pkl"),
            ("LSTM", self.model_lstm, "lstm", ".h5"),
            ("Scaler", self.model_scaler, "scaler", ".pkl"),
            ("Config", self.model_config, "config", ".json")
        ]
        
        # 2 columns layout
        for i in range(0, len(models), 2):
            row = tk.Frame(model_frame, bg="#1a1a2e")
            row.pack(fill='x', padx=5, pady=2)
            
            for j in range(2):
                if i + j < len(models):
                    label, var, key, ext = models[i + j]
                    
                    col = tk.Frame(row, bg="#1a1a2e")
                    col.pack(side='left', fill='x', expand=True, padx=2)
                    
                    tk.Label(col, text=f"{label}:", bg="#1a1a2e", fg="#CCCCCC",
                            width=8, anchor='w', font=("Segoe UI", 8)).pack(side='left')
                    tk.Button(col, text="SELECT", command=lambda v=var, k=key, e=ext: self.select_model(v, k, e),
                             bg="#4A90E2", fg="white", font=("Segoe UI", 7)).pack(side='left', padx=3)
                    tk.Label(col, textvariable=var, bg="#1a1a2e", fg="#FFFFFF",
                            anchor='w', font=("Segoe UI", 7)).pack(side='left', fill='x', expand=True)
        
        # Info
        info_frame = tk.LabelFrame(main_frame, text="🔗 Connection Info", bg="#1a1a2e", fg="#00ff88",
                                   font=("Segoe UI", 10, "bold"))
        info_frame.pack(fill='x', pady=5)
        
        info_text = f"""Files will be stored in MT4/MT5 Common Files:
{MT4_COMMON_PATH}

No network configuration needed.
No WebRequest setup needed.
Works with all MT4/MT5 versions."""
        
        tk.Label(info_frame, text=info_text, bg="#1a1a2e", fg="#CCCCCC",
                justify='left', font=("Consolas", 8)).pack(padx=10, pady=10)
        
        # Status
        self.status_label = tk.Label(main_frame, text="Server Offline", 
                                    bg="#0f0f1e", fg="#BF616A",
                                    font=("Segoe UI", 11, "bold"))
        self.status_label.pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(main_frame, bg="#0f0f1e")
        btn_frame.pack(fill='x', pady=5)
        
        self.start_btn = tk.Button(btn_frame, text="▶️ START SERVER",
                                   command=self.start_server,
                                   bg="#00ff88", fg="#0f0f1e",
                                   font=("Segoe UI", 10, "bold"),
                                   padx=20, pady=8)
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️ STOP SERVER",
                                  command=self.stop_server,
                                  bg="#BF616A", fg="white",
                                  font=("Segoe UI", 10, "bold"),
                                  padx=20, pady=8, state=tk.DISABLED)
        self.stop_btn.pack(side='right', fill='x', expand=True, padx=(5, 0))
        
        # Log
        log_frame = tk.LabelFrame(main_frame, text="📡 Server Log", bg="#1a1a2e", fg="#00ff88",
                                 font=("Segoe UI", 10, "bold"))
        log_frame.pack(fill='both', expand=True, pady=5)
        
        log_text = scrolledtext.ScrolledText(log_frame, state='disabled', 
                                            bg="#0a0a14", fg="#00ff88", 
                                            font=("Consolas", 9), height=18)
        log_text.pack(fill='both', expand=True, padx=5, pady=5)
        global_log_text = log_text
    
    def create_server_banner(self, width=120, height=600):
        """Create server banner"""
        img = Image.new('RGB', (width, height), '#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Gradient background
        for y in range(height):
            ratio = y / height
            r = int(26 + (46 - 26) * ratio)
            g = int(26 + (139 - 26) * ratio)
            b = int(46 + (87 - 46) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Server nodes
        center_start = 240
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
        draw.text((width // 2, height - 35), "File-Based", fill='#CCCCCC', font=sub_font, anchor="mm")
        draw.text((width // 2, height - 15), "No WebRequest", fill='#CCCCCC', font=sub_font, anchor="mm")
        
        return img
    
    def select_model(self, var, key, ext):
        file = filedialog.askopenfilename(
            title=f"Select {key.upper()}",
            filetypes=((f"{ext} files", f"*{ext}"), ("All files", "*.*"))
        )
        if file:
            var.set(os.path.basename(file))
            app_state["model_paths"][key] = file
            log_terminal(f"Selected {key}: {os.path.basename(file)}", status="INFO")
    
    def start_server(self):
        for key in ["xgb", "lr", "lstm", "scaler", "config"]:
            if not app_state["model_paths"][key]:
                messagebox.showerror("Error", f"Please select {key.upper()}")
                return
        
        log_terminal("Loading models...", status="INFO")
        if not load_all_models():
            messagebox.showerror("Error", "Failed to load models")
            return
        
        try:
            bridge = threading.Thread(target=file_bridge_worker, daemon=True)
            bridge.start()
            app_state["bridge_instance"] = bridge
            app_state["is_server_running"] = True
            
            log_terminal("SERVER ONLINE", header=True)
            
            self.status_label.config(text="✅ Server Online", fg="#00ff88")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", 
                              f"✅ Server running!\n\n"
                              f"📂 Files location:\n{MT4_COMMON_PATH}\n\n"
                              f"No WebRequest setup needed!")
            
        except Exception as e:
            log_terminal(f"Start failed: {e}", status="ERROR")
            messagebox.showerror("Error", str(e))
    
    def stop_server(self):
        app_state["is_server_running"] = False
        time.sleep(1)
        
        log_terminal("SERVER STOPPED", header=True)
        
        self.status_label.config(text="⚫ Server Offline", fg="#BF616A")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        messagebox.showinfo("Stopped", "Server stopped")

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