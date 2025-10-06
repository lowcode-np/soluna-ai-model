# Soluna AI: Robust File-Based Signal Server

# ==================================
# 1. IMPORTS
# ==================================

import os
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

# ML/DL Libraries
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==================================
# 2. CONFIGURATION & SETUP
# ==================================

# --- Path Configuration ---
MT4_COMMON_PATH = os.path.expanduser("~\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
REQUEST_DIR = os.path.join(MT4_COMMON_PATH, "SolunaBridge", "requests")
RESPONSE_DIR = os.path.join(MT4_COMMON_PATH, "SolunaBridge", "responses")
CONFIG_FILE = os.path.join(MT4_COMMON_PATH, "SolunaBridge", "server_config.txt")

# --- Robustness & Performance Settings ---
BRIDGE_CHECK_INTERVAL = 0.3  # Faster check interval for new requests (in seconds)
FILE_WAIT_TIME = 0.15        # Time to wait for file writing to complete before reading/deleting
MAX_PROCESSED_CACHE = 50     # Max number of processed file keys to keep in cache per terminal
MAX_PROCESSING_SET = 100     # Max number of files concurrently being processed
FILE_MAX_AGE_SECONDS = 300   # Files older than 5 minutes will be cleaned up
MAX_RETRIES = 3              # Max retries for file operations

# Create necessary directories and config file
os.makedirs(REQUEST_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

with open(CONFIG_FILE, 'w') as f:
    f.write(f"{MT4_COMMON_PATH}\n")
    f.write(f"SolunaBridge/requests\n")
    f.write(f"SolunaBridge/responses\n")

# ==================================
# 3. GLOBAL STATE & LOGGING
# ==================================

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
    },
    "stats": {
        "total_requests": 0,
        "successful": 0,
        "errors": 0,
        "last_request_time": None
    }
}

global_log_text = None

# --- Emoji Mapping ---
EMOJI_MAP = {
    "INFO": "ℹ️",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "SIGNAL": "📡",
    "FILE": "📂",
    "WARNING": "⚠️",
    "CLEANUP": "🧹"
}

def log_terminal(message, status="INFO", header=False):
    """Prints and updates the GUI log with a timestamp and status."""
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
            # Handle potential Tkinter errors during updates
            pass

# ==================================
# 4. SAFE FILE OPERATIONS
# ==================================

def safe_file_read(filepath, max_retries=MAX_RETRIES):
    """Safely reads and decodes a JSON file with retry and size check."""
    for attempt in range(max_retries):
        try:
            time.sleep(FILE_WAIT_TIME)  # Wait for file system stability

            if not os.path.exists(filepath):
                return None

            # Check file size stability
            size1 = os.path.getsize(filepath)
            time.sleep(0.05)
            size2 = os.path.getsize(filepath)

            if size1 != size2:
                time.sleep(0.1)
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(0.1)
                continue
            log_terminal(f"JSON decode error: {os.path.basename(filepath)}", status="ERROR")
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1)
                continue
            log_terminal(f"Read error: {e}", status="ERROR")
            return None

    return None

def safe_file_write(filepath, data, max_retries=MAX_RETRIES):
    """Safely writes data to a JSON file using a temp file for atomicity."""
    for attempt in range(max_retries):
        try:
            temp_file = filepath + ".tmp"

            # Write to temp file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomic rename
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_file, filepath)

            return True

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1)
                continue
            log_terminal(f"Write error: {e}", status="ERROR")
            return False

    return False

def safe_file_delete(filepath, max_retries=MAX_RETRIES):
    """Safely deletes a file with retry for permission errors."""
    for attempt in range(max_retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except PermissionError:
            time.sleep(0.1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1)
                continue
            log_terminal(f"Delete error: {e}", status="WARNING")
            return False
    return False

def cleanup_old_files(directory, max_age_seconds=FILE_MAX_AGE_SECONDS):
    """Deletes files older than max_age_seconds in the specified directory."""
    try:
        now = time.time()
        count = 0

        for root, dirs, files in os.walk(directory):
            for filename in files:
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(root, filename)

                try:
                    file_age = now - os.path.getmtime(filepath)
                    if file_age > max_age_seconds:
                        safe_file_delete(filepath)
                        count += 1
                except:
                    pass

        if count > 0:
            log_terminal(f"Cleaned {count} old files", status="CLEANUP")

    except Exception as e:
        log_terminal(f"Cleanup error: {e}", status="WARNING")

# ==================================
# 5. TECHNICAL INDICATORS & FEATURE ENGINEERING
# ==================================

# --- Technical Indicators ---
def get_atr(high, low, close, n=14):
    """Calculates Average True Range (ATR)."""
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def get_adx(high, low, close, n=14):
    """Calculates Average Directional Index (ADX) and +/- DI."""
    plus_dm = high.diff().where(lambda x: x > 0, 0)
    minus_dm = (-low.diff()).where(lambda x: x > 0, 0)
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
    """Calculates Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_macd(close, fast=12, slow=26, signal=9):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_bollinger_bands(close, n=20, std=2):
    """Calculates Bollinger Bands (BB)."""
    sma = close.rolling(window=n).mean()
    std_dev = close.rolling(window=n).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def create_gold_features_from_config(df_raw, config):
    """Generates all features required for the models based on the config."""
    df = df_raw.copy()
    open_p, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    # Moving Averages
    df['SMA_10'] = close.rolling(window=config['SMA_SHORT']).mean()
    df['SMA_50'] = close.rolling(window=config['SMA_MEDIUM']).mean()
    df['SMA_200'] = close.rolling(window=config['SMA_LONG']).mean()
    df['EMA_12'] = close.ewm(span=config['EMA_FAST']).mean()
    df['EMA_26'] = close.ewm(span=config['EMA_SLOW']).mean()

    # Directional Movement
    df['adx'], df['plus_di'], df['minus_di'] = get_adx(high, low, close, n=config['ADX_PERIOD'])

    # Ichimoku Cloud (simplified indicators)
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(52)
    df['price_above_kumo'] = np.where((close > senkou_span_a) & (close > senkou_span_b), 1, 0)
    df['price_below_kumo'] = np.where((close < senkou_span_a) & (close < senkou_span_b), 1, 0)
    df['tenkan_cross_kijun'] = np.sign(tenkan_sen - kijun_sen)

    # Oscillators
    df['rsi'] = get_rsi(close, n=config['RSI_PERIOD'])
    df['macd'], df['macd_signal'] = get_macd(close, fast=config['MACD_FAST'],
                                              slow=config['MACD_SLOW'],
                                              signal=config['MACD_SIGNAL'])
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    df['Momentum_5'] = close - close.shift(5)

    # Volatility & Range
    atr = get_atr(high, low, close, n=config['ATR_PERIOD'])
    df['atr'] = atr
    bb_upper, bb_middle, bb_lower = get_bollinger_bands(close, n=config['BB_PERIOD'],
                                                         std=config['BB_STD'])
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

    # Volume Indicators
    df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20'].replace(0, np.nan)

    # Price Action & Candlestick Patterns
    candle_range = high - low
    body_size = abs(close - open_p)
    df['body_to_range_ratio'] = body_size / candle_range
    df['upper_wick_ratio'] = (high - np.maximum(open_p, close)) / candle_range
    df['lower_wick_ratio'] = (np.minimum(open_p, close) - low) / candle_range

    # Pinbar detection
    df['is_bullish_pinbar'] = ((df['lower_wick_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.2)).astype(int)
    df['is_bearish_pinbar'] = ((df['upper_wick_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.2)).astype(int)

    # Engulfing pattern detection
    prev_open, prev_close = open_p.shift(1), close.shift(1)
    df['is_bullish_engulfing'] = ((close > open_p) & (prev_close < prev_open) &
                                  (close > prev_open) & (open_p < prev_close)).astype(int)
    df['is_bearish_engulfing'] = ((close < open_p) & (prev_close > prev_open) &
                                  (close < prev_open) & (open_p > prev_close)).astype(int)

    # Range proximity
    df['dist_from_20h_high'] = (close - high.rolling(window=20).max()) / atr
    df['dist_from_20h_low'] = (close - low.rolling(window=20).min()) / atr

    # Final cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df, config['FEATURE_NAMES']

# ==================================
# 6. MODEL MANAGEMENT & SIGNAL GENERATION
# ==================================

def load_all_models():
    """Loads all machine learning models (XGB, LR, LSTM, Scaler, Config) from disk."""
    try:
        paths = app_state["model_paths"]

        # 1. Check if all required files are selected and exist
        for key in ["xgb", "lr", "lstm", "scaler", "config"]:
            if not paths[key] or not os.path.exists(paths[key]):
                raise FileNotFoundError(f"Missing {key.upper()} file: {paths[key]}")

        # 2. Load Configuration
        with open(paths["config"], 'r') as f:
            app_state["training_config"] = json.load(f)
        log_terminal(f"Training config loaded", status="SUCCESS")

        # 3. Load Scaler and Sklearn Models (XGB, LR)
        app_state["scaler"] = joblib.load(paths["scaler"])
        app_state["models"]["xgb"] = joblib.load(paths["xgb"])
        app_state["models"]["lr"] = joblib.load(paths["lr"])

        # 4. Load Keras Model (LSTM) - force CPU to prevent GPU/threading issues
        with tf.device('/cpu:0'):
            app_state["models"]["lstm"] = load_model(paths["lstm"])

        log_terminal("All models loaded", status="SUCCESS")
        return True

    except Exception as e:
        log_terminal(f"Model loading failed: {e}", status="ERROR")
        traceback.print_exc()
        return False

def generate_signal(historical_data):
    """
    Processes historical data, generates features, and runs ensemble prediction.
    The ensemble uses a simple majority vote (XGB, LR, LSTM).
    """
    try:
        if not app_state["models"] or app_state["scaler"] is None or app_state["training_config"] is None:
            return {"error": "Models not loaded"}

        config = app_state["training_config"]

        # Data preparation
        df = pd.DataFrame(historical_data)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Feature creation
        df_features, feature_cols = create_gold_features_from_config(df, config)

        lstm_seq_len = config['LSTM_SEQ_LEN']

        # Ensure enough data for LSTM sequence
        if len(df_features) < lstm_seq_len:
            return {"error": f"Need at least {lstm_seq_len} candles"}

        # Prepare input for Sklearn models (XGB, LR)
        X_new = df_features[feature_cols].iloc[-1:].values
        X_new_scaled = app_state["scaler"].transform(X_new)

        pred_xgb = app_state["models"]["xgb"].predict(X_new_scaled)[0]
        pred_lr = app_state["models"]["lr"].predict(X_new_scaled)[0]

        # Prepare input for Keras model (LSTM)
        X_lstm_seq = app_state["scaler"].transform(
            df_features[feature_cols].iloc[-lstm_seq_len:].values
        )
        X_lstm_seq = X_lstm_seq.reshape(1, lstm_seq_len, -1)

        with tf.device('/cpu:0'):
            pred_lstm_proba = app_state["models"]["lstm"].predict(X_lstm_seq, verbose=0)
        pred_lstm = np.argmax(pred_lstm_proba[0])

        # Ensemble (Majority Vote)
        votes = [pred_xgb, pred_lr, pred_lstm]
        final_prediction = max(set(votes), key=votes.count)

        signal_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}

        # Prepare result
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
            "server_time": datetime.now().isoformat()
        }

        app_state["stats"]["successful"] += 1
        return result

    except Exception as e:
        app_state["stats"]["errors"] += 1
        log_terminal(f"Signal error: {e}", status="ERROR")
        traceback.print_exc()
        return {"error": str(e), "error_details": traceback.format_exc()}

# ==================================
# 7. FILE BRIDGE WORKER
# ==================================

def file_bridge_worker():
    """
    Worker thread that continuously monitors the request directory for new files,
    processes them, and writes the response to the corresponding response directory.
    Supports multi-terminal setup (subdirectories under REQUEST_DIR).
    """
    log_terminal("File Bridge started (Multi-Terminal Support)", status="FILE")
    log_terminal(f"Monitoring: {REQUEST_DIR}", status="FILE")

    processed = {}  # Cache of processed files per terminal: {terminal_id: {filename, ...}}
    processing = set() # Set of files currently being processed: {terminal_id/filename, ...}
    last_cleanup = time.time()

    while app_state["is_server_running"]:
        try:
            # Cleanup check every 60 seconds
            if time.time() - last_cleanup > 60:
                cleanup_old_files(REQUEST_DIR)
                cleanup_old_files(RESPONSE_DIR)
                last_cleanup = time.time()

            if not os.path.exists(REQUEST_DIR):
                time.sleep(1)
                continue

            # Iterate through potential terminal subdirectories
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
                    if not filename.endswith('.json') or filename.endswith('.tmp'):
                        continue

                    file_key = f"{terminal_id}/{filename}"

                    if filename in processed[terminal_id] or file_key in processing:
                        continue

                    filepath = os.path.join(terminal_path, filename)

                    try:
                        processing.add(file_key)

                        # Read request
                        data = safe_file_read(filepath)

                        if data is None:
                            processing.discard(file_key)
                            safe_file_delete(filepath)
                            continue

                        request_id = f"[{terminal_id}] {filename.replace('.json', '')}"
                        log_terminal(f"Request: {request_id}", status="FILE")

                        app_state["stats"]["total_requests"] += 1
                        app_state["stats"]["last_request_time"] = datetime.now()

                        # Process signal
                        result = generate_signal(data.get('candles', []))

                        # Write response
                        response_file = os.path.join(terminal_res_dir, filename)
                        if safe_file_write(response_file, result):
                            log_terminal(f"Response: {result.get('signal', 'ERROR')} -> {terminal_id}",
                                       status="FILE")
                        else:
                            log_terminal(f"Failed to write response: {filename}", status="ERROR")

                        # Delete processed request
                        safe_file_delete(filepath)

                        processed[terminal_id].add(filename)
                        processing.discard(file_key)

                    except Exception as e:
                        log_terminal(f"Error processing {file_key}: {e}", status="ERROR")
                        processing.discard(file_key)

                        try:
                            # Write an error response
                            error_data = {
                                "error": str(e),
                                "error_details": traceback.format_exc(),
                                "server_time": datetime.now().isoformat()
                            }
                            response_file = os.path.join(terminal_res_dir, filename)
                            safe_file_write(response_file, error_data)
                            safe_file_delete(filepath)
                        except:
                            pass

                # Manage cache size
                if len(processed[terminal_id]) > MAX_PROCESSED_CACHE:
                    processed[terminal_id] = set(list(processed[terminal_id])[-MAX_PROCESSED_CACHE//2:])

            # Manage processing set size
            if len(processing) > MAX_PROCESSING_SET:
                processing = set(list(processing)[-MAX_PROCESSING_SET//2:])

            time.sleep(BRIDGE_CHECK_INTERVAL)

        except Exception as e:
            log_terminal(f"Bridge error: {e}", status="ERROR")
            traceback.print_exc()
            time.sleep(2)

    log_terminal("File Bridge stopped", status="FILE")

# ==================================
# 8. GUI APPLICATION
# ==================================

class SplashScreen:
    """A simple splash screen displayed during application startup."""
    def __init__(self, root, duration_seconds=5):
        self.root = root
        # ... (implementation as in original code) ...
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

class SignalServerApp:
    """The main application window and GUI logic."""
    def __init__(self, root):
        self.root = root
        self.root.title("Soluna AI File Server (Robust)")
        self.root.geometry("900x650")
        self.root.configure(bg="#0f0f1e")
        self.root.resizable(False, False)

        self.model_xgb = tk.StringVar(value="Not selected")
        self.model_lr = tk.StringVar(value="Not selected")
        self.model_lstm = tk.StringVar(value="Not selected")
        self.model_scaler = tk.StringVar(value="Not selected")
        self.model_config = tk.StringVar(value="Not selected")

        self.setup_ui()
        self.update_stats()

    def setup_ui(self):
        # ... (UI setup implementation as in original code) ...
        global global_log_text

        container = tk.Frame(self.root, bg="#0f0f1e")
        container.pack(fill='both', expand=True)

        # Banner
        banner_frame = tk.Frame(container, bg="#1a1a2e", width=120)
        banner_frame.pack(side='left', fill='y')
        banner_frame.pack_propagate(False)
        banner_img = self.create_server_banner(width=120, height=650)
        banner_photo = ImageTk.PhotoImage(banner_img)
        banner_label = tk.Label(banner_frame, image=banner_photo, bg="#1a1a2e")
        banner_label.image = banner_photo
        banner_label.pack(fill='both', expand=True)

        main_frame = tk.Frame(container, bg="#0f0f1e")
        main_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        tk.Label(main_frame, text="SOLUNA SERVER", bg="#0f0f1e", fg="#00ff88",
                font=("Segoe UI", 16, "bold")).pack(pady=(0, 5))

        tk.Label(main_frame, text="Robust File-Based Communication",
                bg="#0f0f1e", fg="#888888", font=("Segoe UI", 9)).pack()

        tk.Frame(main_frame, bg="#00ff88", height=2).pack(fill='x', pady=10)

        # Statistics Frame
        stats_frame = tk.LabelFrame(main_frame, text="📊 Statistics", bg="#1a1a2e", fg="#00ff88",
                                    font=("Segoe UI", 10, "bold"))
        stats_frame.pack(fill='x', pady=5)

        stats_inner = tk.Frame(stats_frame, bg="#1a1a2e")
        stats_inner.pack(fill='x', padx=10, pady=5)

        self.stats_label = tk.Label(stats_inner, text="No requests yet",
                                    bg="#1a1a2e", fg="#CCCCCC",
                                    font=("Consolas", 8), justify='left')
        self.stats_label.pack(anchor='w')

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

        info_text = f"""Files: {MT4_COMMON_PATH}
✓ No network configuration needed
✓ No WebRequest setup needed
✓ Works with all MT4/MT5 versions
✓ Automatic retry on errors
✓ Old file cleanup (5 min)"""

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
                                            font=("Consolas", 9), height=15)
        log_text.pack(fill='both', expand=True, padx=5, pady=5)
        global_log_text = log_text

    def update_stats(self):
        """Updates the real-time statistics displayed in the GUI."""
        if app_state["is_server_running"]:
            stats = app_state["stats"]

            success_rate = 0
            if stats["total_requests"] > 0:
                success_rate = (stats["successful"] / stats["total_requests"]) * 100

            last_req = "Never"
            if stats["last_request_time"]:
                last_req = stats["last_request_time"].strftime("%H:%M:%S")

            text = f"""Total Requests: {stats["total_requests"]}
Successful: {stats["successful"]} ({success_rate:.1f}%)
Errors: {stats["errors"]}
Last Request: {last_req}"""

            self.stats_label.config(text=text)

        self.root.after(1000, self.update_stats)  # Update every second

    def create_server_banner(self, width=120, height=650):
        """Creates the custom graphical server banner for the side panel."""
        # ... (Image generation implementation as in original code) ...
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
        center_start = 280
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
        draw.text((width // 2, height - 55), "Robust", fill='#CCCCCC', font=sub_font, anchor="mm")
        draw.text((width // 2, height - 35), "File-Based", fill='#CCCCCC', font=sub_font, anchor="mm")
        draw.text((width // 2, height - 15), "No WebRequest", fill='#CCCCCC', font=sub_font, anchor="mm")

        return img

    def select_model(self, var, key, ext):
        """Opens a file dialog to select a model file."""
        file = filedialog.askopenfilename(
            title=f"Select {key.upper()}",
            filetypes=((f"{ext} files", f"*{ext}"), ("All files", "*.*"))
        )
        if file:
            var.set(os.path.basename(file))
            app_state["model_paths"][key] = file
            log_terminal(f"Selected {key}: {os.path.basename(file)}", status="INFO")

    def start_server(self):
        """Initializes models and starts the file bridge worker thread."""
        # Check if all models are selected
        for key in ["xgb", "lr", "lstm", "scaler", "config"]:
            if not app_state["model_paths"][key]:
                messagebox.showerror("Error", f"Please select {key.upper()}")
                return

        log_terminal("Loading models...", status="INFO")
        if not load_all_models():
            messagebox.showerror("Error", "Failed to load models")
            return

        try:
            # Reset stats
            app_state["stats"] = {
                "total_requests": 0,
                "successful": 0,
                "errors": 0,
                "last_request_time": None
            }

            # Start worker thread
            bridge = threading.Thread(target=file_bridge_worker, daemon=True)
            bridge.start()
            app_state["bridge_instance"] = bridge
            app_state["is_server_running"] = True

            log_terminal("SERVER ONLINE", header=True)
            log_terminal("Robust mode: Auto-retry, safe file operations", status="SUCCESS")

            self.status_label.config(text="✅ Server Online (Robust)", fg="#00ff88")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            messagebox.showinfo("Success",
                              f"✅ Server running!\n\n"
                              f"📂 Files: {MT4_COMMON_PATH}\n\n"
                              f"🛡️ Features:\n"
                              f"• Automatic retry on errors\n"
                              f"• Safe file operations\n"
                              f"• Old file cleanup\n"
                              f"• Multi-terminal support")

        except Exception as e:
            log_terminal(f"Start failed: {e}", status="ERROR")
            messagebox.showerror("Error", str(e))

    def stop_server(self):
        """Sets the flag to stop the worker thread and updates the GUI."""
        app_state["is_server_running"] = False
        time.sleep(1) # Give the thread a moment to shut down gracefully

        log_terminal("SERVER STOPPED", header=True)

        self.status_label.config(text="⚫ Server Offline", fg="#BF616A")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        stats = app_state["stats"]
        messagebox.showinfo("Stopped",
                          f"Server stopped\n\n"
                          f"Total requests: {stats['total_requests']}\n"
                          f"Successful: {stats['successful']}\n"
                          f"Errors: {stats['errors']}")

# ==================================
# 9. MAIN EXECUTION
# ==================================

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw() # Hide main window initially

    try:
        # Attempt to set application icon
        root.iconbitmap(".icon/server.ico")
    except:
        pass

    app = SignalServerApp(root)
    splash = SplashScreen(root, duration_seconds=5)
    splash.start()

    root.mainloop()