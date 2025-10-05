# -*- coding: utf-8 -*-
# Soluna AI: Model Trainer with Enhanced Terminal Output

# --- Core Libraries ---
import os
import sys
import threading
import time
import warnings
import logging
import json
from datetime import datetime, timedelta
from queue import Queue
import joblib

# --- Data Science & Machine Learning Libraries ---
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- GUI and Imaging Libraries ---
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont

# --- Suppress Warnings for Clean Output ---
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.WARNING)

# =============================================================================
# 1. APPLICATION CONFIGURATION
# =============================================================================
class AppConfig:
    """Stores all static configuration variables for the application."""
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    # MODEL_DIR is now determined by user input, this can be kept as a fallback or removed.
    MODEL_DIR = os.path.join(BASE_PATH, '.models')
    NUM_CLASSES = 3

# =============================================================================
# 2. ENHANCED TERMINAL OUTPUT WITH EMOJIS & PROGRESS
# =============================================================================
global_log_text = None
message_queue = Queue()

# Color codes for terminal
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Emoji mapping for different status types
EMOJI_MAP = {
    "INFO": "ℹ️",
    "DATA": "📊",
    "TASK": "⚙️",
    "MODEL": "🤖",
    "TRAIN": "🎯",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "FATAL": "💥",
    "WARNING": "⚠️",
    "SETUP": "🔧",
    "REPORT": "📋",
    "ACTION": "🚀",
    "PROGRESS": "⏳",
    "COMPLETE": "🎉"
}

def create_progress_bar(current, total, width=30, fill='█', empty='░'):
    """Creates a visual progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = fill * filled + empty * (width - filled)
    percentage = percent * 100
    return f"[{bar}] {percentage:.1f}%"

def log_terminal(message, status="INFO", header=False, progress=None):
    """Enhanced log formatting with emojis and colors."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = EMOJI_MAP.get(status, "•")
    
    output = ""
    if header:
        separator = "═" * 80
        centered_message = message.center(80)
        output = f"\n{TermColors.BOLD}{TermColors.HEADER}{separator}{TermColors.ENDC}\n"
        output += f"{TermColors.BOLD}{TermColors.OKCYAN}{centered_message}{TermColors.ENDC}\n"
        output += f"{TermColors.BOLD}{TermColors.HEADER}{separator}{TermColors.ENDC}\n"
    elif progress:
        bar = create_progress_bar(progress['current'], progress['total'])
        output = f"\r{emoji} [{timestamp}] {message} {bar}"
        sys.stdout.write(output)
        sys.stdout.flush()
        if global_log_text:
            message_queue.put(f"{emoji} [{timestamp}] {message} {bar}\n")
        return
    else:
        color = TermColors.ENDC
        if status in ["SUCCESS", "COMPLETE"]:
            color = TermColors.OKGREEN
        elif status in ["ERROR", "FATAL"]:
            color = TermColors.FAIL
        elif status == "WARNING":
            color = TermColors.WARNING
        elif status in ["MODEL", "TRAIN"]:
            color = TermColors.OKCYAN
        elif status == "DATA":
            color = TermColors.OKBLUE
            
        output = f"{color}{emoji} [{timestamp}] [{status:<8}] {message}{TermColors.ENDC}\n"
    
    if global_log_text:
        clean_output = output.replace(TermColors.HEADER, "").replace(TermColors.OKBLUE, "")
        clean_output = clean_output.replace(TermColors.OKCYAN, "").replace(TermColors.OKGREEN, "")
        clean_output = clean_output.replace(TermColors.WARNING, "").replace(TermColors.FAIL, "")
        clean_output = clean_output.replace(TermColors.ENDC, "").replace(TermColors.BOLD, "")
        clean_output = clean_output.replace(TermColors.UNDERLINE, "")
        message_queue.put(clean_output)
    
    print(output, end='', flush=True)

def _update_log_ui(text):
    """Helper function for thread-safe updates to the Tkinter log widget."""
    if global_log_text:
        try:
            global_log_text.configure(state='normal')
            global_log_text.insert(tk.END, text)
            global_log_text.see(tk.END)
            global_log_text.configure(state='disabled')
            global_log_text.update_idletasks()
        except Exception as e:
            print(f"--- GUI UPDATE ERROR: {e} ---")

def process_queue(root_instance):
    """Checks the message queue and updates the UI from the main thread."""
    while not message_queue.empty():
        try:
            text = message_queue.get_nowait()
            _update_log_ui(text)
        except:
            break
    root_instance.after(100, lambda: process_queue(root_instance))

class UILoggerCallback(tf.keras.callbacks.Callback):
    """Enhanced Keras callback with progress tracking."""
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        log_terminal(f"Starting LSTM training for {self.epochs} epochs", status="TRAIN")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1
        
        progress_bar = create_progress_bar(current_epoch, self.epochs, width=25)
        
        metrics_str = ""
        for key, value in logs.items():
            if isinstance(value, float):
                metrics_str += f"{key}: {value:.4f} | "
        
        log_message = f"Epoch {current_epoch}/{self.epochs} {progress_bar} {metrics_str[:-3]}"
        log_terminal(log_message, status="TRAIN")

# =============================================================================
# 3. TECHNICAL INDICATORS LIBRARY
# =============================================================================
def get_atr(high, low, close, n=14):
    """Calculate Average True Range (ATR)."""
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def get_adx(high, low, close, n=14):
    """Calculate Average Directional Index (ADX)."""
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
    """Calculate Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_macd(close, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_bollinger_bands(close, n=20, std=2):
    """Calculate Bollinger Bands."""
    sma = close.rolling(window=n).mean()
    std_dev = close.rolling(window=n).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

# =============================================================================
# 4. DATA LOADING AND FEATURE ENGINEERING
# =============================================================================
def load_data_from_csv(file_path, timeframe='1h'):
    """Loads, parses, and resamples time-series data from a CSV file."""
    log_terminal(f"Loading dataset: {os.path.basename(file_path)}", status="DATA")
    
    column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(file_path, sep=',', header=None, names=column_names, parse_dates=False)
    
    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns or 'Time' not in df.columns:
        raise ValueError("CSV file must contain 'Date' and 'Time' columns.")
    
    log_terminal("Parsing datetime and resampling data...", status="PROGRESS")
    datetime_series = df['Date'] + ' ' + df['Time']
    df.index = pd.to_datetime(datetime_series, format='%Y.%m.%d %H:%M', errors='coerce')
    df.index.name = 'Datetime'
    df.drop(columns=['Date', 'Time'], inplace=True)
    df = df[df.index.notna()]
    
    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_resampled = df.resample(timeframe).apply(ohlc_dict).dropna()
    
    if df_resampled.empty:
        raise ValueError("Resampling resulted in empty data.")
    
    log_terminal(f"Resampled to {timeframe} timeframe → {len(df_resampled):,} candles", status="SUCCESS")
    
    start_date_5yr = df_resampled.index.max() - timedelta(days=5 * 365.25)
    df_train = df_resampled[df_resampled.index >= start_date_5yr]
    
    log_terminal(f"Training period: {df_train.index.min().strftime('%Y-%m-%d')} to {df_train.index.max().strftime('%Y-%m-%d')}", status="INFO")
    log_terminal(f"Total training samples: {len(df_train):,}", status="INFO")
    
    return df_train

def create_gold_features(df_raw, params):
    """Engineers a comprehensive set of features from the raw OHLCV data."""
    log_terminal("Starting feature engineering process...", status="TASK")
    df = df_raw.copy()
    open, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    log_terminal("Computing moving averages...", status="PROGRESS")
    df['SMA_10'] = close.rolling(window=params['SMA_SHORT']).mean()
    df['SMA_50'] = close.rolling(window=params['SMA_MEDIUM']).mean()
    df['SMA_200'] = close.rolling(window=params['SMA_LONG']).mean()
    df['EMA_12'] = close.ewm(span=params['EMA_FAST']).mean()
    df['EMA_26'] = close.ewm(span=params['EMA_SLOW']).mean()
    
    log_terminal("Calculating directional indicators (ADX)...", status="PROGRESS")
    df['adx'], df['plus_di'], df['minus_di'] = get_adx(high, low, close, n=params['ADX_PERIOD'])

    log_terminal("Building Ichimoku Cloud indicators...", status="PROGRESS")
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(52)
    df['price_above_kumo'] = np.where((close > senkou_span_a) & (close > senkou_span_b), 1, 0)
    df['price_below_kumo'] = np.where((close < senkou_span_a) & (close < senkou_span_b), 1, 0)
    df['tenkan_cross_kijun'] = np.sign(tenkan_sen - kijun_sen)

    log_terminal("Computing momentum indicators (RSI, MACD, ROC)...", status="PROGRESS")
    df['rsi'] = get_rsi(close, n=params['RSI_PERIOD'])
    df['macd'], df['macd_signal'] = get_macd(close, fast=params['MACD_FAST'], slow=params['MACD_SLOW'], signal=params['MACD_SIGNAL'])
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    df['Momentum_5'] = close - close.shift(5)

    log_terminal("Analyzing volatility (ATR, Bollinger Bands)...", status="PROGRESS")
    atr = get_atr(high, low, close, n=params['ATR_PERIOD'])
    df['atr'] = atr
    bb_upper, bb_middle, bb_lower = get_bollinger_bands(close, n=params['BB_PERIOD'], std=params['BB_STD'])
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

    log_terminal("Processing volume indicators (OBV)...", status="PROGRESS")
    df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20'].replace(0, np.nan)

    log_terminal("Detecting candlestick patterns...", status="PROGRESS")
    candle_range = high - low
    body_size = abs(close - open)
    df['body_to_range_ratio'] = body_size / candle_range
    df['upper_wick_ratio'] = (high - np.maximum(open, close)) / candle_range
    df['lower_wick_ratio'] = (np.minimum(open, close) - low) / candle_range
    
    df['is_bullish_pinbar'] = ((df['lower_wick_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.2)).astype(int)
    df['is_bearish_pinbar'] = ((df['upper_wick_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.2)).astype(int)

    prev_open, prev_close = open.shift(1), close.shift(1)
    df['is_bullish_engulfing'] = ((close > open) & (prev_close < prev_open) & (close > prev_open) & (open < prev_close)).astype(int)
    df['is_bearish_engulfing'] = ((close < open) & (prev_close > prev_open) & (close < prev_open) & (open > prev_close)).astype(int)
    
    df['dist_from_20h_high'] = (close - high.rolling(window=20).max()) / atr
    df['dist_from_20h_low'] = (close - low.rolling(window=20).min()) / atr

    log_terminal("Creating target labels...", status="PROGRESS")
    df['Target'] = 1
    future_returns = close.shift(-params['FUTURE_PERIOD']) / close - 1
    df.loc[future_returns >= params['TREND_THRESHOLD'], 'Target'] = 2
    df.loc[future_returns <= -params['TREND_THRESHOLD'], 'Target'] = 0
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    
    log_terminal(f"Feature engineering complete → {len(feature_cols)} features created", status="SUCCESS")
    log_terminal(f"Final dataset size: {len(df):,} samples", status="INFO")
    
    return df, feature_cols

# =============================================================================
# 5. MODEL TRAINING WORKFLOW
# =============================================================================
def train_and_save_models(file_path, params, save_path):
    """Main function to orchestrate the entire model training and saving pipeline."""
    log_terminal("SOLUNA AI TRAINING SESSION INITIATED", header=True)
    
    try:
        df_raw = load_data_from_csv(file_path, params['TIMEFRAME'])
    except Exception as e:
        log_terminal(f"Data loading failed: {e}", status="FATAL")
        return
    
    try:
        df, f_cols = create_gold_features(df_raw, params)
    except Exception as e:
        log_terminal(f"Feature extraction failed: {e}", status="FATAL")
        return

    log_terminal("Preparing training data...", status="SETUP")
    X = df[f_cols].values
    y = df['Target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log_terminal("Data standardization complete", status="SUCCESS")
    
    tscv = TimeSeriesSplit(n_splits=params['CV_SPLITS'])
    log_terminal(f"Cross-validation: TimeSeriesSplit ({tscv.n_splits} splits)", status="SETUP")

    log_terminal("━" * 60, status="INFO")
    log_terminal("MODEL 1/3: XGBoost Classifier with Hyperparameter Tuning", status="MODEL")
    log_terminal("━" * 60, status="INFO")
    
    xgb_param_dist = {
        'n_estimators': randint(params['XGB_N_EST_MIN'], params['XGB_N_EST_MAX']),
        'max_depth': randint(params['XGB_DEPTH_MIN'], params['XGB_DEPTH_MAX']),
        'learning_rate': uniform(params['XGB_LR_MIN'], params['XGB_LR_MAX'] - params['XGB_LR_MIN']),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }
    
    log_terminal(f"Running RandomizedSearchCV ({params['XGB_SEARCH_ITER']} iterations)...", status="PROGRESS")
    xgb = XGBClassifier(random_state=params['RANDOM_STATE'], eval_metric='mlogloss', use_label_encoder=False)
    
    random_search_xgb = RandomizedSearchCV(
        xgb, param_distributions=xgb_param_dist, n_iter=params['XGB_SEARCH_ITER'],
        cv=tscv, scoring='accuracy', n_jobs=-1, random_state=params['RANDOM_STATE'], verbose=1
    )
    random_search_xgb.fit(X_scaled, y)
    
    log_terminal(f"Best XGBoost accuracy: {random_search_xgb.best_score_:.4f}", status="REPORT")
    log_terminal(f"Optimal parameters: {random_search_xgb.best_params_}", status="REPORT")
    
    xgb_model = random_search_xgb.best_estimator_
    model_path = os.path.join(save_path, "xgb_model.pkl")
    joblib.dump(xgb_model, model_path)
    log_terminal(f"XGBoost model saved → {model_path}", status="SUCCESS")

    log_terminal("━" * 60, status="INFO")
    log_terminal("MODEL 2/3: Logistic Regression", status="MODEL")
    log_terminal("━" * 60, status="INFO")
    
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=params['TEST_SIZE'], 
                                               random_state=params['RANDOM_STATE'], stratify=y)
    
    log_terminal("Training Logistic Regression model...", status="PROGRESS")
    lr_model = LogisticRegression(max_iter=1000, random_state=params['RANDOM_STATE'], C=params['LR_C'])
    lr_model.fit(X_train, y_train)
    
    lr_path = os.path.join(save_path, "lr_model.pkl")
    joblib.dump(lr_model, lr_path)
    log_terminal(f"Logistic Regression model saved → {lr_path}", status="SUCCESS")

    log_terminal("━" * 60, status="INFO")
    log_terminal("MODEL 3/3: Deep LSTM Neural Network", status="MODEL")
    log_terminal("━" * 60, status="INFO")
    
    log_terminal("Creating sequence data for LSTM...", status="PROGRESS")
    
    def create_sequences_for_lstm(data, labels, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(data)):
            X_seq.append(data[i-seq_len:i])
            y_seq.append(labels[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_lstm_seq, y_lstm_seq = create_sequences_for_lstm(X_scaled, y, params['LSTM_SEQ_LEN'])
    y_lstm_cat_seq = to_categorical(y_lstm_seq, num_classes=AppConfig.NUM_CLASSES)
    
    log_terminal(f"LSTM sequences created: {X_lstm_seq.shape}", status="INFO")
    
    log_terminal("Building deep LSTM architecture...", status="SETUP")
    lstm_model = models.Sequential([
        layers.Input(shape=(params['LSTM_SEQ_LEN'], len(f_cols))),
        layers.LSTM(params['LSTM_UNITS_1'], return_sequences=True), 
        layers.Dropout(params['LSTM_DROPOUT_1']),
        layers.LSTM(params['LSTM_UNITS_2'], return_sequences=True), 
        layers.Dropout(params['LSTM_DROPOUT_2']),
        layers.LSTM(params['LSTM_UNITS_3']), 
        layers.Dropout(params['LSTM_DROPOUT_3']),
        layers.Dense(AppConfig.NUM_CLASSES, activation='softmax')
    ])
    
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    log_terminal("Model architecture compiled successfully", status="SUCCESS")
    
    es = EarlyStopping(monitor='val_loss', patience=params['LSTM_PATIENCE'], mode='min', restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(save_path, "lstm_model.h5"), 
                        monitor='val_loss', mode='min', save_best_only=True)
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    callbacks_list = [es, mc, rlrp, UILoggerCallback()]
    
    train_size = int(len(X_train) / len(X_scaled) * len(X_lstm_seq))
    X_lstm_train_final, y_lstm_train_final = X_lstm_seq[:train_size], y_lstm_cat_seq[:train_size]
    
    log_terminal(f"Training LSTM (max {params['N_EPOCHS']} epochs, patience={params['LSTM_PATIENCE']})...", status="TRAIN")
    log_terminal("━" * 60, status="INFO")
    
    history = lstm_model.fit(
        X_lstm_train_final, y_lstm_train_final,
        epochs=params['N_EPOCHS'], batch_size=params['BATCH_SIZE'],
        validation_split=0.1, callbacks=callbacks_list, verbose=0
    )
    
    log_terminal("━" * 60, status="INFO")
    log_terminal(f"LSTM training completed at epoch {len(history.history['loss'])}", status="SUCCESS")
    
    scaler_path = os.path.join(save_path, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    log_terminal(f"Data scaler saved → {scaler_path}", status="SUCCESS")
    
    # Save Training Configuration
    config_data = {
        'SMA_SHORT': params['SMA_SHORT'],
        'SMA_MEDIUM': params['SMA_MEDIUM'],
        'SMA_LONG': params['SMA_LONG'],
        'EMA_FAST': params['EMA_FAST'],
        'EMA_SLOW': params['EMA_SLOW'],
        'RSI_PERIOD': params['RSI_PERIOD'],
        'MACD_FAST': params['MACD_FAST'],
        'MACD_SLOW': params['MACD_SLOW'],
        'MACD_SIGNAL': params['MACD_SIGNAL'],
        'ATR_PERIOD': params['ATR_PERIOD'],
        'ADX_PERIOD': params['ADX_PERIOD'],
        'BB_PERIOD': params['BB_PERIOD'],
        'BB_STD': params['BB_STD'],
        'FUTURE_PERIOD': params['FUTURE_PERIOD'],
        'TREND_THRESHOLD': params['TREND_THRESHOLD'],
        'TIMEFRAME': params['TIMEFRAME'],
        'LSTM_SEQ_LEN': params['LSTM_SEQ_LEN'],
        'NUM_FEATURES': len(f_cols),
        'FEATURE_NAMES': f_cols
    }
    
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    log_terminal(f"Training config saved → {config_path}", status="SUCCESS")
    
    log_terminal("ALL MODELS TRAINED AND SAVED SUCCESSFULLY", status="COMPLETE", header=True)
    log_terminal(f"📁 Model directory: {save_path}", status="INFO")
    log_terminal(f"📊 Total features: {len(f_cols)}", status="REPORT")
    log_terminal(f"🎯 Training samples: {len(df):,}", status="REPORT")
    log_terminal(f"🤖 Models: XGBoost, Logistic Regression, Deep LSTM", status="REPORT")
    log_terminal("Training session completed successfully!", header=True)

# =============================================================================
# 6. MAIN TKINTER GUI APPLICATION  
# =============================================================================
class TrainerApp:
    """The main class for the Tkinter GUI application."""
    def __init__(self, root):
        self.root = root
        self.root.title("Soluna AI Trainer")
        
        self.root.geometry("900x700")
        self.root.configure(bg="#0f0f1e")
        self.root.resizable(False, False)

        self.file_path = tk.StringVar(value="No file selected")
        self.save_path = tk.StringVar(value="No save location selected") # <<<< ADDED
        
        self.timeframe = tk.StringVar(value="1h")
        self.future_period = tk.StringVar(value="10")
        self.trend_threshold = tk.StringVar(value="0.003")
        self.test_size = tk.StringVar(value="0.2")
        self.random_state = tk.StringVar(value="42")
        self.cv_splits = tk.StringVar(value="5")
        
        self.sma_short = tk.StringVar(value="10")
        self.sma_medium = tk.StringVar(value="50")
        self.sma_long = tk.StringVar(value="200")
        self.ema_fast = tk.StringVar(value="12")
        self.ema_slow = tk.StringVar(value="26")
        self.rsi_period = tk.StringVar(value="14")
        self.macd_fast = tk.StringVar(value="12")
        self.macd_slow = tk.StringVar(value="26")
        self.macd_signal = tk.StringVar(value="9")
        self.atr_period = tk.StringVar(value="14")
        self.adx_period = tk.StringVar(value="14")
        self.bb_period = tk.StringVar(value="20")
        self.bb_std = tk.StringVar(value="2")
        
        self.xgb_n_est_min = tk.StringVar(value="200")
        self.xgb_n_est_max = tk.StringVar(value="1000")
        self.xgb_depth_min = tk.StringVar(value="5")
        self.xgb_depth_max = tk.StringVar(value="15")
        self.xgb_lr_min = tk.StringVar(value="0.01")
        self.xgb_lr_max = tk.StringVar(value="0.2")
        self.xgb_search_iter = tk.StringVar(value="50")
        
        self.lr_c = tk.StringVar(value="0.1")
        
        self.n_epochs = tk.StringVar(value="200")
        self.lstm_patience = tk.StringVar(value="20")
        self.batch_size = tk.StringVar(value="64")
        self.lstm_seq_len = tk.StringVar(value="60")
        self.lstm_units_1 = tk.StringVar(value="256")
        self.lstm_units_2 = tk.StringVar(value="128")
        self.lstm_units_3 = tk.StringVar(value="64")
        self.lstm_dropout_1 = tk.StringVar(value="0.3")
        self.lstm_dropout_2 = tk.StringVar(value="0.3")
        self.lstm_dropout_3 = tk.StringVar(value="0.2")
        
        self.is_training = False
        self.setup_ui()

    def setup_ui(self):
        """Build all the UI components."""
        global global_log_text
        
        container = tk.Frame(self.root, bg="#0f0f1e")
        container.pack(fill='both', expand=True)
        
        banner_frame = tk.Frame(container, bg="#1a1a2e", width=120)
        banner_frame.pack(side='left', fill='y')
        banner_frame.pack_propagate(False)
        banner_img = self.create_banner_image(width=120, height=700)
        banner_photo = ImageTk.PhotoImage(banner_img)
        banner_label = tk.Label(banner_frame, image=banner_photo, bg="#1a1a2e")
        banner_label.image = banner_photo
        banner_label.pack(fill='both', expand=True)
        
        main_frame = tk.Frame(container, bg="#0f0f1e")
        main_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        header_frame = tk.Frame(main_frame, bg="#0f0f1e")
        header_frame.pack(fill='x', pady=(0, 5))
        tk.Label(header_frame, text="SOLUNA AI TRAINER", bg="#0f0f1e", fg="#FFD700", 
                font=("Segoe UI", 14, "bold")).pack(anchor='w')
        tk.Label(header_frame, text="Advanced Neural Network Training with Full Parameter Control", 
                bg="#0f0f1e", fg="#888888", font=("Segoe UI", 7)).pack(anchor='w')
        tk.Frame(main_frame, bg="#FFD700", height=2).pack(fill='x', pady=(0, 5))
        
        self._create_file_selection_card(main_frame)
        self._create_save_location_card(main_frame) # <<<< ADDED
        
        params_container = tk.Frame(main_frame, bg="#0f0f1e")
        params_container.pack(fill='both', expand=True, pady=(0, 5))
        
        canvas = tk.Canvas(params_container, bg="#0f0f1e", highlightthickness=0, height=280)
        scrollbar = ttk.Scrollbar(params_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f0f1e")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_reqwidth())
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"), width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self._create_data_params_card(scrollable_frame)
        self._create_technical_params_card(scrollable_frame)
        self._create_xgb_params_card(scrollable_frame)
        self._create_lr_params_card(scrollable_frame)
        self._create_lstm_params_card(scrollable_frame)
        
        self.train_btn = tk.Button(main_frame, text="🚀 START TRAINING", 
                                   command=self.start_training_thread, 
                                   bg="#FFD700", fg="#0f0f1e", 
                                   font=("Segoe UI", 10, "bold"), 
                                   relief="flat", cursor="hand2", padx=15, pady=6)
        self.train_btn.pack(fill='x', pady=5)
        
        log_card = self._create_card(main_frame, "📟 Training Console", expand=True)
        log_text = scrolledtext.ScrolledText(log_card, state='disabled', 
                                            bg="#0a0a14", fg="#00ff88", 
                                            font=("Consolas", 8), 
                                            relief="flat", borderwidth=0, wrap=tk.WORD, height=6)
        log_text.pack(fill='both', expand=True, padx=4, pady=4)
        global_log_text = log_text

    def _create_card(self, parent, title, expand=True):
        card = tk.Frame(parent, bg="#1a1a2e", relief="flat", bd=0)
        card.pack(fill='both' if expand else 'x', expand=expand, pady=(0, 4))
        header = tk.Frame(card, bg="#2a2a3e")
        header.pack(fill='x')
        tk.Label(header, text=title, bg="#2a2a3e", fg="#FFD700", 
                font=("Segoe UI", 8, "bold")).pack(anchor='w', padx=6, pady=3)
        return card

    def _create_file_selection_card(self, parent):
        file_card = self._create_card(parent, "⚙️ Dataset", expand=False)
        file_inner = tk.Frame(file_card, bg="#1a1a2e")
        file_inner.pack(fill='x', padx=5, pady=4)
        file_path_frame = tk.Frame(file_inner, bg="#2a2a3e")
        file_path_frame.pack(fill='x')
        tk.Label(file_path_frame, textvariable=self.file_path, bg="#2a2a3e", 
                fg="#4A90E2", anchor="w", font=("Segoe UI", 7)).pack(side='left', 
                fill='x', expand=True, padx=4, pady=3)
        tk.Button(file_path_frame, text="BROWSE", command=self.select_file, 
                 bg="#4A90E2", fg="white", font=("Segoe UI", 7, "bold"), 
                 relief="flat", cursor="hand2", padx=10, pady=2).pack(side='right', 
                 padx=2, pady=2)
    
    # <<<< ADDED METHOD
    def _create_save_location_card(self, parent):
        save_card = self._create_card(parent, "📁 Save Model File", expand=False)
        save_inner = tk.Frame(save_card, bg="#1a1a2e")
        save_inner.pack(fill='x', padx=5, pady=4)
        save_path_frame = tk.Frame(save_inner, bg="#2a2a3e")
        save_path_frame.pack(fill='x')
        tk.Label(save_path_frame, textvariable=self.save_path, bg="#2a2a3e", 
                fg="#4AE290", anchor="w", font=("Segoe UI", 7)).pack(side='left', 
                fill='x', expand=True, padx=4, pady=3)
        tk.Button(save_path_frame, text="BROWSE", command=self.select_save_directory, 
                 bg="#4AE290", fg="#0f0f1e", font=("Segoe UI", 7, "bold"), 
                 relief="flat", cursor="hand2", padx=10, pady=2).pack(side='right', 
                 padx=2, pady=2)

    def _create_data_params_card(self, parent):
        card = self._create_card(parent, "📊 Data & General Parameters", expand=False)
        grid = tk.Frame(card, bg="#1a1a2e")
        grid.pack(fill='both', expand=True, padx=5, pady=4)
        
        params = [
            ("Timeframe", self.timeframe, "1m, 5m, 15m, 1h, 4h, 1d"),
            ("Future Period", self.future_period, "Candles ahead for target"),
            ("Trend Threshold", self.trend_threshold, "Min % change for signal"),
            ("Test Split", self.test_size, "Train/test split ratio"),
            ("Random State", self.random_state, "Seed for reproducibility"),
            ("CV Splits", self.cv_splits, "Cross-validation folds"),
        ]
        
        self._create_param_grid(grid, params, cols=3)
    
    def _create_technical_params_card(self, parent):
        card = self._create_card(parent, "📈 Technical Indicators Parameters", expand=False)
        grid = tk.Frame(card, bg="#1a1a2e")
        grid.pack(fill='both', expand=True, padx=5, pady=4)
        
        params = [
            ("SMA Short", self.sma_short, "Short MA"),
            ("SMA Medium", self.sma_medium, "Medium MA"),
            ("SMA Long", self.sma_long, "Long MA"),
            ("EMA Fast", self.ema_fast, "Fast EMA"),
            ("EMA Slow", self.ema_slow, "Slow EMA"),
            ("RSI Period", self.rsi_period, "RSI period"),
            ("MACD Fast", self.macd_fast, "MACD fast"),
            ("MACD Slow", self.macd_slow, "MACD slow"),
            ("MACD Signal", self.macd_signal, "MACD signal"),
            ("ATR Period", self.atr_period, "ATR period"),
            ("ADX Period", self.adx_period, "ADX period"),
            ("BB Period", self.bb_period, "BB period"),
            ("BB Std Dev", self.bb_std, "BB std dev"),
        ]
        
        self._create_param_grid(grid, params, cols=4)
    
    def _create_xgb_params_card(self, parent):
        card = self._create_card(parent, "🌳 XGBoost Parameters", expand=False)
        grid = tk.Frame(card, bg="#1a1a2e")
        grid.pack(fill='both', expand=True, padx=5, pady=4)
        
        params = [
            ("N Est Min", self.xgb_n_est_min, "Min trees"),
            ("N Est Max", self.xgb_n_est_max, "Max trees"),
            ("Depth Min", self.xgb_depth_min, "Min depth"),
            ("Depth Max", self.xgb_depth_max, "Max depth"),
            ("LR Min", self.xgb_lr_min, "Min LR"),
            ("LR Max", self.xgb_lr_max, "Max LR"),
            ("Search Iter", self.xgb_search_iter, "Trials"),
        ]
        
        self._create_param_grid(grid, params, cols=4)
    
    def _create_lr_params_card(self, parent):
        card = self._create_card(parent, "📉 Logistic Regression Parameters", expand=False)
        grid = tk.Frame(card, bg="#1a1a2e")
        grid.pack(fill='both', expand=True, padx=5, pady=4)
        
        params = [
            ("Regularization C", self.lr_c, "Inverse regularization"),
        ]
        
        self._create_param_grid(grid, params, cols=2)
    
    def _create_lstm_params_card(self, parent):
        card = self._create_card(parent, "🧠 LSTM Neural Network Parameters", expand=False)
        grid = tk.Frame(card, bg="#1a1a2e")
        grid.pack(fill='both', expand=True, padx=5, pady=4)
        
        params = [
            ("Max Epochs", self.n_epochs, "Max epochs"),
            ("Patience", self.lstm_patience, "Early stop"),
            ("Batch Size", self.batch_size, "Batch size"),
            ("Seq Length", self.lstm_seq_len, "Seq length"),
            ("LSTM Units 1", self.lstm_units_1, "Layer 1"),
            ("LSTM Units 2", self.lstm_units_2, "Layer 2"),
            ("LSTM Units 3", self.lstm_units_3, "Layer 3"),
            ("Dropout 1", self.lstm_dropout_1, "Drop 1"),
            ("Dropout 2", self.lstm_dropout_2, "Drop 2"),
            ("Dropout 3", self.lstm_dropout_3, "Drop 3"),
        ]
        
        self._create_param_grid(grid, params, cols=4)
    
    def _create_param_grid(self, parent, params, cols=2):
        for i in range(0, len(params), cols):
            row = tk.Frame(parent, bg="#1a1a2e")
            row.pack(fill='x', pady=1)
            
            for j in range(cols):
                if i + j < len(params):
                    label_text, var, tooltip = params[i + j]
                    col = tk.Frame(row, bg="#1a1a2e")
                    col.pack(side='left', fill='x', expand=True, padx=1)
                    
                    label_frame = tk.Frame(col, bg="#1a1a2e")
                    label_frame.pack(fill='x')
                    tk.Label(label_frame, text=label_text, bg="#1a1a2e", fg="#CCCCCC", 
                            font=("Segoe UI", 7, "bold"), anchor='w').pack(side='left')
                    tk.Label(label_frame, text="ⓘ", bg="#1a1a2e", fg="#4A90E2", 
                            font=("Segoe UI", 6), cursor="question_arrow").pack(side='left', padx=1)
                    
                    entry_bg = tk.Frame(col, bg="#2a2a3e")
                    entry_bg.pack(fill='x')
                    entry = tk.Entry(entry_bg, textvariable=var, bg="#2a2a3e", fg="#FFFFFF", 
                            font=("Segoe UI", 7), relief="flat", 
                            insertbackground="#FFD700")
                    entry.pack(fill='x', padx=4, pady=2)
                    
                    self._create_tooltip(entry, tooltip)

    def _create_tooltip(self, widget, text):
        def on_enter(event):
            widget.config(bg="#3a3a4e")
        def on_leave(event):
            widget.config(bg="#2a2a3e")
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def select_file(self):
        file_selected = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_selected: 
            self.file_path.set(file_selected)

    # <<<< ADDED METHOD
    def select_save_directory(self):
        dir_selected = filedialog.askdirectory(
            title="Select a Parent Folder to Save Models In"
        )
        if dir_selected:
            self.save_path.set(dir_selected)

    def start_training_thread(self):
        # <<<< MODIFIED
        if self.file_path.get() == "No file selected":
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        if self.save_path.get() == "No save location selected":
            messagebox.showerror("Error", "Please select a save location first.")
            return
        if self.is_training:
            messagebox.showwarning("In Progress", "Training is already running.")
            return
        
        parent_path = self.save_path.get()
        save_path = os.path.join(parent_path, ".model")

        try:
            params = {
                'TIMEFRAME': self.timeframe.get(),
                'FUTURE_PERIOD': int(self.future_period.get()),
                'TREND_THRESHOLD': float(self.trend_threshold.get()),
                'TEST_SIZE': float(self.test_size.get()),
                'RANDOM_STATE': int(self.random_state.get()),
                'CV_SPLITS': int(self.cv_splits.get()),
                
                'SMA_SHORT': int(self.sma_short.get()),
                'SMA_MEDIUM': int(self.sma_medium.get()),
                'SMA_LONG': int(self.sma_long.get()),
                'EMA_FAST': int(self.ema_fast.get()),
                'EMA_SLOW': int(self.ema_slow.get()),
                'RSI_PERIOD': int(self.rsi_period.get()),
                'MACD_FAST': int(self.macd_fast.get()),
                'MACD_SLOW': int(self.macd_slow.get()),
                'MACD_SIGNAL': int(self.macd_signal.get()),
                'ATR_PERIOD': int(self.atr_period.get()),
                'ADX_PERIOD': int(self.adx_period.get()),
                'BB_PERIOD': int(self.bb_period.get()),
                'BB_STD': float(self.bb_std.get()),
                
                'XGB_N_EST_MIN': int(self.xgb_n_est_min.get()),
                'XGB_N_EST_MAX': int(self.xgb_n_est_max.get()),
                'XGB_DEPTH_MIN': int(self.xgb_depth_min.get()),
                'XGB_DEPTH_MAX': int(self.xgb_depth_max.get()),
                'XGB_LR_MIN': float(self.xgb_lr_min.get()),
                'XGB_LR_MAX': float(self.xgb_lr_max.get()),
                'XGB_SEARCH_ITER': int(self.xgb_search_iter.get()),
                
                'LR_C': float(self.lr_c.get()),
                
                'N_EPOCHS': int(self.n_epochs.get()),
                'LSTM_PATIENCE': int(self.lstm_patience.get()),
                'BATCH_SIZE': int(self.batch_size.get()),
                'LSTM_SEQ_LEN': int(self.lstm_seq_len.get()),
                'LSTM_UNITS_1': int(self.lstm_units_1.get()),
                'LSTM_UNITS_2': int(self.lstm_units_2.get()),
                'LSTM_UNITS_3': int(self.lstm_units_3.get()),
                'LSTM_DROPOUT_1': float(self.lstm_dropout_1.get()),
                'LSTM_DROPOUT_2': float(self.lstm_dropout_2.get()),
                'LSTM_DROPOUT_3': float(self.lstm_dropout_3.get()),
            }
            
            if not (0 < params['TEST_SIZE'] < 1):
                raise ValueError("Test size must be between 0 and 1.")
            if params['TIMEFRAME'] not in ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w']:
                raise ValueError("Invalid timeframe. Use: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return
        
        self.is_training = True
        self.train_btn.config(text="⏳ TRAINING IN PROGRESS...", state=tk.DISABLED, bg="#888888")
        threading.Thread(target=self._run_training_process, 
                        args=(self.file_path.get(), params, save_path), daemon=True).start()

    def _run_training_process(self, file_path, params, save_path):
        try:
            os.makedirs(save_path, exist_ok=True)
            log_terminal(f"Model directory selected: {save_path}", status="ACTION")
            train_and_save_models(file_path, params, save_path)
        except Exception as e:
            log_terminal(f"Critical error occurred: {e}", status="FATAL")
        finally:
            self.root.after(0, self._reset_ui_state)

    def _reset_ui_state(self):
        self.is_training = False
        self.train_btn.config(text="🚀 START TRAINING", state=tk.NORMAL, bg="#FFD700")
    
    def create_banner_image(self, width=120, height=700):
        """Generates a stylish banner image for the GUI sidebar."""
        img = Image.new('RGB', (width, height), '#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        for y in range(height):
            ratio = y / height
            r = int(26 + (138 - 26) * ratio)
            g = int(26 + (114 - 26) * ratio)
            b = int(46 + (173 - 46) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        pattern_height = 4 * 60
        center_start = 100 + (550 - pattern_height) // 2
        
        for i in range(4):
            x = width // 2
            y = center_start + i * 60
            radius = 16 - i * 2
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                         outline='#FFD700', width=2)
        
        points = [
            (25, center_start + 15),
            (95, center_start + 60 + 15),
            (50, center_start + 120 + 15),
            (90, center_start + 180 + 15),
            (35, center_start + 240)
        ]
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='#4A90E2', width=2)
            draw.ellipse([points[i][0] - 3, points[i][1] - 3, 
                         points[i][0] + 3, points[i][1] + 3], 
                         fill='#FFD700')
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 20)
            sub_font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            title_font, sub_font = ImageFont.load_default(), ImageFont.load_default()
        
        draw.text((width // 2, 35), "SOLUNA", fill='#FFD700', font=title_font, anchor="mm")
        draw.text((width // 2, 60), "AI", fill='#FFFFFF', font=title_font, anchor="mm")
        
        draw.text((width // 2, height - 35), "Deep Learning", fill='#CCCCCC', font=sub_font, anchor="mm")
        draw.text((width // 2, height - 15), "Trading Platform", fill='#CCCCCC', font=sub_font, anchor="mm")
        
        return img

# =============================================================================
# 7. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    try:
        root.iconbitmap(".ico/trainer.ico")
    except:
        pass
    app = TrainerApp(root)
    process_queue(root) 
    log_terminal("Soluna AI Trainer System Ready", status="SUCCESS", header=True)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass 
    root.mainloop()