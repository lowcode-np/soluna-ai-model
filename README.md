# 🌙 Soluna AI - Trading Signal Platform

<div align="center">

[![SourceCode](https://img.shields.io/badge/SourceCode-v1.5-orange.svg)](https://drive.google.com/uc?export=download&id=1f0z6NaHOn1DuQI-YsDoBN0sMpoi87o_R)

**Open-source AI-powered trading signal platform with complete ML pipeline**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)](https://xgboost.readthedocs.io/)

[Features](#-features) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[API Reference](#-api-reference) •
[Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [MQL Integration](#-mql-integration)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Soluna AI** is a comprehensive open-source platform designed for building, training, and deploying machine learning models for financial market trading signals. The platform combines the power of ensemble learning with an intuitive GUI and production-ready API server.

### Why Soluna AI?

- 🎨 **User-Friendly**: GUI-based tools eliminate complex coding for model training
- 🔧 **Highly Customizable**: Fine-tune technical indicators and neural network architecture
- 🚀 **Production Ready**: RESTful API server for real-time signal generation
- 🤖 **Ensemble Learning**: Combines XGBoost, Logistic Regression, and LSTM for robust predictions
- 🔌 **MT4/MT5 Ready**: Includes MQH library for seamless MetaTrader integration
- 📊 **Complete Pipeline**: From data import to live deployment in one platform

---

## ✨ Features

### 🎓 AI Trainer
![Soluna AI Trainer](https://cdn.imgchest.com/files/f089fa0acaab.png)
- **Visual Configuration Interface**: Configure all parameters through an intuitive GUI
- **Advanced Feature Engineering**: 
  - Moving Averages (SMA, EMA)
  - Momentum Indicators (RSI, MACD, ROC)
  - Volatility Metrics (ATR, Bollinger Bands)
  - Directional Indicators (ADX, Plus/Minus DI)
  - Ichimoku Cloud signals
  - Candlestick pattern recognition
  - Volume analysis (OBV)
- **Multi-Model Training**:
  - XGBoost with automated hyperparameter tuning
  - Logistic Regression for linear patterns
  - Deep LSTM Neural Network for sequential learning
- **Smart Labeling**: Automated signal generation based on configurable profit targets and ATR
- **Model Export**: Save trained models, scalers, and configuration for deployment

### 🖥️ Signal Server
![Soluna AI Server](https://cdn.imgchest.com/files/6ca64abe28ca.png)
- **RESTful API**: Production-ready Flask server
- **MT4 File Bridge**: Integrated file-based communication (no WebRequest needed)
- **Ensemble Prediction**: Combines predictions from all three models using majority voting
- **Real-time Processing**: Generate signals from live market data
- **Health Monitoring**: Built-in health check endpoints
- **Configuration Tracking**: Uses training configuration for consistent feature generation
- **Detailed Responses**: Returns model votes, confidence levels, and metadata

### 🔌 MetaTrader Integration

- **Universal MQH Library**: Compatible with both MT4 and MT5
- **Easy Integration**: Simple struct-based API
- **Example EA Included**: Ready-to-use Expert Advisor template
- **Robust Error Handling**: Comprehensive error messages and validation

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SOLUNA AI PLATFORM                      │
└──────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │                           │
        ┌───────▼────────┐           ┌──────▼───────┐
        │  AI TRAINER    │           │SIGNAL SERVER │
        │   (Training)   │           │ (Production) │
        └───────┬────────┘           └──────┬───────┘
                │                           │
        ┌───────▼────────┐           ┌──────▼───────┐
        │ • Data Import  │           │ • REST API   │
        │ • Feature Eng  │           │ • Ensemble   │
        │ • Model Train  │           │ • MT4 Bridge │
        │ • Export       │           │ • Monitoring │
        └───────┬────────┘           └──────┬───────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                    ┌─────────▼──────────┐
                    │       MODELS       │
                    │ • xgb_model.pkl    │
                    │ • lr_model.pkl     │
                    │ • lstm_model.h5    │
                    │ • scaler.pkl       │
                    │ • config.json      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  MT4 FILE BRIDGE   │
                    │                    │
                    │  C:\MT4Bridge\     │
                    │  ├─ requests\      │
                    │  └─ responses\     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  MQL INTEGRATION   │
                    │ • MT4/MT5 Library  │
                    │ • Example EA       │
                    └────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Step 1: Clone Repository

```bash
git clone https://github.com/lowcode-np/soluna-ai-model.git
cd soluna-ai-model
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Manual Installation:**
```bash
pip install tensorflow pandas scikit-learn xgboost numpy flask joblib pillow
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import xgboost; print('✅ Installation successful!')"
```

---

## 🚀 Quick Start

### Train Your First Model (5 Minutes)

1. **Prepare Data**
   - Format: CSV file with columns: `Date, Time, Open, High, Low, Close, Volume`
   - Minimum: 10,000 rows recommended for good results

2. **Launch Trainer**
   ```bash
   "Soluna AI - Trainer.vbs"
   ```

3. **Configure & Train**
   - Click **BROWSE** → Select your CSV file
   - Adjust parameters (or use defaults)
   - Click **🚀 START TRAINING**
   - Choose empty folder for output
   - Wait for training completion (5-30 minutes depending on data size)

4. **Launch Server**
   ```bash
   "Soluna AI - Server.vbs"
   ```

5. **Load Models**
   - Select all 5 files from training output:
     - `xgb_model.pkl`
     - `lr_model.pkl`
     - `lstm_model.h5`
     - `scaler.pkl`
     - `config.json`
   - Click **▶️ START SERVER**

6. **Test API**
   ```bash
   curl http://127.0.0.1:5000/health
   ```

---

## 📖 Usage Guide

### Part 1: Training Models

#### Data Preparation

Your CSV file should follow this format:

```csv
Date,Time,Open,High,Low,Close,Volume
2024-01-01,00:00,2050.50,2055.30,2048.20,2052.80,15420
2024-01-01,01:00,2052.80,2060.10,2051.50,2058.40,18930
...
```

**Tips:**
- Use at least 10,000 rows for meaningful training
- More data = better model performance
- Ensure no missing values
- Data should be sorted chronologically

#### Trainer Interface Guide

**📊 Data Configuration**
- **Data File**: Your OHLCV CSV file
- **Start Row**: Skip initial rows if needed
- **Timeframe**: Select your trading timeframe
- **ATR Period**: For profit target calculation (default: 14)
- **Profit Target**: Multiple of ATR for signal labeling (default: 1.5)

**📈 Technical Indicators**
- **RSI Period**: Relative Strength Index lookback (default: 14)
- **MACD**: Fast/Slow/Signal periods (default: 12/26/9)
- **SMA**: Short/Medium/Long periods (default: 10/50/200)
- **EMA**: Fast/Slow periods (default: 12/26)
- **Bollinger Bands**: Period and Standard Deviation (default: 20/2)
- **ADX/ATR Periods**: Directional and volatility indicators (default: 14)

**🤖 Model Settings**

*XGBoost:*
- **Max Depth**: Tree depth (default: 6)
- **Learning Rate**: Step size shrinkage (default: 0.1)
- **N Estimators**: Number of trees (default: 100)
- **CV Folds**: Cross-validation folds (default: 3)

*Logistic Regression:*
- **Max Iterations**: Solver iterations (default: 1000)
- **C (Regularization)**: Inverse regularization strength (default: 1.0)

*LSTM:*
- **Units**: LSTM layer size (default: 128)
- **Dropout**: Regularization rate (default: 0.3)
- **Sequence Length**: Lookback period (default: 30)
- **Epochs**: Training iterations (default: 50)
- **Batch Size**: Samples per update (default: 32)

#### Training Process

1. Click **START TRAINING**
2. Monitor progress in log console:
   ```
   ✅ Data loaded: 15000 rows
   ✅ Features created: 45 features
   ✅ Signals labeled: BUY=3500, SELL=3200, NEUTRAL=8300
   ⏳ Training XGBoost with hyperparameter tuning...
   ✅ XGBoost trained: Accuracy=0.78
   ⏳ Training Logistic Regression...
   ✅ Logistic Regression trained: Accuracy=0.72
   ⏳ Training LSTM Neural Network...
   ✅ LSTM trained: Accuracy=0.75
   ✅ All models saved successfully!
   ```

### Part 2: Running Signal Server

#### Server Interface Guide

**⚙️ Model Configuration**
- Load all 5 files from training output folder
- Must load all 5 files (CSV file is NOT required for server operation)
- The `config.json` stores parameter values used during training

**📡 Server Settings**
- **Host**: Default `127.0.0.1` (localhost)
- **Port**: Default `5000` (change if port conflict)

**📗 API Endpoints**
- `POST /signal`: Get trading signal
- `GET /health`: Check server status

#### Server Monitoring

Monitor server activity in real-time:
```
🟢 Server Online - 127.0.0.1:5000
✅ Training config loaded
  → RSI Period: 14
  → SMA Periods: 10, 50, 200
  → LSTM Sequence: 30
📡 Signal: BUY (67%) @ 2058.50
```

---

## 🔌 API Reference

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "running",
  "models_loaded": true,
  "config_loaded": true
}
```

### Get Trading Signal

**Endpoint:** `POST /signal`

**Request Body:**
```json
{
  "candles": [
    {
      "time": "2025-01-01 00:00",
      "open": 2050.0,
      "high": 2055.0,
      "low": 2048.0,
      "close": 2052.0,
      "volume": 1000
    },
    {
      "time": "2025-01-01 01:00",
      "open": 2052.0,
      "high": 2060.0,
      "low": 2051.0,
      "close": 2058.0,
      "volume": 1200
    }
    // ... minimum 300 candles required
  ]
}
```

**Response:**
```json
{
  "timestamp": "2025-01-01 01:00:00",
  "signal": "BUY",
  "models": {
    "xgb": "BUY",
    "lr": "NEUTRAL",
    "lstm": "BUY"
  },
  "confidence": "67%",
  "price": 2058.0,
  "config_used": {
    "rsi_period": 14,
    "timeframe": "1h"
  }
}
```

**Error Response:**
```json
{
  "error": "Need minimum 300 candles, received 150"
}
```

### Python Example

```python
import requests
import json

url = "http://127.0.0.1:5000/signal"
headers = {"Content-Type": "application/json"}

# Prepare candle data
candles = [
    {
        "time": "2025-01-01 00:00",
        "open": 2050.0,
        "high": 2055.0,
        "low": 2048.0,
        "close": 2052.0,
        "volume": 1000
    }
    # ... add more candles
]

data = {"candles": candles}

# Get signal
response = requests.post(url, headers=headers, json=data)
signal = response.json()

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']}")
print(f"Price: {signal['price']}")
```

---

## 🔌 MQL Integration

### Setup for MT4/MT5

1. **Create Bridge Folder**
   ```
   cmd
   mkdir C:\MT4Bridge\requests
   mkdir C:\MT4Bridge\responses
   ```
   
2. **Copy Library**
   ```
   Copy SolunaSignalClient.mqh to:
   MT4/MT5 → MQL4/MQL5 → Include folder
   ```

3. **Start Server**
  - Python server includes integrated MT4 Bridge
  - No additional configuration needed
  - Bridge monitors C:\MT4Bridge\ automatically

### Basic Usage

```cpp
#include <SolunaSignalClient.mqh>

// Initialize client (File-based - no server config)
CSolunaSignalClient client;
client.SetMinCandles(300);
client.SetTimeout(30);  // seconds

// Check file system
if(client.CheckHealth())
{
   Print("✅ File system ready!");
}

// Get signal
SolunaSignal signal;
if(client.GetSignal(_Symbol, PERIOD_H1, 500, signal))
{
   Print("Signal: ", signal.signal);
   Print("Confidence: ", signal.confidence);
   
   if(signal.signal == "BUY")
   {
      // Execute buy order
   }
}
```

### Complete EA Example

See `SolunaSignalExample.mq4/5` for a full working Expert Advisor with:
- Automatic signal checking
- Trade execution
- Error handling
- Position management

---

## ⚙️ Configuration

### Training Configuration (config.json)

This file is automatically generated during training and contains all parameters used. **Note:** The server only uses this file to read parameter values - it does NOT require the original CSV file.

```json
{
  "DATA_FILE": "XAUUSD_H1.csv",  // ⚠️ Reference only, NOT used by server
  "TIMEFRAME": "1h",
  "START_ROW": 0,
  "ATR_PERIOD": 14,
  "PROFIT_TARGET_ATR_MULTIPLIER": 1.5,
  
  "RSI_PERIOD": 14,
  "MACD_FAST": 12,
  "MACD_SLOW": 26,
  "MACD_SIGNAL": 9,
  "SMA_SHORT": 10,
  "SMA_MEDIUM": 50,
  "SMA_LONG": 200,
  "EMA_FAST": 12,
  "EMA_SLOW": 26,
  "BB_PERIOD": 20,
  "BB_STD": 2,
  "ADX_PERIOD": 14,
  "ATR_PERIOD": 14,
  
  "XGB_MAX_DEPTH": 6,
  "XGB_LEARNING_RATE": 0.1,
  "XGB_N_ESTIMATORS": 100,
  "XGB_CV_FOLDS": 3,
  
  "LR_MAX_ITER": 1000,
  "LR_C": 1.0,
  
  "LSTM_UNITS": 128,
  "LSTM_DROPOUT": 0.3,
  "LSTM_SEQ_LEN": 30,
  "LSTM_EPOCHS": 50,
  "LSTM_BATCH_SIZE": 32,
  
  "FEATURE_NAMES": [...],
  "TRAINING_DATE": "2025-01-15",
  "NUM_CLASSES": 3
}
```

---

## 💡 Examples

### Example 1: Training on Gold (XAUUSD)

```bash
# 1. Download XAUUSD 1H data from your broker
# 2. Save as CSV with proper format
# 3. Launch trainer
python "Soluna AI - Trainer.vbs"

# 4. Configure:
#    - Data File: XAUUSD_H1.csv
#    - Timeframe: 1h
#    - ATR Period: 14
#    - Profit Target: 2.0 (for more conservative signals)
#    
# 5. Start training
# 6. Wait for completion
```

### Example 2: EUR/USD Scalping Model

```bash
# For faster timeframes, adjust:
# - Timeframe: 5m or 15m
# - LSTM Sequence: 60 (more history)
# - Profit Target: 1.0 (tighter targets)
# - SMA Short: 5 (faster moving average)
```

### Example 3: Multi-Symbol Strategy

```python
# Train separate models for each symbol
symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

for symbol in symbols:
    # Train model for each symbol
    # Save to separate folders
    # Load different models in server for each symbol
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Reporting Issues

- Use GitHub Issues to report bugs
- Include detailed steps to reproduce
- Attach relevant logs and screenshots

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow existing code style
- Add comments for complex logic
- Test thoroughly before submitting
- Update documentation as needed

### Ideas for Contributions

- Additional technical indicators
- More model architectures (Transformer, GRU)
- Backtesting framework
- Risk management module
- Multi-timeframe analysis
- Sentiment analysis integration

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

**You are free to:**
- ✅ Share — copy and redistribute the material
- ✅ Adapt — remix, transform, and build upon the material

**Under the following terms:**
- 📝 Attribution — You must give appropriate credit
- 🚫 NonCommercial — You may not use for commercial purposes
- 🔄 ShareAlike — If you remix, you must distribute under same license

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- XGBoost Team for the powerful gradient boosting library
- TensorFlow Team for the deep learning framework
- scikit-learn contributors for machine learning tools
- MetaQuotes for MetaTrader platform

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lowcode-np/soluna-ai-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lowcode-np/soluna-ai-model/discussions)
- **Email**: support@soluna-ai.com (if available)

---

## 🗺️ Roadmap

- [ ] Web-based dashboard for monitoring
- [ ] Real-time backtesting engine
- [ ] Multi-symbol parallel processing
- [ ] Advanced risk management features
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Docker containerization
- [ ] Webhook support for trading platforms
- [ ] Mobile app for signal notifications

---

<div align="center">

**Made with ❤️ by the Soluna AI Team**

⭐ Star us on GitHub if you find this project helpful!

[Report Bug](https://github.com/lowcode-np/soluna-ai-model/issues) •
[Request Feature](https://github.com/lowcode-np/soluna-ai-model/issues) •
[Documentation](https://github.com/lowcode-np/soluna-ai-model/wiki)

</div>
