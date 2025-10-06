//+------------------------------------------------------------------+
//|                                         SolunaSignalExample.mq5  |
//|                                  Example EA using Soluna Signal  |
//|                                  File-based version (No WebReq)  |
//+------------------------------------------------------------------+
#property copyright "Soluna AI"
#property version   "2.00"
#property strict

#include <SolunaSignalClient.mqh>

//--- Input parameters
input int      CandleCount = 500;            // Number of candles to send
input int      SignalInterval = 60;          // Check signal every N seconds
input bool     EnableTrading = false;        // Enable automatic trading
input double   LotSize = 0.01;               // Lot size for trading

//--- Global variables
CSolunaSignalClient g_client;
datetime g_last_check = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize client (File-based - no server config needed)
   g_client.SetMinCandles(300);
   g_client.SetTimeout(30);  // seconds
   
   // Check if file system is ready
   Print("Checking file system...");
   if(g_client.CheckHealth())
   {
      Print("✅ File system OK - Python bridge ready!");
      Print("📂 Request: C:\\MT4Bridge\\requests\\");
      Print("📂 Response: C:\\MT4Bridge\\responses\\");
   }
   else
   {
      Print("❌ Failed: ", g_client.GetLastError());
      Print("⚠️  Make sure C:\\MT4Bridge\\ folder exists!");
      Print("⚠️  Make sure Python server is running!");
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Soluna Signal EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if it's time to get new signal
   if(TimeCurrent() - g_last_check < SignalInterval)
      return;
   
   g_last_check = TimeCurrent();
   
   // Get signal
   SolunaSignal signal;
   
   Print("📤 Requesting signal from Soluna AI...");
   
   if(g_client.GetSignal(_Symbol, PERIOD_CURRENT, CandleCount, signal))
   {
      // Signal received successfully
      PrintSignal(signal);
      
      // Execute trade if enabled
      if(EnableTrading)
      {
         ExecuteTrade(signal);
      }
   }
   else
   {
      Print("❌ Failed to get signal: ", g_client.GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Print signal information                                         |
//+------------------------------------------------------------------+
void PrintSignal(SolunaSignal &signal)
{
   Print("========================================");
   Print("📡 Soluna AI Signal Received");
   Print("========================================");
   Print("Timestamp:   ", signal.timestamp);
   Print("Signal:      ", signal.signal);
   Print("Confidence:  ", signal.confidence);
   Print("Price:       ", DoubleToString(signal.price, _Digits));
   Print("----------------------------------------");
   Print("Model Votes:");
   Print("  XGBoost:   ", signal.xgb_signal);
   Print("  Logistic:  ", signal.lr_signal);
   Print("  LSTM:      ", signal.lstm_signal);
   Print("========================================");
}

//+------------------------------------------------------------------+
//| Execute trade based on signal                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(SolunaSignal &signal)
{
   // Check if we already have an open position
   if(PositionSelect(_Symbol))
   {
      Print("⚠️  Position already open, skipping trade");
      return;
   }
   
   // Execute based on signal
   if(signal.signal == "BUY")
   {
      OpenBuy(signal);
   }
   else if(signal.signal == "SELL")
   {
      OpenSell(signal);
   }
   else
   {
      Print("ℹ️  Signal is NEUTRAL, no trade executed");
   }
}

//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
void OpenBuy(SolunaSignal &signal)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation = 10;
   request.magic = 123456;
   request.comment = "Soluna AI BUY - " + signal.confidence;
   
   if(OrderSend(request, result))
   {
      Print("✅ BUY order placed successfully! Ticket: ", result.order);
   }
   else
   {
      Print("❌ BUY order failed: ", result.comment);
   }
}

//+------------------------------------------------------------------+
//| Open Sell position                                               |
//+------------------------------------------------------------------+
void OpenSell(SolunaSignal &signal)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.deviation = 10;
   request.magic = 123456;
   request.comment = "Soluna AI SELL - " + signal.confidence;
   
   if(OrderSend(request, result))
   {
      Print("✅ SELL order placed successfully! Ticket: ", result.order);
   }
   else
   {
      Print("❌ SELL order failed: ", result.comment);
   }
}