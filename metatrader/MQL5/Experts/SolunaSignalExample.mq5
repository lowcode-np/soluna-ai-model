//+------------------------------------------------------------------+
//|                                            SolunaSignalEA.mq5    |
//|                                          Example Implementation  |
//+------------------------------------------------------------------+
#property copyright "Soluna AI"
#property version   "5.00"
#property strict

#include <SolunaSignalClient.mqh>

input int InpCandleCount = 300;       // Number of candles to analyze
input ENUM_TIMEFRAMES  SignalInterval = PERIOD_CURRENT;  // Check every
input int InpTimeout = 35;            // Timeout seconds
input bool InpCleanupOnInit = true;   // Cleanup old files on init

CSolunaSignalClient g_client;
datetime g_last_request_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_client.SetTimeout(InpTimeout);
   g_client.SetMinCandles(InpCandleCount);
   g_client.SetMaxRetries(3);
   
   // Health check
   if(!g_client.CheckHealth())
   {
      Alert("Health check failed! Check Common/Files permissions");
      Print("Error: ", g_client.GetLastError());
      return INIT_FAILED;
   }
   
   // Cleanup old files
   if(InpCleanupOnInit)
   {
      Comment("Cleaning up old files...");
      g_client.CleanupOldFiles();
   }
   
   Comment("Terminal ID: ", g_client.GetTerminalID());
   Comment("Ready to receive signals");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(TimeCurrent() - g_last_request_time < SignalInterval * 60)
      return;
   
   SolunaSignal signal;
   
   Comment("Requesting signal...");
   
   if(g_client.GetSignal(_Symbol, _Period, InpCandleCount, signal))
   {
      if(signal.is_valid)
      {
         Comment("========== SIGNAL RECEIVED ==========\n",
               "Signal: ", signal.signal, "\n",
               "Confidence: ", signal.confidence, "\n",
               "Price: ", DoubleToString(signal.price, _Digits), "\n",
               "XGB: ", signal.xgb_signal, "\n",
               "LR: ", signal.lr_signal, "\n",
               "LSTM: ", signal.lstm_signal, "\n",
               "Timestamp: ", signal.timestamp, "\n",
               "Server Time: ", signal.server_time, "\n",
               "=====================================");
         
         g_last_request_time = TimeCurrent();
      }
      else
      {
         Comment("Invalid signal received");
      }
   }
   else
   {
      Comment("Failed to get signal: ", g_client.GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("EA stopped. Reason: ", reason);
   Comment();
   g_client.CleanupOldFiles();
}