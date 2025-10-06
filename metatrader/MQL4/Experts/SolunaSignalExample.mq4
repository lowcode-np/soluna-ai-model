//+------------------------------------------------------------------+
//|                                         SolunaSignalExample.mq4  |
//|                              File-Based - No WebRequest Needed   |
//+------------------------------------------------------------------+
#property copyright "Soluna AI"
#property version   "4.00"
#property strict

#include <SolunaSignalClient.mqh>

//--- Input parameters
input int      CandleCount            = 500;             // Number of candles
input ENUM_TIMEFRAMES  SignalInterval = PERIOD_CURRENT;  // Check every
input bool     EnableTrading          = false;           // Enable auto-trading
input double   LotSize                = 0.01;            // Lot size

//--- Global variables
CSolunaSignalClient g_client;
datetime g_last_check = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_client.SetMinCandles(300);
   g_client.SetTimeout(30);
   
   Print("========================================");
   Print("Soluna AI - File-Based Mode");
   Print("========================================");
   Print("No WebRequest configuration needed!");
   Print("Files stored in Common/Files");
   Print("");
   
   if(g_client.CheckHealth())
   {
      Print("File system OK");
      Print("Make sure server.py is running");
      Print("Ready to receive signals");
   }
   else
   {
      Print("ERROR: ", g_client.GetLastError());
      Print("Check Common/Files accessibility");
   }
   
   Print("========================================");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   if(TimeCurrent() - g_last_check < SignalInterval*60)
      return;
   
   g_last_check = TimeCurrent();
   
   SolunaSignal signal;
   
   Print("Requesting signal...");
   
   if(g_client.GetSignal(_Symbol, PERIOD_CURRENT, CandleCount, signal))
   {
      PrintSignal(signal);
      
      if(EnableTrading)
      {
         ExecuteTrade(signal);
      }
   }
   else
   {
      Print("Failed: ", g_client.GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Print signal                                                     |
//+------------------------------------------------------------------+
void PrintSignal(SolunaSignal &signal)
{
   Print("========================================");
   Print("SIGNAL RECEIVED");
   Print("========================================");
   Print("Time:       ", signal.timestamp);
   Print("Signal:     ", signal.signal);
   Print("Confidence: ", signal.confidence);
   Print("Price:      ", DoubleToString(signal.price, _Digits));
   Print("----------------------------------------");
   Print("XGBoost:    ", signal.xgb_signal);
   Print("Logistic:   ", signal.lr_signal);
   Print("LSTM:       ", signal.lstm_signal);
   Print("========================================");
}

//+------------------------------------------------------------------+
//| Execute trade                                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(SolunaSignal &signal)
{
   if(OrdersTotal() > 0)
   {
      for(int i = 0; i < OrdersTotal(); i++)
      {
         if(OrderSelect(i, SELECT_BY_POS) && OrderSymbol() == _Symbol)
         {
            Print("Order exists, skipping");
            return;
         }
      }
   }
   
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
      Print("NEUTRAL signal, no trade");
   }
}

//+------------------------------------------------------------------+
//| Open Buy                                                         |
//+------------------------------------------------------------------+
void OpenBuy(SolunaSignal &signal)
{
   double ask = MarketInfo(_Symbol, MODE_ASK);
   int ticket = OrderSend(_Symbol, OP_BUY, LotSize, ask, 10, 0, 0, 
                         "Soluna " + signal.confidence, 123456, 0, clrGreen);
   
   if(ticket > 0)
   {
      Print("BUY order: ", ticket);
   }
   else
   {
      Print("BUY failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Open Sell                                                        |
//+------------------------------------------------------------------+
void OpenSell(SolunaSignal &signal)
{
   double bid = MarketInfo(_Symbol, MODE_BID);
   int ticket = OrderSend(_Symbol, OP_SELL, LotSize, bid, 10, 0, 0,
                         "Soluna " + signal.confidence, 123456, 0, clrRed);
   
   if(ticket > 0)
   {
      Print("SELL order: ", ticket);
   }
   else
   {
      Print("SELL failed: ", GetLastError());
   }
}