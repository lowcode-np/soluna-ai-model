//+------------------------------------------------------------------+
//|                                          SolunaSignalClient.mqh  |
//|                                     Soluna AI Signal Integration |
//|                         Compatible with both MT4 and MT5         |
//+------------------------------------------------------------------+
#property copyright "Soluna AI"
#property link      "https://soluna-ai.com"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Signal Response Structure                                        |
//+------------------------------------------------------------------+
struct SolunaSignal
{
   string            timestamp;        // Signal timestamp
   string            signal;           // BUY, SELL, NEUTRAL
   string            xgb_signal;       // XGBoost model signal
   string            lr_signal;        // Logistic Regression signal
   string            lstm_signal;      // LSTM model signal
   string            confidence;       // Confidence level (e.g., "67%")
   double            price;            // Current price
   int               rsi_period;       // RSI period used in config
   string            timeframe;        // Timeframe from config
   bool              is_valid;         // Whether signal is valid
   string            error_message;    // Error message if any
};

//+------------------------------------------------------------------+
//| Candle Data Structure                                            |
//+------------------------------------------------------------------+
struct CandleData
{
   datetime          time;
   double            open;
   double            high;
   double            low;
   double            close;
   long              volume;
};

//+------------------------------------------------------------------+
//| Soluna Signal Client Class                                       |
//+------------------------------------------------------------------+
class CSolunaSignalClient
{
private:
   string            m_server_url;     // Server URL
   string            m_host;           // Server host
   int               m_port;           // Server port
   int               m_timeout;        // Request timeout in ms
   int               m_min_candles;    // Minimum candles required
   
   // Helper functions
   string            CandlesToJSON(CandleData &candles[]);
   bool              ParseSignalResponse(string response, SolunaSignal &signal);
   string            TimeToString(datetime time);
   string            DoubleToStr(double value, int digits);
   
public:
                     CSolunaSignalClient();
                     CSolunaSignalClient(string host, int port);
                    ~CSolunaSignalClient();
   
   // Configuration
   void              SetServer(string host, int port);
   void              SetTimeout(int timeout_ms);
   void              SetMinCandles(int min_candles);
   
   // Main functions
   bool              CheckHealth();
   bool              GetSignal(string symbol, ENUM_TIMEFRAMES timeframe, int candle_count, SolunaSignal &signal);
   SolunaSignal      GetSignalSimple(string symbol, ENUM_TIMEFRAMES timeframe, int candle_count);
   
   // Utility functions
   string            GetLastError();
   bool              IsServerOnline();
   
private:
   string            m_last_error;
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CSolunaSignalClient::CSolunaSignalClient()
{
   m_host = "127.0.0.1";
   m_port = 5000;
   m_timeout = 30000;  // 30 seconds
   m_min_candles = 300;
   m_server_url = "http://" + m_host + ":" + IntegerToString(m_port);
   m_last_error = "";
}

//+------------------------------------------------------------------+
//| Constructor with parameters                                       |
//+------------------------------------------------------------------+
CSolunaSignalClient::CSolunaSignalClient(string host, int port)
{
   m_host = host;
   m_port = port;
   m_timeout = 30000;
   m_min_candles = 300;
   m_server_url = "http://" + m_host + ":" + IntegerToString(m_port);
   m_last_error = "";
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CSolunaSignalClient::~CSolunaSignalClient()
{
}

//+------------------------------------------------------------------+
//| Set server configuration                                          |
//+------------------------------------------------------------------+
void CSolunaSignalClient::SetServer(string host, int port)
{
   m_host = host;
   m_port = port;
   m_server_url = "http://" + m_host + ":" + IntegerToString(m_port);
}

//+------------------------------------------------------------------+
//| Set request timeout                                              |
//+------------------------------------------------------------------+
void CSolunaSignalClient::SetTimeout(int timeout_ms)
{
   m_timeout = timeout_ms;
}

//+------------------------------------------------------------------+
//| Set minimum candles                                              |
//+------------------------------------------------------------------+
void CSolunaSignalClient::SetMinCandles(int min_candles)
{
   m_min_candles = min_candles;
}

//+------------------------------------------------------------------+
//| Check server health                                              |
//+------------------------------------------------------------------+
bool CSolunaSignalClient::CheckHealth()
{
   string url = m_server_url + "/health";
   string headers = "Content-Type: application/json\r\n";
   char data[], result[];
   string result_headers;
   
   int res = WebRequest("GET", url, headers, m_timeout, data, result, result_headers);
   
   if(res == -1)
   {
      m_last_error = "WebRequest error: " + IntegerToString(GetLastError());
      return false;
   }
   
   if(res != 200)
   {
      m_last_error = "HTTP error: " + IntegerToString(res);
      return false;
   }
   
   string response = CharArrayToString(result);
   
   // Check if response contains "running" and "true"
   if(StringFind(response, "running") >= 0 && StringFind(response, "true") >= 0)
   {
      m_last_error = "";
      return true;
   }
   
   m_last_error = "Server not ready";
   return false;
}

//+------------------------------------------------------------------+
//| Get trading signal                                               |
//+------------------------------------------------------------------+
bool CSolunaSignalClient::GetSignal(string symbol, ENUM_TIMEFRAMES timeframe, int candle_count, SolunaSignal &signal)
{
   // Initialize signal
   signal.is_valid = false;
   signal.error_message = "";
   
   // Validate candle count
   if(candle_count < m_min_candles)
   {
      m_last_error = "Need at least " + IntegerToString(m_min_candles) + " candles";
      signal.error_message = m_last_error;
      return false;
   }
   
   // Prepare candle data array
   CandleData candles[];
   ArrayResize(candles, candle_count);
   
   // Fetch historical data
   for(int i = 0; i < candle_count; i++)
   {
      #ifdef __MQL5__
         datetime time_arr[];
         double open_arr[], high_arr[], low_arr[], close_arr[];
         long volume_arr[];
         
         if(CopyTime(symbol, timeframe, i, 1, time_arr) <= 0 ||
            CopyOpen(symbol, timeframe, i, 1, open_arr) <= 0 ||
            CopyHigh(symbol, timeframe, i, 1, high_arr) <= 0 ||
            CopyLow(symbol, timeframe, i, 1, low_arr) <= 0 ||
            CopyClose(symbol, timeframe, i, 1, close_arr) <= 0 ||
            CopyTickVolume(symbol, timeframe, i, 1, volume_arr) <= 0)
         {
            m_last_error = "Failed to copy historical data at index " + IntegerToString(i);
            signal.error_message = m_last_error;
            return false;
         }
         
         candles[candle_count - 1 - i].time = time_arr[0];
         candles[candle_count - 1 - i].open = open_arr[0];
         candles[candle_count - 1 - i].high = high_arr[0];
         candles[candle_count - 1 - i].low = low_arr[0];
         candles[candle_count - 1 - i].close = close_arr[0];
         candles[candle_count - 1 - i].volume = volume_arr[0];
      #else
         int shift = candle_count - 1 - i;
         candles[candle_count - 1 - i].time = iTime(symbol, timeframe, shift);
         candles[candle_count - 1 - i].open = iOpen(symbol, timeframe, shift);
         candles[candle_count - 1 - i].high = iHigh(symbol, timeframe, shift);
         candles[candle_count - 1 - i].low = iLow(symbol, timeframe, shift);
         candles[candle_count - 1 - i].close = iClose(symbol, timeframe, shift);
         candles[candle_count - 1 - i].volume = iVolume(symbol, timeframe, shift);
      #endif
   }
   
   // Convert to JSON
   string json_data = CandlesToJSON(candles);
   
   // Prepare HTTP request
   string url = m_server_url + "/signal";
   string headers = "Content-Type: application/json\r\n";
   char post_data[], result[];
   string result_headers;
   
   StringToCharArray(json_data, post_data, 0, StringLen(json_data));
   
   // Send request
   int res = WebRequest("POST", url, headers, m_timeout, post_data, result, result_headers);
   
   if(res == -1)
   {
      m_last_error = "WebRequest error: " + IntegerToString(GetLastError());
      signal.error_message = m_last_error;
      return false;
   }
   
   if(res != 200)
   {
      m_last_error = "HTTP error: " + IntegerToString(res);
      signal.error_message = m_last_error;
      return false;
   }
   
   // Parse response
   string response = CharArrayToString(result);
   return ParseSignalResponse(response, signal);
}

//+------------------------------------------------------------------+
//| Get signal (simple version)                                      |
//+------------------------------------------------------------------+
SolunaSignal CSolunaSignalClient::GetSignalSimple(string symbol, ENUM_TIMEFRAMES timeframe, int candle_count)
{
   SolunaSignal signal;
   GetSignal(symbol, timeframe, candle_count, signal);
   return signal;
}

//+------------------------------------------------------------------+
//| Convert candles to JSON format                                   |
//+------------------------------------------------------------------+
string CSolunaSignalClient::CandlesToJSON(CandleData &candles[])
{
   string json = "{\"candles\":[";
   
   int total = ArraySize(candles);
   for(int i = 0; i < total; i++)
   {
      if(i > 0) json += ",";
      
      json += "{";
      json += "\"time\":\"" + TimeToString(candles[i].time) + "\",";
      json += "\"open\":" + DoubleToStr(candles[i].open, 2) + ",";
      json += "\"high\":" + DoubleToStr(candles[i].high, 2) + ",";
      json += "\"low\":" + DoubleToStr(candles[i].low, 2) + ",";
      json += "\"close\":" + DoubleToStr(candles[i].close, 2) + ",";
      json += "\"volume\":" + IntegerToString(candles[i].volume);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Parse signal response                                            |
//+------------------------------------------------------------------+
bool CSolunaSignalClient::ParseSignalResponse(string response, SolunaSignal &signal)
{
   // Simple JSON parsing (basic implementation)
   signal.is_valid = false;
   
   // Check for error
   if(StringFind(response, "\"error\"") >= 0)
   {
      int start = StringFind(response, "\"error\"");
      int colon = StringFind(response, ":", start);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      
      if(quote1 >= 0 && quote2 > quote1)
      {
         signal.error_message = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
         m_last_error = signal.error_message;
      }
      return false;
   }
   
   // Parse signal
   int pos = StringFind(response, "\"signal\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      signal.signal = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
   }
   
   // Parse confidence
   pos = StringFind(response, "\"confidence\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      signal.confidence = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
   }
   
   // Parse price
   pos = StringFind(response, "\"price\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int comma = StringFind(response, ",", colon);
      string price_str = StringSubstr(response, colon + 1, comma - colon - 1);
      signal.price = StringToDouble(price_str);
   }
   
   // Parse XGB signal
   pos = StringFind(response, "\"xgb\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      signal.xgb_signal = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
   }
   
   // Parse LR signal
   pos = StringFind(response, "\"lr\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      signal.lr_signal = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
   }
   
   // Parse LSTM signal
   pos = StringFind(response, "\"lstm\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      signal.lstm_signal = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
   }
   
   // Parse timestamp
   pos = StringFind(response, "\"timestamp\"");
   if(pos >= 0)
   {
      int colon = StringFind(response, ":", pos);
      int quote1 = StringFind(response, "\"", colon);
      int quote2 = StringFind(response, "\"", quote1 + 1);
      signal.timestamp = StringSubstr(response, quote1 + 1, quote2 - quote1 - 1);
   }
   
   signal.is_valid = true;
   m_last_error = "";
   return true;
}

//+------------------------------------------------------------------+
//| Convert datetime to string                                       |
//+------------------------------------------------------------------+
string CSolunaSignalClient::TimeToString(datetime time)
{
   MqlDateTime dt;
   TimeToStruct(time, dt);
   
   string result = IntegerToString(dt.year) + "-" +
                   StringFormat("%02d", dt.mon) + "-" +
                   StringFormat("%02d", dt.day) + " " +
                   StringFormat("%02d", dt.hour) + ":" +
                   StringFormat("%02d", dt.min);
   
   return result;
}

//+------------------------------------------------------------------+
//| Convert double to string                                         |
//+------------------------------------------------------------------+
string CSolunaSignalClient::DoubleToStr(double value, int digits)
{
   return DoubleToString(value, digits);
}

//+------------------------------------------------------------------+
//| Get last error                                                   |
//+------------------------------------------------------------------+
string CSolunaSignalClient::GetLastError()
{
   return m_last_error;
}

//+------------------------------------------------------------------+
//| Check if server is online                                        |
//+------------------------------------------------------------------+
bool CSolunaSignalClient::IsServerOnline()
{
   return CheckHealth();
}