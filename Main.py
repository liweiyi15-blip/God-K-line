import discord
from discord import app_commands
from discord.ext import tasks
import json
import os
from datetime import datetime, time, timedelta
import asyncio
import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv
from collections import defaultdict
import aiohttp
import io
import matplotlib

# [日志配置]
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# --- 强制使用非交互式后端 ---
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

# --- 加载环境变量 ---
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
FMP_API_KEY = os.getenv("FMP_API_KEY")

try:
    ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID"))
except (TypeError, ValueError):
    ALERT_CHANNEL_ID = 0
    logging.warning("ALERT_CHANNEL_ID not set or invalid.")

# --- 全局配置 ---
MARKET_TIMEZONE = pytz.timezone('America/New_York')

SETTINGS_FILE = "/app/data/settings.json"
if not os.path.exists("/app/data"):
    SETTINGS_FILE = "settings.json"

# [修改] 删除盘前，只保留盘中扫描时间
TIME_MARKET_OPEN = time(9, 30)
TIME_MARKET_SCAN_START = time(10, 0) # 10点才开始报
TIME_MARKET_CLOSE = time(16, 0)

# --- 核心策略配置 ---
CONFIG = {
    "filter": {
        "max_60d_gain": 3.0,
        "max_rsi": 82,
        "max_bias_50": 0.45,
        "max_upper_shadow": 0.4,
        "max_day_change": 0.15,
        "min_vol_ratio": 1.2, 
        "min_bb_squeeze_width": 0.08, 
        "min_adx_for_squeeze": 15
    },
    "pattern": {
        "pivot_window": 5 
    },
    "system": {
        "cooldown_days": 3,
        "max_charts_per_scan": 5,
        "history_days": 400
    },
    "SCORE": { 
        "MIN_ALERT_SCORE": 70, 
        "WEIGHTS": {
            "GOD_TIER_NX": 40,    
            "PATTERN_BREAK": 30,  
            "BB_SQUEEZE": 15,         
            "STRONG_ADX": 20,     
            "ADX_ACTIVATION": 15, 
            "HEAVY_VOLUME": 10,   
            "KDJ_REBOUND": 8,        
            "MACD_ZERO_CROSS": 8, 
            "NX_BREAKOUT": 7,        
            "CANDLE_PATTERN": 5,
            "MACD_DIVERGE": 5,       
            "CAPITULATION": 12       
        },
        "EMOJI": { 
            100: "TOP", 90: "HIGH", 80: "MID", 70: "LOW", 60: "TEST"
        }
    }
}

# --- 静态股票池 (已剔除 ANSS/WBA, 加入 SNPS) ---
STOCK_POOLS = {
    "NASDAQ_100": [
        "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "ADBE", 
        "COST", "PEP", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "CMCSA", "AZN", "QCOM", 
        "TXN", "AMGN", "HON", "INTU", "SBUX", "GILD", "BKNG", "DIOD", "MDLZ", "ISRG", 
        "REGN", "LRCX", "VRTX", "ADP", "ADI", "MELI", "KLAC", "PANW", "SNPS", "CDNS", 
        "CHTR", "MAR", "CSX", "ORLY", "MNST", "NXPI", "CTAS", "FTNT", "WDAY", "DXCM", 
        "PCAR", "KDP", "PAYX", "IDXX", "AEP", "LULU", "EXC", "BIIB", "ADSK", "XEL", 
        "ROST", "MCHP", "CPRT", "DLTR", "EA", "FAST", "CTSH", "VRSK", "CSGP", "ODFL", 
        "EBAY", "ILMN", "GFS", "ALGN", "TEAM", "CDW", "WBD", "SIRI", "ZM", "ENPH", 
        "JD", "PDD", "LCID", "RIVN", "ZS", "DDOG", "CRWD", "TTD", "BKR", "CEG", "GEHC", 
        "ON", "FANG"
    ],
    "GOD_TIER": [
        "NVDA", "AMD", "TSM", "SMCI", "AVGO", "ARM", "PLTR", "AI", "PATH", "BABA", 
        "PDD", "BIDU", "NIO", "LI", "XPEV", "COIN", "MARA", "MSTR"
    ]
}

settings = {}

# --- 辅助函数 ---
def load_settings():
    global settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            settings = {"users": {}, "signal_history": {}}
            save_settings()
    except Exception as e:
        logging.error(f"Error loading settings: {e}")
        settings = {"users": {}, "signal_history": {}}

def save_settings():
    try:
        dir_name = os.path.dirname(SETTINGS_FILE)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving settings: {e}")

def get_user_data(user_id):
    uid_str = str(user_id)
    if "users" not in settings: settings["users"] = {}
    if uid_str not in settings["users"]:
        settings["users"][uid_str] = {"stocks": [], "daily_status": {}}
    return settings["users"][uid_str]

# --- 核心逻辑 (指标计算) ---
def calculate_nx_indicators(df):
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
      
    df = df[df['close'] > 0]
      
    # 1. Nx 均线
    df['Nx_Blue_UP'] = df['high'].ewm(span=24, adjust=False).mean()
    df['Nx_Blue_DW'] = df['low'].ewm(span=23, adjust=False).mean()
    df['Nx_Yellow_UP'] = df['high'].ewm(span=89, adjust=False).mean()
    df['Nx_Yellow_DW'] = df['low'].ewm(span=90, adjust=False).mean()
      
    # 2. MACD
    price_col = 'close'
    exp12 = df[price_col].ewm(span=12, adjust=False).mean()
    exp26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
      
    # 3. RSI 
    delta = df[price_col].diff()
    gain = (delta.clip(lower=0)).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
      
    rs = gain / loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
      
    # 4. Volume MA
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
      
    # 5. ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 6. BB
    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']

    # 7. KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv_denom = (high_max - low_min).replace(0, 1e-9)
    df['RSV'] = (df['close'] - low_min) / rsv_denom * 100
    df['K'] = df['RSV'].ewm(com=2).mean() 
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 8. ADX / DMI 系统
    alpha = 1/14
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['mdm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
      
    df['TR_s'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['PDM_s'] = df['pdm'].ewm(alpha=alpha, adjust=False).mean()
    df['MDM_s'] = df['mdm'].ewm(alpha=alpha, adjust=False).mean()
      
    tr_s_safe = df['TR_s'].replace(0, 1e-9)
    df['PDI'] = 100 * (df['PDM_s'] / tr_s_safe)
    df['MDI'] = 100 * (df['MDM_s'] / tr_s_safe)
      
    dx_denom = (df['PDI'] + df['MDI']).replace(0, 1e-9)
    df['DX'] = 100 * abs(df['PDI'] - df['MDI']) / dx_denom
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()

    # 9. 乖离率 (Bias)
    df['MA50'] = df['close'].rolling(50).mean()
    ma50_safe = df['MA50'].replace(0, np.nan) 
    df['BIAS_50'] = (df['close'] - ma50_safe) / ma50_safe

    # 10. K线形态特征
    candle_range = (df['high'] - df['low']).replace(0, 1e-9)
    upper_shadow = np.where(df['close'] >= df['open'], df['high'] - df['close'], df['high'] - df['open'])
    df['Upper_Shadow_Ratio'] = upper_shadow / candle_range

    return df

def process_dataframe_sync(hist_data):
    if not hist_data: return None
    df = pd.DataFrame(hist_data)
    if 'date' not in df.columns: return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index(ascending=True)
    return calculate_nx_indicators(df)

def merge_and_recalc_sync(df, quote):
    if df is None or quote is None: return df
    try:
        quote_time = pd.to_datetime(quote['timestamp'], unit='s').tz_localize('UTC').tz_convert(MARKET_TIMEZONE)
        quote_date = quote_time.normalize().tz_localize(None) 
          
        last_idx = df.index[-1]
        last_date = pd.to_datetime(last_idx).normalize()
        if last_date.tzinfo is not None:
             last_date = last_date.tz_localize(None)

        current_price = quote['price']
        safe_high = max(quote['dayHigh'], current_price, quote['open'])
        safe_low = min(quote['dayLow'], current_price, quote['open'])

        new_row = {
            'open': quote['open'],
            'high': safe_high,
            'low': safe_low,
            'close': current_price,
            'volume': quote['volume'],
            'date': quote_date 
        }
          
        df_mod = df.copy()
          
        if last_date == quote_date:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col == 'high':
                    df_mod.at[last_idx, col] = max(df_mod.at[last_idx, col], new_row[col])
                elif col == 'low':
                    df_mod.at[last_idx, col] = min(df_mod.at[last_idx, col], new_row[col])
                else:
                    df_mod.at[last_idx, col] = new_row[col]
        elif last_date < quote_date:
            new_df = pd.DataFrame([new_row])
            new_df = new_df.set_index('date')
            df_mod = pd.concat([df_mod, new_df])
          
        if 'marketCap' in quote:
            df_mod.attrs['marketCap'] = quote['marketCap']
            
        return calculate_nx_indicators(df_mod)
        
    except Exception as e:
        logging.error(f"[Merge Error] {e}")
        return df

def _safely_process_fmp_data(data, sym):
    try:
        if isinstance(data, list) and len(data) > 0 and 'date' in data[0] and 'close' in data[0]:
            return process_dataframe_sync(data)
        elif isinstance(data, dict) and 'historical' in data:
            return process_dataframe_sync(data['historical'])
        elif isinstance(data, list) and len(data) > 0 and 'historical' in data[0]:
            for item in data:
                if item.get('symbol') == sym:
                    return process_dataframe_sync(item['historical'])
        return None
    except Exception as e:
        logging.error(f"[Data Process Error] {sym}: {e}")
        return None

async def fetch_historical_batch(symbols: list, days=None):
    if not symbols: return {}
    if days is None: days = CONFIG["system"]["history_days"]
      
    results = {}
    now = datetime.now()
    from_date = (now - timedelta(days=days + 60)).strftime("%Y-%m-%d") 
    to_date = now.strftime("%Y-%m-%d")
      
    connector = aiohttp.TCPConnector(limit=15)
    semaphore = asyncio.Semaphore(15)
      
    headers = {
        "User-Agent": "Mozilla/5.0 (StockBot/1.0)",
        "Accept": "application/json"
    }

    async def fetch_single(session, sym):
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={sym}&from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
        async with semaphore:
            try:
                async with session.get(url, ssl=False) as response:
                    if response.status == 429:
                        logging.warning(f"[429 Rate Limit] {sym}. Retrying in 5s...")
                        await asyncio.sleep(5)
                        response = await session.get(url, ssl=False)
                          
                    if response.status == 200:
                        data = await response.json()
                        df = await asyncio.to_thread(_safely_process_fmp_data, data, sym)
                          
                        if df is not None and not df.empty:
                            results[sym] = df
                        else:
                            logging.warning(f"[数据为空] {sym}")
                    else:
                        logging.error(f"[HTTP 错误] {sym} Status: {response.status}")
            except Exception as e:
                logging.error(f"[异常] {sym}: {e}")

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks_list = [fetch_single(session, sym) for sym in symbols]
        await asyncio.gather(*tasks_list)
    return results

async def fetch_realtime_quotes(symbols: list):
    if not symbols: return {}
    quotes_map = {}
    connector = aiohttp.TCPConnector(limit=20)
    semaphore = asyncio.Semaphore(20)
    headers = {"User-Agent": "StockBot/1.0", "Accept": "application/json"}
      
    async def fetch_single_quote(session, sym):
        url = f"https://financialmodelingprep.com/stable/quote?symbol={sym}&apikey={FMP_API_KEY}"
        async with semaphore:
            try:
                async with session.get(url, ssl=False) as response:
                    if response.status == 429:
                        logging.warning(f"[429 Rate Limit] Quote {sym}. Retrying in 5s...")
                        await asyncio.sleep(5)
                        response = await session.get(url, ssl=False)

                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list):
                            for item in data:
                                s = item.get('symbol')
                                if s: quotes_map[s] = item
                        elif isinstance(data, dict):
                             s = data.get('symbol')
                             if s: quotes_map[s] = data
            except Exception as e:
                logging.error(f"[Quote Exception] {sym}: {e}")

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks_list = [fetch_single_quote(session, sym) for sym in symbols]
        await asyncio.gather(*tasks_list)
    return quotes_map

def find_pivots(df, window=5):
    highs = df['high'].values
    lows = df['low'].values
    dates = df.index
    
    pivots_high = [] 
    pivots_low = []
    
    start_idx = max(0, len(df) - 60)
    
    for i in range(start_idx + window, len(df) - window):
        is_high = True
        is_low = True
        current_high = highs[i]
        current_low = lows[i]
        
        for j in range(1, window + 1):
            if highs[i-j] >= current_high or highs[i+j] > current_high:
                is_high = False
            if lows[i-j] <= current_low or lows[i+j] < current_low:
                is_low = False
        
        if is_high: pivots_high.append((dates[i], current_high, i))
        if is_low: pivots_low.append((dates[i], current_low, i))
            
    return pivots_high, pivots_low

def identify_patterns(df):
    """
    无限延长画线 + 3.5% 新鲜度检查
    """
    if len(df) < 30: return None, [], []
    
    pivots_high, pivots_low = find_pivots(df, window=4)
    
    res_line, sup_line = [], []
    pattern_name = None

    vis_start_idx = max(0, len(df) - 80) 
    curr_idx = len(df) - 1
    
    t_end = df.index[curr_idx]
    t_start = df.index[vis_start_idx]

    if len(pivots_high) >= 2:
        ph1, ph2 = pivots_high[-2], pivots_high[-1]
        x1, y1 = ph1[2], ph1[1]
        x2, y2 = ph2[2], ph2[1]
        
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            
            p_start = m * vis_start_idx + c
            p_end = m * curr_idx + c
            
            res_line = [[(t_start, p_start), (t_end, p_end)]]
            
            if len(pivots_low) >= 2:
                pl1, pl2 = pivots_low[-2], pivots_low[-1]
                lx1, ly1 = pl1[2], pl1[1]
                lx2, ly2 = pl2[2], pl2[1]
                
                if lx2 != lx1:
                    lm = (ly2 - ly1) / (lx2 - lx1)
                    lc = ly1 - lm * lx1
                    
                    lp_start = lm * vis_start_idx + lc
                    lp_end = lm * curr_idx + lc
                    
                    sup_line = [[(t_start, lp_start), (t_end, lp_end)]]
                    
                    curr_price = df['close'].iloc[-1]
                    res_today = m * curr_idx + c
                    
                    if m < 0.005 and (lm > m + 0.01):
                         if curr_price > res_today:
                             # Freshness Check
                             if curr_price < res_today * 1.035:
                                 pattern_name = "形态突破 (刚启动)"
    
    return pattern_name, res_line, sup_line

def detect_candle_patterns(df):
    if len(df) < 5: return []
    patterns = []
    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    curr_body = abs(curr['close'] - curr['open'])
    prev1_body = abs(prev1['close'] - prev1['open'])
    prev2_body = abs(prev2['close'] - prev2['open'])
    
    is_bullish_engulfing = (prev1['close'] < prev1['open']) and \
                           (curr['close'] > curr['open']) and \
                           (curr['open'] < prev1['close']) and \
                           (curr['close'] > prev1['open'])
    if is_bullish_engulfing: patterns.append("Bullish Engulfing (吞没)")
        
    is_morning_star = (prev2['close'] < prev2['open']) and \
                      (prev1_body < prev2_body * 0.3) and \
                      (curr['close'] > curr['open']) and \
                      (curr['close'] > (prev2['open'] + prev2['close'])/2)
    if is_morning_star: patterns.append("Morning Star (早晨之星)")
        
    lower_shadow = min(curr['close'], curr['open']) - curr['low']
    upper_shadow = curr['high'] - max(curr['close'], curr['open'])
    if lower_shadow > 2 * curr_body and upper_shadow < 0.5 * curr_body:
        patterns.append("Hammer (锤子线)")

    return patterns

def get_volume_projection_factor(ny_now, minutes_elapsed):
    TOTAL_MINUTES = 390
    if minutes_elapsed <= 10: return 13.0
    elif minutes_elapsed <= 60: return 13.0 - (13.0 - 8.0) * (minutes_elapsed - 10) / 50
    else: return 8.0 - (8.0 - 4.0) * (minutes_elapsed - 60) / (TOTAL_MINUTES - 60)

# [新增] 计算双线：止损位 (ATR) 和 支撑位 (Structure)
def calculate_risk_levels(df):
    """
    返回 (stop_loss, support)
    Stop Loss: 2.8x ATR (Hard Risk Control)
    Support: Pivot Low (Structural Level)
    """
    curr_close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else curr_close * 0.05
    
    # 1. 计算止损 (Stop Loss) - 回归 2.8x ATR
    stop_loss = curr_close - (2.8 * atr)
    
    # 2. 计算支撑 (Support) - 找前低结构
    _, pivots_low = find_pivots(df, window=5)
    support = stop_loss # 默认 fallback
    
    if pivots_low:
        last_pivot_low = pivots_low[-1][1]
        # 如果前低在现价下方，且不要离得太远(比如跌了50%)，则作为支撑
        if last_pivot_low < curr_close:
             support = last_pivot_low
             
    return stop_loss, support

# --- 核心信号检查函数 ---
def check_signals_sync(df):
    if len(df) < 60: return False, 0, "数据不足", [], []
    
    last_date = df.index[-1].date()
    today_date = datetime.now(MARKET_TIMEZONE).date()
    
    if (today_date - last_date).days > 4:
        return False, 0, f"DATA_STALE: 数据严重滞后 ({last_date})", [], []

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    score = 0
    weights = CONFIG["SCORE"]["WEIGHTS"]
    violations = [] 

    low_60 = df['low'].tail(60).min()
    if curr['close'] > low_60 * CONFIG["filter"]["max_60d_gain"]: 
        violations.append("RISK: 短期涨幅过大")
        
    prev_close_safe = prev['close'] if prev['close'] > 0 else 1.0
    day_gain = (curr['close'] - prev['close']) / prev_close_safe

    if abs(day_gain) > CONFIG["filter"]["max_day_change"]: 
        violations.append("RISK: 单日波动过大")
        
    if curr['RSI'] > CONFIG["filter"]["max_rsi"]: 
        violations.append("RISK: RSI严重超买")
      
    if curr['BIAS_50'] > CONFIG["filter"]["max_bias_50"]:
        violations.append("RISK: 乖离率过大")

    if curr['Upper_Shadow_Ratio'] > CONFIG["filter"]["max_upper_shadow"]:
        violations.append("RISK: 长上影线压力")

    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
    is_open_market = 0 < minutes_elapsed < 390
    
    is_volume_ok = False
    proj_vol_final = curr['volume']
    
    # 只有盘中检查量能，盘前已被外部循环屏蔽
    if is_open_market:
        if minutes_elapsed < 30:
            is_volume_ok = True 
        else:
            proj_factor = get_volume_projection_factor(ny_now, max(minutes_elapsed, 1))
            trend_modifier = 1 - (min(curr['ADX'], 40) - 20) / 200 
            proj_factor *= max(0.8, trend_modifier)
            proj_vol_final = curr['volume'] * proj_factor
            
            if proj_vol_final >= curr['Vol_MA20'] * CONFIG["filter"]["min_vol_ratio"]:
                is_volume_ok = True
    else:
        # 盘后复盘使用
        if curr['volume'] >= curr['Vol_MA20'] * CONFIG["filter"]["min_vol_ratio"]:
            is_volume_ok = True
            
    if not is_volume_ok:
        violations.append("FILTER: 量能不足 (死鱼股)")

    if proj_vol_final > curr['Vol_MA20'] * 2.0:
        score += weights["HEAVY_VOLUME"]

    candle_patterns = detect_candle_patterns(df)
    if candle_patterns:
        triggers.append(f"K线: {', '.join(candle_patterns)}")
        score += weights["CANDLE_PATTERN"]

    bb_min_width = CONFIG["filter"]["min_bb_squeeze_width"]
    bb_open_width = bb_min_width * 1.05 
      
    if prev['BB_Width'] < bb_min_width: 
        if curr['BB_Width'] > bb_open_width and curr['close'] > curr['BB_Mid']: 
            if curr['ADX'] > CONFIG["filter"]["min_adx_for_squeeze"] and curr['PDI'] > curr['MDI']:
                triggers.append(f"BB Squeeze: 紧缩结束+开口向上")
                score += weights["BB_SQUEEZE"]

    is_strong_trend = curr['ADX'] > 25 and curr['PDI'] > curr['MDI']
    is_adx_rising = curr['ADX'] > prev['ADX']
    
    if is_strong_trend and is_adx_rising:
        score += weights["STRONG_ADX"]
        
    recent_adx_min = df['ADX'].iloc[-10:-1].min()
    adx_activating = (recent_adx_min < 20) and \
                     (df['ADX'].iloc[-1] > df['ADX'].iloc[-2] > df['ADX'].iloc[-3])
                     
    if adx_activating:
        triggers.append(f"趋势激活: 盘整结束 ADX拐头")
        score += weights["ADX_ACTIVATION"]

    had_breakout = (df['close'].tail(10) > df['Nx_Blue_UP'].tail(10)).any()
    on_support = (curr['low'] >= curr['Nx_Blue_DW'] * 0.99) and (curr['close'] > curr['Nx_Blue_DW'])
    
    if is_strong_trend and had_breakout and on_support and (curr['close'] > curr['open']):
        triggers.append("Nx 结构: 蓝梯回踩确认")
        score += weights["GOD_TIER_NX"] 

    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name:
        triggers.append(pattern_name)
        score += weights["PATTERN_BREAK"]

    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        if curr['PDI'] > curr['MDI']:
            triggers.append(f"Nx 突破: 站上蓝梯")
            score += weights["NX_BREAKOUT"]
      
    is_zero_cross = prev['DIF'] < 0 and curr['DIF'] > 0 and curr['DIF'] > curr['DEA']
    if is_zero_cross:
        triggers.append(f"MACD 金叉")
        score += weights["MACD_ZERO_CROSS"]

    if prev['J'] < 0 and curr['J'] > 0 and curr['K'] > curr['D']:
        triggers.append(f"KDJ 反击")
        score += weights["KDJ_REBOUND"]
      
    price_low_20 = df['close'].tail(20).min()
    price_is_low = curr['close'] <= price_low_20 * 1.02
    macd_low_20 = df['MACD'].tail(20).min()
    if price_is_low and curr['MACD'] < 0:
        if curr['MACD'] > macd_low_20 * 0.8 and curr['DIF'] > df['DIF'].tail(20).min():
             triggers.append(f"MACD 底背离")
             score += weights["MACD_DIVERGE"]

    pinbar_ratio = (curr['close'] - curr['low']) / (curr['high'] - curr['low'] + 1e-9)
    market_cap = df.attrs.get('marketCap', float('inf')) 
      
    if curr['low'] < curr['BB_Low']:
        if proj_vol_final > curr['Vol_MA20'] * 2.5:
            if pinbar_ratio > 0.5 and market_cap < 5_000_000_000:
                triggers.append(f"抛售高潮")
                score += weights["CAPITULATION"]

    is_triggered = (score >= CONFIG["SCORE"]["MIN_ALERT_SCORE"]) and (len(violations) == 0)
    
    final_reason_parts = triggers + violations
    final_reason = "\n".join(final_reason_parts) if final_reason_parts else "无明显信号"

    return is_triggered, score, final_reason, res_line, sup_line

async def check_signals(df):
    return await asyncio.to_thread(check_signals_sync, df)

# [修改] 接收 stop_price 和 support_price
def _generate_chart_sync(df, ticker, res_line=[], sup_line=[], stop_price=None, support_price=None):
    buf = io.BytesIO()
      
    last_close = df['close'].iloc[-1]
    
    # 默认值保护
    if stop_price is None: stop_price = last_close * 0.95
    if support_price is None: support_price = last_close * 0.90

    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    my_style = mpf.make_mpf_style(base_mpl_style="ggplot", marketcolors=s, gridstyle=":")
    plot_df = df.tail(80)
      
    stop_line_data = [stop_price] * len(plot_df)
    supp_line_data = [support_price] * len(plot_df)

    add_plots = [
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        
        # [修改] 双线绘制
        mpf.make_addplot(stop_line_data, color='red', linestyle='--', width=1.2),   # 止损线 (Red)
        mpf.make_addplot(supp_line_data, color='green', linestyle=':', width=1.2),  # 支撑线 (Green)
        
        mpf.make_addplot(plot_df['MACD'], panel=2, type='bar', color='dimgray', alpha=0.5, ylabel='MACD'),
        mpf.make_addplot(plot_df['DIF'], panel=2, color='orange'),
        mpf.make_addplot(plot_df['DEA'], panel=2, color='blue'),
    ]
      
    kwargs = dict(type='candle', style=my_style, title=f"{ticker} Analysis", ylabel='Price', addplot=add_plots, volume=True, panel_ratios=(6, 2, 2), tight_layout=True, savefig=dict(fname=buf, format='png', bbox_inches='tight', pad_inches=0))
      
    all_lines = []
    
    if res_line: all_lines.extend(res_line)
    if sup_line: all_lines.extend(sup_line)
        
    if all_lines:
        kwargs['alines'] = dict(alines=all_lines, colors='darkgray', linewidths=1.5, linestyle='-.')
      
    try:
        mpf.plot(plot_df, **kwargs)
        buf.seek(0)
    finally:
        plt.close('all')
        
    return buf

async def generate_chart(df, ticker, res_line=[], sup_line=[], stop_price=None, support_price=None):
    return await asyncio.to_thread(_generate_chart_sync, df, ticker, res_line, sup_line, stop_price, support_price)

async def update_stats_data():
    if "signal_history" not in settings: return
    updates_made = False
    symbols_to_check = set()
    history = settings["signal_history"]
    for date_str, tickers_data in history.items():
        try:
            signal_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError: continue
        today = datetime.now().date()
        if signal_date >= today: continue 
        for ticker, data in tickers_data.items():
            need_1d = data.get("ret_1d") is None
            need_5d = data.get("ret_5d") is None and (today - signal_date).days > 5
            need_20d = data.get("ret_20d") is None and (today - signal_date).days > 20
            if need_1d or need_5d or need_20d: symbols_to_check.add(ticker)
    if not symbols_to_check: return
    data_map = await fetch_historical_batch(list(symbols_to_check), days=60)
    for date_str, tickers_data in history.items():
        signal_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        for ticker, data in tickers_data.items():
            if ticker not in data_map: continue
            df = data_map[ticker]
            try:
                after_signal = df[df.index.date > signal_date]
                if after_signal.empty: continue
                signal_price = data['price']
                if signal_price <= 0: continue
                if data.get("ret_1d") is None and len(after_signal) >= 1:
                    price_1d = after_signal.iloc[0]['close']
                    data["ret_1d"] = round(((price_1d - signal_price) / signal_price) * 100, 2)
                    updates_made = True
                if data.get("ret_5d") is None and len(after_signal) >= 5:
                    price_5d = after_signal.iloc[4]['close'] 
                    data["ret_5d"] = round(((price_5d - signal_price) / signal_price) * 100, 2)
                    updates_made = True
                if data.get("ret_20d") is None and len(after_signal) >= 20:
                    price_20d = after_signal.iloc[19]['close'] 
                    data["ret_20d"] = round(((price_20d - signal_price) / signal_price) * 100, 2)
                    updates_made = True
            except: pass
    if updates_made: save_settings()

def get_level_by_score(score): 
    if score >= 100: return CONFIG["SCORE"]["EMOJI"].get(100, "TOP")
    if score >= 90: return CONFIG["SCORE"]["EMOJI"].get(90, "HIGH")
    if score >= 80: return CONFIG["SCORE"]["EMOJI"].get(80, "MID")
    if score >= 70: return CONFIG["SCORE"]["EMOJI"].get(70, "LOW")
    return CONFIG["SCORE"]["EMOJI"].get(60, "TEST") 

# [修改] 增加 support 参数，并显示在 Embed 中
def create_alert_embed(ticker, score, price, reason, stop_loss, support, df, filename):
    level_str = get_level_by_score(score)
    if "RISK" in reason or "FILTER" in reason or "STALE" in reason:
        color = 0x95a5a6 
    else:
        color = 0x00ff00 if score >= 80 else 0x3498db
    
    embed = discord.Embed(title=f"{ticker} 深度分析报告 | 得分 {score}", color=color)
    embed.description = f"**现价:** `${price:.2f}`\n**评级:** {level_str}"
    
    curr = df.iloc[-1]
    
    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
    
    vol_label = "**量比:**"
    vol_ratio = 0.0
    
    if 0 < minutes_elapsed < 390:
        vol_label = "**量比 (预测):**"
        proj_factor = get_volume_projection_factor(ny_now, max(minutes_elapsed, 1))
        projected_vol = curr['volume'] * proj_factor
        vol_ratio = projected_vol / df['Vol_MA20'].iloc[-1]
    else:
        vol_label = "**量比:**"
        vol_ratio = curr['volume'] / df['Vol_MA20'].iloc[-1]
    
    indicator_text = (
        f"**RSI(14):** `{curr['RSI']:.1f}`\n"
        f"**ADX:** `{curr['ADX']:.1f}`\n"
        f"{vol_label} `{vol_ratio:.1f}x`\n" 
        f"**MACD:** `{curr['MACD']:.2f}`\n"
        f"**Bias(50):** `{curr['BIAS_50']*100:.1f}%`"
    )
    embed.add_field(name="技术指标", value=indicator_text, inline=True)
    
    risk_per_trade = 10000 * 0.02
    risk_diff = price - stop_loss # 用止损价计算风险
    shares = int(risk_per_trade / risk_diff) if risk_diff > 0 else 0
    
    risk_text = (
        f"**支撑位:** `${support:.2f}`\n"
        f"**止损价:** `${stop_loss:.2f}`\n"
        f"**建议仓位:** `{shares} 股`\n"
        f"*(基于 $10k/2% 风险)*"
    )
    embed.add_field(name="风险管理", value=risk_text, inline=True)
    
    embed.add_field(name="触发详情", value=f"```{reason}```", inline=False)
    
    embed.set_image(url=f"attachment://{filename}")
    embed.set_footer(text=f"StockBot 智能分析 • {ny_now.strftime('%H:%M:%S')} ET")
    
    return embed

class StockBotClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.alert_channel = None

    async def on_ready(self):
        load_settings()
        logging.info(f'Logged in as {self.user}')
        if ALERT_CHANNEL_ID != 0:
            self.alert_channel = self.get_channel(ALERT_CHANNEL_ID)
            if self.alert_channel is None:
                logging.error(f"Could not find channel with ID {ALERT_CHANNEL_ID}")
        else:
            logging.warning("No ALERT_CHANNEL_ID provided in env.")
            
        if not self.monitor_stocks.is_running():
            self.monitor_stocks.start()
        await self.tree.sync()

    @tasks.loop(minutes=5)
    async def monitor_stocks(self):
        if not self.alert_channel: return
        now_et = datetime.now(MARKET_TIMEZONE)
        curr_time, today_str = now_et.time(), now_et.strftime('%Y-%m-%d')
        
        # [修改] 删除 pre-market 判断，只保留 10:00 后的扫描
        is_open_scan = TIME_MARKET_SCAN_START <= curr_time <= TIME_MARKET_CLOSE
        
        if not is_open_scan: return
        
        logging.info(f"[{now_et.strftime('%H:%M')}] Scanning started...")
        users_data = settings.get("users", {})
        all_tickers = set()
        ticker_user_map = defaultdict(list)
        
        for uid, udata in users_data.items():
            for k in list(udata['daily_status'].keys()):
                if not k.endswith(today_str): del udata['daily_status'][k]
            for ticker in udata.get("stocks", []):
                all_tickers.add(ticker)
                ticker_user_map[ticker].append(uid)

        if not all_tickers: 
            logging.info("No tickers to scan.")
            return

        hist_map = await fetch_historical_batch(list(all_tickers))
        quotes_map = {}
        if TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE:
            quotes_map = await fetch_realtime_quotes(list(all_tickers))

        alerts_buffer = []
        if "signal_history" not in settings: settings["signal_history"] = {}
        if today_str not in settings["signal_history"]: settings["signal_history"][today_str] = {}

        for ticker, df_hist in hist_map.items():
            df = df_hist
            if ticker in quotes_map:
                df = await asyncio.to_thread(merge_and_recalc_sync, df_hist, quotes_map[ticker])

            user_ids = ticker_user_map[ticker]
            all_alerted = True
            users_to_ping = []
            for uid in user_ids:
                status_key = f"{ticker}-{today_str}"
                status = users_data[uid]['daily_status'].get(status_key, "NONE")
                # 只有 NONE 状态才会发 (即当天第一次)
                if status == "NONE":
                    users_to_ping.append(uid)
                    all_alerted = False
            
            if all_alerted: continue

            history = settings.get("signal_history", {})
            in_cooldown = False
            cooldown_days = CONFIG["system"]["cooldown_days"]
            last_signal_score = 0
            
            for i in range(1, cooldown_days + 1): 
                past_date = (now_et.date() - timedelta(days=i)).strftime("%Y-%m-%d")
                if past_date in history and ticker in history[past_date]:
                    last_signal_score = history[past_date][ticker].get("score", 0)
                    in_cooldown = True 
            
            is_triggered, score, reason, res_line, sup_line = await check_signals(df)
            
            today_signal_data = settings["signal_history"][today_str].get(ticker)
            if today_signal_data:
                today_score = today_signal_data.get("score", 0)
                if score <= today_score:
                    is_triggered = False
                    logging.info(f"Ticker {ticker} skipped because a signal with score {today_score} was already sent today.")

            if is_triggered and in_cooldown and last_signal_score > 0:
                if score <= last_signal_score:
                    logging.info(f"Ticker {ticker} skipped due to cooldown (Last Score: {last_signal_score}).")
                    is_triggered = False
            
            if is_triggered:
                price = df['close'].iloc[-1]
                
                # [修改] 计算双线
                stop_loss, support = calculate_risk_levels(df)
                
                alert_obj = {
                    "ticker": ticker,
                    "score": score, 
                    "priority": score, 
                    "price": price,
                    "reason": reason,
                    "support": support,
                    "stop_loss": stop_loss, # 新增
                    "df": df,
                    "res_line": res_line,
                    "sup_line": sup_line,
                    "users": users_to_ping
                }
                alerts_buffer.append(alert_obj)

        if alerts_buffer:
            alerts_buffer.sort(key=lambda x: x["priority"], reverse=True)
            max_charts = CONFIG["system"]["max_charts_per_scan"]
            sent_charts = 0
            summary_list = []

            for alert in alerts_buffer:
                ticker = alert["ticker"]
                score = alert["score"]
                users = alert["users"]
                
                current_hist = settings["signal_history"][today_str].get(ticker, {})
                settings["signal_history"][today_str][ticker] = {
                    "score": score,
                    "price": alert["price"],
                    "time": now_et.strftime('%H:%M'),
                    "reason": alert["reason"],
                    "ret_1d": current_hist.get("ret_1d"),
                    "ret_5d": current_hist.get("ret_5d"),
                    "ret_20d": current_hist.get("ret_20d"),
                }
                
                for uid in users:
                    status_key = f"{ticker}-{today_str}"
                    users_data[uid]['daily_status'][status_key] = "MARKET_SENT"
                
                mentions = " ".join([f"<@{uid}>" for uid in users])
                
                if sent_charts < max_charts:
                    # [修改] 传递双线给画图
                    chart_buf = await generate_chart(
                        alert["df"], ticker, alert["res_line"], alert["sup_line"], 
                        alert["stop_loss"], alert["support"]
                    )
                    filename = f"{ticker}.png"
                    
                    embed = create_alert_embed(
                        ticker, score, alert['price'], alert['reason'], 
                        alert['stop_loss'], alert['support'], alert['df'], filename
                    )
                    
                    try:
                        file = discord.File(chart_buf, filename=filename)
                        await self.alert_channel.send(content=mentions, embed=embed, file=file)
                        sent_charts += 1
                        await asyncio.sleep(1.5)
                    except Exception as e: logging.error(f"Send Error: {e}")
                    finally:
                        chart_buf.close() 
                else:
                    summary_list.append(f"**{ticker}** ({score})")

            if summary_list:
                summary_msg = f"**其他提醒 (摘要)**:\n" + " | ".join(summary_list)
                try: 
                    await self.alert_channel.send(content=summary_msg)
                except: pass
            
            save_settings()
        
        logging.info(f"[{now_et.strftime('%H:%M')}] Scan finished. Alerts: {len(alerts_buffer)}")

intents = discord.Intents.default()
client = StockBotClient(intents=intents)

@client.tree.command(name="watch_add", description="Add stocks to watch list (e.g., AAPL, TSLA)")
@app_commands.describe(codes="Stock Symbols")
async def watch_add(interaction: discord.Interaction, codes: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in codes.replace(',', ' ').replace('，', ' ').split() if t.strip()]))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"Added: `{', '.join(new_list)}`")

@client.tree.command(name="watch_remove", description="Remove stocks from watch list")
@app_commands.describe(codes="Stock Symbols")
async def watch_remove(interaction: discord.Interaction, codes: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    to_remove = set([t.strip().upper() for t in codes.replace(',', ' ').replace('，', ' ').split() if t.strip()])
    current_list = user_data["stocks"]
    new_list = [s for s in current_list if s not in to_remove]
    user_data["stocks"] = new_list
    for t in to_remove:
        keys_to_del = [k for k in user_data['daily_status'] if k.startswith(t)]
        for k in keys_to_del: del user_data['daily_status'][k]
    save_settings()
    await interaction.followup.send(f"Removed: `{', '.join(to_remove)}`")

@client.tree.command(name="watch_list", description="Show my watch list")
async def watch_list(interaction: discord.Interaction):
    stocks = get_user_data(interaction.user.id)["stocks"]
    if len(stocks) > 60: display_str = ", ".join(stocks[:60]) + f"... ({len(stocks)})"
    else: display_str = ", ".join(stocks) if stocks else 'Empty'
    await interaction.response.send_message(f"List:\n`{display_str}`", ephemeral=True)

@client.tree.command(name="watch_clear", description="Clear all watched stocks")
async def watch_clear(interaction: discord.Interaction):
    user_data = get_user_data(interaction.user.id)
    user_data["stocks"] = []
    user_data["daily_status"] = {}
    save_settings()
    await interaction.response.send_message("Cleared.", ephemeral=True)

@client.tree.command(name="watch_import", description="Import preset lists")
@app_commands.choices(preset=[
    app_commands.Choice(name="NASDAQ 100", value="NASDAQ_100"),
    app_commands.Choice(name="GOD TIER", value="GOD_TIER")
])
async def watch_import(interaction: discord.Interaction, preset: app_commands.Choice[str]):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = STOCK_POOLS.get(preset.value, [])
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"Imported {preset.name} ({len(new_list)} stocks).")

@client.tree.command(name="stats", description="View historical signal accuracy")
async def stats_command(interaction: discord.Interaction):
    await interaction.response.defer()
    await update_stats_data()
    history = settings.get("signal_history", {})
    if not history:
        await interaction.followup.send("No data.")
        return
    stats = {
        "1d": {"c":0, "w":0, "r":0}, "5d": {"c":0, "w":0, "r":0}, "20d": {"c":0, "w":0, "r":0}
    }
    recent = []
    for d in sorted(history.keys(), reverse=True):
        for t, data in history[d].items():
            score = data.get("score", 0) 
            level_str = get_level_by_score(score)
            r1, r5, r20 = data.get("ret_1d"), data.get("ret_5d"), data.get("ret_20d")
            
            if r1 is not None:
                stats["1d"]["c"]+=1; stats["1d"]["r"]+=r1
                if r1>0: stats["1d"]["w"]+=1
            if r5 is not None:
                stats["5d"]["c"]+=1; stats["5d"]["r"]+=r5
                if r5>0: stats["5d"]["w"]+=1
            if r20 is not None:
                stats["20d"]["c"]+=1; stats["20d"]["r"]+=r20
                if r20>0: stats["20d"]["w"]+=1

            if len(recent) < 8:
                rets = []
                if r1 is not None: rets.append(f"1D:{r1}%")
                if r5 is not None: rets.append(f"1W:{r5}%")
                if r20 is not None: rets.append(f"1M:{r20}%")
                ret_str = " | ".join(rets) if rets else "Pending"
                recent.append(f"`{d}` {level_str} **{t}** ({score})\n - {ret_str}")

    embed = discord.Embed(title="回测统计数据", description="(基于历史信号)", color=0x00BFFF)
    def mk_stat(k, l):
        s = stats[k]
        if s["c"]==0: return f"{l}: 无数据"
        return f"**{l}**\n胜率: `{s['w']/s['c']*100:.1f}%`\n平均收益: `{s['r']/s['c']:.2f}%`"
        
    embed.add_field(name="1日表现", value=mk_stat("1d", "次日"), inline=True)
    embed.add_field(name="5日表现", value=mk_stat("5d", "一周"), inline=True)
    embed.add_field(name="20日表现", value=mk_stat("20d", "一月"), inline=True)
    if recent: embed.add_field(name="最近信号", value="\n".join(recent), inline=False)
    await interaction.followup.send(embed=embed)

@client.tree.command(name="test", description="Test single stock")
async def test_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    
    logging.info(f"[TEST Command] Testing: {ticker}")

    data_map = await fetch_historical_batch([ticker])
    quotes_map = await fetch_realtime_quotes([ticker])
    
    if not data_map or ticker not in data_map:
        await interaction.followup.send(f"Failed `{ticker}` (Check logs for 403/429 or data error)")
        return
        
    df = data_map[ticker]
    if ticker in quotes_map:
        df = await asyncio.to_thread(merge_and_recalc_sync, df, quotes_map[ticker])

    is_triggered, score, reason, r_l, s_l = await check_signals(df)
    
    price = df['close'].iloc[-1]
    
    # [修改] 计算双线
    stop_loss, support = calculate_risk_levels(df)

    if not reason: 
        reason = f"无明显信号 (得分: {score})"
    
    # [修改] 传递双线给画图
    chart_buf = await generate_chart(df, ticker, r_l, s_l, stop_loss, support)
    filename = f"{ticker}_test.png"
    
    # [修改] 传递双线给 Embed
    embed = create_alert_embed(ticker, score, price, reason, stop_loss, support, df, filename)

    try:
        f = discord.File(chart_buf, filename=filename)
        await interaction.followup.send(embed=embed, file=f)
    except Exception as e:
        logging.error(f"Send Error: {e}")
        await interaction.followup.send(f"Failed to send image: {e}")
    finally:
        chart_buf.close()

if __name__ == "__main__":
    if DISCORD_TOKEN: client.run(DISCORD_TOKEN)
