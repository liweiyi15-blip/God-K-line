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
from scipy.stats import linregress
import aiohttp
import io
import matplotlib

# [新增] 引入 logging 模块
import logging
# 配置日志输出格式，方便在 Railway 上查看
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


# --- 强制使用非交互式后端，防止Docker/Railway崩溃 ---
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

# 适配 Railway Volume 路径
SETTINGS_FILE = "/app/data/settings.json"
if not os.path.exists("/app/data"):
    # 本地开发回退路径
    SETTINGS_FILE = "settings.json"

TIME_PRE_MARKET_START = time(9, 0)
TIME_MARKET_OPEN = time(9, 30)
TIME_MARKET_CLOSE = time(16, 0)

# --- 核心策略配置 ---
CONFIG = {
    "filter": {
        "max_60d_gain": 3.0,      
        "max_rsi": 82,              
        "max_bias_50": 0.45,        
        "max_upper_shadow": 0.4,    
        "min_adx_trend": 20,        
        "max_day_change": 0.15,   
        "min_vol_ratio": 1.3,     
        "intraday_vol_ratio_normal": 1.8, 
        "intraday_vol_ratio_open": 3.5, 
        "min_converge_angle": 0.05
    },
    "pattern": {
        "min_r2": 0.70,
        "window": 60
    },
    "system": {
        "cooldown_days": 3,            
        "max_charts_per_scan": 5,
        "history_days": 400
    },
    "emoji": {
        "GOD_TIER": "👑", "S_TIER": "🔥", "A_TIER": "📈", 
        "B_TIER": "💎", "C_TIER": "🚀", "RISK": "🛡️"
    },
    "priority": {
        "GOD_TIER": 100, "S_TIER": 90, "A_TIER": 80, 
        "B_TIER": 70, "C_TIER": 60, "NORMAL": 0
    }
}

# --- 静态股票池 ---
STOCK_POOLS = {
    "NASDAQ_100": ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "ADBE", "COST", "PEP", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "CMCSA", "AZN", "QCOM", "TXN", "AMGN", "HON", "INTU", "SBUX", "GILD", "BKNG", "DIOD", "MDLZ", "ISRG", "REGN", "LRCX", "VRTX", "ADP", "ADI", "MELI", "KLAC", "PANW", "SNPS", "CDNS", "CHTR", "MAR", "CSX", "ORLY", "MNST", "NXPI", "CTAS", "FTNT", "WDAY", "DXCM", "PCAR", "KDP", "PAYX", "IDXX", "AEP", "LULU", "EXC", "BIIB", "ADSK", "XEL", "ROST", "MCHP", "CPRT", "DLTR", "EA", "FAST", "CTSH", "WBA", "VRSK", "CSGP", "ODFL", "ANSS", "EBAY", "ILMN", "GFS", "ALGN", "TEAM", "CDW", "WBD", "SIRI", "ZM", "ENPH", "JD", "PDD", "LCID", "RIVN", "ZS", "DDOG", "CRWD", "TTD", "BKR", "CEG", "GEHC", "ON", "FANG"],
    "GOD_TIER": ["NVDA", "AMD", "TSM", "SMCI", "AVGO", "ARM", "PLTR", "AI", "PATH", "BABA", "PDD", "BIDU", "NIO", "LI", "XPEV", "COIN", "MARA", "MSTR"]
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
    """计算核心指标 (包含 ADX, Bias, K线形态)"""
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

async def fetch_historical_batch(symbols: list, days=None):
    if not symbols: return {}
    if days is None: days = CONFIG["system"]["history_days"]
    
    results = {}
    now = datetime.now()
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
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
                    if response.status == 200:
                        data = await response.json()
                        df = None
                        if isinstance(data, list) and len(data) > 0 and 'date' in data[0] and 'close' in data[0]:
                            df = await asyncio.to_thread(process_dataframe_sync, data)
                        elif isinstance(data, dict) and 'historical' in data:
                            df = await asyncio.to_thread(process_dataframe_sync, data['historical'])
                        elif isinstance(data, list) and len(data) > 0 and 'historical' in data[0]:
                            for item in data:
                                if item.get('symbol') == sym:
                                    df = await asyncio.to_thread(process_dataframe_sync, item['historical'])
                                    break
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

def linreg_trend(points, min_r2):
    if len(points) < 4: return None
    x = np.arange(len(points))
    y = points.values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_sq = r_value ** 2
    if r_sq < min_r2: return None
    return slope, intercept, r_sq

def identify_patterns(df):
    window = CONFIG["pattern"]["window"]
    min_r2 = CONFIG["pattern"]["min_r2"]
    if len(df) < window + 5: return None, [], []
    recent = df.tail(window).copy()
    recent = recent.reset_index()
    recent['pivot_high'] = recent['high'].rolling(5, center=True).max() == recent['high']
    recent['pivot_low'] = recent['low'].rolling(5, center=True).min() == recent['low']
    high_points = recent[recent['pivot_high']]
    low_points = recent[recent['pivot_low']]
    
    if len(high_points) >= 3 and len(low_points) >= 3:
        h_data = high_points['high'].tail(8)
        l_data = low_points['low'].tail(8)
        res_trend = linreg_trend(h_data, min_r2)
        sup_trend = linreg_trend(l_data, min_r2)
        
        if res_trend and sup_trend:
            slope_res, int_res, r2_res = res_trend
            slope_sup, int_sup, r2_sup = sup_trend
            
            if slope_res < 0 and (slope_sup > slope_res + CONFIG["filter"]["min_converge_angle"]):
                curr_idx = recent.index[-1]
                resistance_today = slope_res * curr_idx + int_res
                curr_close = recent['close'].iloc[-1]
                
                prev_idx = recent.index[-2]
                res_prev = slope_res * prev_idx + int_res
                prev_close = recent['close'].iloc[-2]
                
                if prev_close <= res_prev * 1.02:
                    if curr_close > resistance_today:
                        t1, t2 = recent['date'].iloc[0], recent['date'].iloc[-1]
                        p1, p2 = slope_res * recent.index[0] + int_res, slope_res * recent.index[-1] + int_res
                        t3, t4 = t1, t2
                        p3, p4 = slope_sup * recent.index[0] + int_sup, slope_sup * recent.index[-1] + int_sup
                        return "🚩 **放量旗形突破(机构算法)**", [[(t1,p1), (t2,p2)]], [[(t3,p3), (t4,p4)]]
    return None, [], []

def get_volume_projection_factor(ny_now, minutes_elapsed):
    """[新增] 采用分段/动态成交量推算公式"""
    
    # 市场开放总分钟数
    TOTAL_MINUTES = 390
    
    # 极早期开盘阶段 (前10分钟)
    if minutes_elapsed <= 10:
        # 10分钟内，推算因子固定为 13 倍 (即开盘阶段的平均相对量)
        return 13.0
    
    # 早盘阶段 (10 - 60 分钟)
    elif minutes_elapsed <= 60:
        # 从 13 倍开始线性递减到 8.0 倍
        factor = 13.0 - (13.0 - 8.0) * (minutes_elapsed - 10) / 50
        return factor
    
    # 盘中阶段 (60 - 390 分钟)
    else:
        # 从 8.0 倍开始线性递减到 4.0 倍
        factor = 8.0 - (8.0 - 4.0) * (minutes_elapsed - 60) / (TOTAL_MINUTES - 60)
        return factor


def check_signals_sync(df):
    if len(df) < 60: return False, "", "NONE", [], []
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    level = "NORMAL"

    low_60 = df['low'].tail(60).min()
    # 基础风控
    if curr['close'] > low_60 * CONFIG["filter"]["max_60d_gain"]: return False, "", "RISK_FILTER", [], []
    prev_close_safe = prev['close'] if prev['close'] > 0 else 1.0
    if abs((curr['close'] - prev['close']) / prev_close_safe) > CONFIG["filter"]["max_day_change"]: return False, "", "RISK_FILTER", [], []
    if curr['RSI'] > CONFIG["filter"]["max_rsi"]: return False, "", "RISK_FILTER", [], []
    
    # [优化] 乖离率风控
    if curr['BIAS_50'] > CONFIG["filter"]["max_bias_50"]:
         return False, "", "RISK_OVEREXTENDED", [], []

    # [优化] K线形态风控 (拒绝长上影线)
    if curr['Upper_Shadow_Ratio'] > CONFIG["filter"]["max_upper_shadow"]:
        return False, "", "REJECT_WICK", [], []

    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
    
    is_open_market = 0 < minutes_elapsed < 390
    
    if is_open_market:
        safe_minutes = max(minutes_elapsed, 1) # 至少1分钟
        # [优化] 使用分段推算因子
        projection_factor = get_volume_projection_factor(ny_now, safe_minutes)
        vol_threshold = CONFIG["filter"]["intraday_vol_ratio_normal"] 
        
        proj_vol = curr['volume'] * projection_factor
    else:
        proj_vol = curr['volume']
        vol_threshold = CONFIG["filter"]["min_vol_ratio"]
        
    is_heavy_volume = proj_vol > curr['Vol_MA20'] * vol_threshold

    # --- 策略部分 ---

    # 策略 1: BB Squeeze (增强版: 需要 ADX 确认)
    if curr['BB_Width'] < 0.08:
        if curr['close'] > curr['BB_Up'] and is_heavy_volume:
            if curr['ADX'] > 15 and curr['PDI'] > curr['MDI']:
                triggers.append(f"🚀 **BB Squeeze (Trend Confirm)**: 紧缩突破 + 趋势增强")
                if level == "NORMAL": level = "S_TIER"

    # 策略 2: Nx 蓝梯 (增强版: 需要强趋势)
    recent_10 = df.tail(10)
    had_breakout = (recent_10['close'] > recent_10['Nx_Blue_UP']).any()
    on_support = curr['close'] > curr['Nx_Blue_DW'] and curr['low'] <= curr['Nx_Blue_UP'] * 1.02
    
    is_strong_trend = curr['ADX'] > CONFIG["filter"]["min_adx_trend"] and curr['PDI'] > curr['MDI']

    if had_breakout and on_support and is_heavy_volume and is_strong_trend:
        triggers.append(f"👑 **Nx 趋势起爆**: 蓝梯回踩 + 强动能(ADX>20)")
        level = "GOD_TIER"

    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name and is_heavy_volume:
        triggers.append(pattern_name)
        if level != "GOD_TIER": level = "S_TIER"

    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        if curr['PDI'] > curr['MDI']:
            triggers.append(f"📈 **Nx 蓝梯突破**: 趋势转多确认")
            if level not in ["GOD_TIER", "S_TIER"]: level = "A_TIER"

    # 策略 3: KDJ / MACD
    price_low_20 = df['close'].tail(20).min()
    price_is_low = curr['close'] <= price_low_20 * 1.02
    
    if prev['J'] < 0 and curr['J'] > 0 and curr['K'] > curr['D']:
        triggers.append(f"💎 **KDJ 绝地反击**: 极度超卖 J 值回升")
        if level == "NORMAL": level = "B_TIER"
    
    macd_low_20 = df['MACD'].tail(20).min()
    if price_is_low and curr['MACD'] < 0:
        if curr['MACD'] > macd_low_20 * 0.8:
             if curr['DIF'] > df['DIF'].tail(20).min():
                triggers.append(f"🛡️ **Cd 结构底背离**: 价格新低动能衰竭")
                if level not in ["GOD_TIER", "S_TIER", "A_TIER"]: level = "B_TIER"

    # 策略 4: 抛售高潮 (小盘股)
    pinbar_ratio = (curr['close'] - curr['low']) / (curr['high'] - curr['low'] + 1e-9)
    market_cap = df.attrs.get('marketCap', float('inf')) 
    
    if curr['low'] < curr['BB_Low']:
        if proj_vol > curr['Vol_MA20'] * 2.5:
            if pinbar_ratio > 0.5:
                if market_cap < 5_000_000_000:
                    triggers.append(f"🛡️ **抛售高潮 (小盘股)**: 恐慌盘涌出后 V 反")
                    level = "A_TIER"

    if triggers:
        if is_downtrend and len(triggers) < 2 and level not in ["GOD_TIER", "S_TIER"]:
            return False, "", "WEAK_SIGNAL", [], []
        return True, "\n".join(triggers), level, res_line, sup_line
    return False, "", "NONE", [], []

async def check_signals(df):
    return await asyncio.to_thread(check_signals_sync, df)

def _generate_chart_sync(df, ticker, res_line=[], sup_line=[]):
    buf = io.BytesIO()
    
    last_close = df['close'].iloc[-1]
    last_atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else last_close * 0.05
    stop_price = last_close - 2 * last_atr

    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    my_style = mpf.make_mpf_style(base_mpl_style="ggplot", marketcolors=s, gridstyle=":")
    plot_df = df.tail(80)
    
    stop_line = [stop_price] * len(plot_df)

    add_plots = [
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        mpf.make_addplot(stop_line, color='red', linestyle='--', width=1.2),
        mpf.make_addplot(plot_df['MACD'], panel=2, type='bar', color='dimgray', alpha=0.5, ylabel='MACD'),
        mpf.make_addplot(plot_df['DIF'], panel=2, color='orange'),
        mpf.make_addplot(plot_df['DEA'], panel=2, color='blue'),
    ]
    
    # [优化 1] 增加 tight_layout=True 来消除边框空白
    kwargs = dict(type='candle', style=my_style, title=f"{ticker} Analysis", ylabel='Price', addplot=add_plots, volume=True, panel_ratios=(6, 2, 2), tight_layout=True, savefig=buf)
    
    if res_line: 
        all_lines = []
        if res_line: all_lines.extend(res_line)
        if sup_line: all_lines.extend(sup_line)
        # [优化 2] 改变线颜色和宽度，使其更清晰
        kwargs['alines'] = dict(alines=all_lines, colors='darkgray', linewidths=2.0, linestyle='-')
    
    try:
        mpf.plot(plot_df, **kwargs)
        buf.seek(0)
    finally:
        # 清理内存
        plt.close('all')
        
    return buf

async def generate_chart(df, ticker, res_line=[], sup_line=[]):
    return await asyncio.to_thread(_generate_chart_sync, df, ticker, res_line, sup_line)

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
        
        is_pre = TIME_PRE_MARKET_START <= curr_time < TIME_MARKET_OPEN
        is_open = TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE
        if not (is_pre or is_open): return
        
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

        # 1. 获取历史数据
        hist_map = await fetch_historical_batch(list(all_tickers))
        
        # 2. 获取实时报价
        quotes_map = {}
        if is_open:
            quotes_map = await fetch_realtime_quotes(list(all_tickers))

        alerts_buffer = []

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
                should_alert = False
                if is_pre and status == "NONE": should_alert = True
                if is_open and status in ["NONE", "PRE_SENT"]: should_alert = True
                if should_alert:
                    users_to_ping.append(uid)
                    all_alerted = False
            
            if all_alerted: continue

            history = settings.get("signal_history", {})
            in_cooldown = False
            cooldown_days = CONFIG["system"]["cooldown_days"]
            
            last_signal_level = None
            
            # [修复] 冷却逻辑: i 从 1 开始，排除今天
            for i in range(1, cooldown_days + 1): 
                past_date = (now_et.date() - timedelta(days=i)).strftime("%Y-%m-%d")
                if past_date in history and ticker in history[past_date]:
                    last_signal_level = history[past_date][ticker]["level"]
                    in_cooldown = True 
            
            is_triggered, reason, level, res_line, sup_line = await check_signals(df)
            
            if in_cooldown and last_signal_level:
                current_prio = CONFIG["priority"].get(level, 0)
                last_prio = CONFIG["priority"].get(last_signal_level, 0)
                if current_prio <= last_prio:
                    logging.info(f"Ticker {ticker} skipped due to cooldown ({last_signal_level}).")
                    continue 

            if is_triggered:
                price = df['close'].iloc[-1]
                atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else (price * 0.05)
                stop_loss = price - (2 * atr_val)
                
                alert_obj = {
                    "ticker": ticker,
                    "level": level,
                    "priority": CONFIG["priority"].get(level, 0),
                    "price": price,
                    "reason": reason,
                    "support": stop_loss,
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
                level = alert["level"]
                users = alert["users"]
                
                if "signal_history" not in settings: settings["signal_history"] = {}
                if today_str not in settings["signal_history"]: settings["signal_history"][today_str] = {}
                
                if ticker not in settings["signal_history"][today_str]:
                    settings["signal_history"][today_str][ticker] = {
                        "level": level,
                        "price": alert["price"],
                        "time": now_et.strftime('%H:%M'),
                        "reason": alert["reason"],
                        "ret_1d": None, "ret_5d": None, "ret_20d": None
                    }

                for uid in users:
                    status_key = f"{ticker}-{today_str}"
                    new_status = "PRE_SENT" if is_pre else ("BOTH_SENT" if users_data[uid]['daily_status'].get(status_key) == "PRE_SENT" else "MARKET_SENT")
                    users_data[uid]['daily_status'][status_key] = new_status
                
                mentions = " ".join([f"<@{uid}>" for uid in users])
                emoji = CONFIG["emoji"].get(level, "🚨")
                
                if sent_charts < max_charts:
                    chart_buf = await generate_chart(alert["df"], ticker, alert["res_line"], alert["sup_line"])
                    msg = (
                        f"{mentions}\n【{emoji} {level} 信号】\n"
                        f"🎯 **{ticker}** | 💰 `${alert['price']:.2f}`\n"
                        f"{'-'*20}\n{alert['reason']}\n{'-'*20}\n"
                        f"🛑 动态止损(2ATR): `${alert['support']:.2f}`"
                    )
                    try:
                        file = discord.File(chart_buf, filename=f"{ticker}.png")
                        await self.alert_channel.send(content=msg, file=file)
                        sent_charts += 1
                        await asyncio.sleep(1.5)
                    except Exception as e: logging.error(f"Send Error: {e}")
                    finally:
                        chart_buf.close() 
                else:
                    summary_list.append(f"{emoji} **{ticker}** ({level})")

            if summary_list:
                summary_msg = f"📋 **其他触发信号 (简报)**:\n" + " | ".join(summary_list)
                try: 
                    await self.alert_channel.send(content=summary_msg)
                except: pass
            
            save_settings()
        
        logging.info(f"[{now_et.strftime('%H:%M')}] Scan finished. Alerts: {len(alerts_buffer)}")

intents = discord.Intents.default()
client = StockBotClient(intents=intents)

@client.tree.command(name="watch_add", description="批量添加关注 (如: AAPL, TSLA)")
@app_commands.describe(codes="股票代码")
async def watch_add(interaction: discord.Interaction, codes: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in codes.replace(',', ' ').replace('，', ' ').split() if t.strip()]))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已关注: `{', '.join(new_list)}`")

@client.tree.command(name="watch_remove", description="批量移除关注")
@app_commands.describe(codes="股票代码")
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
    await interaction.followup.send(f"🗑️ 已移除: `{', '.join(to_remove)}`")

@client.tree.command(name="watch_list", description="查看我的关注列表")
async def watch_list(interaction: discord.Interaction):
    stocks = get_user_data(interaction.user.id)["stocks"]
    if len(stocks) > 60: display_str = ", ".join(stocks[:60]) + f"... ({len(stocks)})"
    else: display_str = ", ".join(stocks) if stocks else '空'
    await interaction.response.send_message(f"📋 **列表**:\n`{display_str}`", ephemeral=True)

@client.tree.command(name="watch_clear", description="清空所有关注")
async def watch_clear(interaction: discord.Interaction):
    user_data = get_user_data(interaction.user.id)
    user_data["stocks"] = []
    user_data["daily_status"] = {}
    save_settings()
    await interaction.response.send_message("🗑️ 已清空。", ephemeral=True)

@client.tree.command(name="watch_import", description="一键导入推荐列表")
@app_commands.choices(preset=[
    app_commands.Choice(name="纳指100 (NASDAQ 100)", value="NASDAQ_100"),
    app_commands.Choice(name="神级热门 (GOD TIER)", value="GOD_TIER")
])
async def watch_import(interaction: discord.Interaction, preset: app_commands.Choice[str]):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = STOCK_POOLS.get(preset.value, [])
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已导入 {preset.name} (共 {len(new_list)} 只)。")

@client.tree.command(name="stats", description="查看历史信号胜率")
async def stats_command(interaction: discord.Interaction):
    await interaction.response.defer()
    await update_stats_data()
    history = settings.get("signal_history", {})
    if not history:
        await interaction.followup.send("📭 暂无数据。")
        return
    stats = {
        "1d": {"c":0, "w":0, "r":0}, "5d": {"c":0, "w":0, "r":0}, "20d": {"c":0, "w":0, "r":0}
    }
    recent = []
    for d in sorted(history.keys(), reverse=True):
        for t, data in history[d].items():
            level = data.get("level", "NORMAL")
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
                emoji = CONFIG["emoji"].get(level, "🔹")
                rets = []
                if r1: rets.append(f"1D:{r1}%")
                if r5: rets.append(f"1W:{r5}%")
                if r20: rets.append(f"1M:{r20}%")
                ret_str = " | ".join(rets) if rets else "⏳"
                recent.append(f"`{d}` {emoji} **{t}**\n╚ {ret_str}")

    embed = discord.Embed(title="📊 多周期回测统计", color=0x00BFFF)
    def mk_stat(k, l):
        s = stats[k]
        if s["c"]==0: return f"{l}: 无数据"
        return f"**{l}**\n胜: `{s['w']/s['c']*100:.1f}%`\n盈: `{s['r']/s['c']:.2f}%`"
        
    embed.add_field(name="📅 1 Day", value=mk_stat("1d", "次日"), inline=True)
    embed.add_field(name="🗓️ 5 Days", value=mk_stat("5d", "一周"), inline=True)
    embed.add_field(name="🌙 20 Days", value=mk_stat("20d", "一月"), inline=True)
    if recent: embed.add_field(name="🕒 最近信号", value="\n".join(recent), inline=False)
    await interaction.followup.send(embed=embed)

@client.tree.command(name="test", description="测试单股")
async def test_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    
    logging.info(f"[TEST 指令收到] 正在测试: {ticker}")

    data_map = await fetch_historical_batch([ticker])
    quotes_map = await fetch_realtime_quotes([ticker])
    
    if not data_map or ticker not in data_map:
        await interaction.followup.send(f"❌ 失败 `{ticker}` (请查看后台详细日志，可能被403/429拦截或数据解析失败)")
        return
        
    df = data_map[ticker]
    if ticker in quotes_map:
        df = await asyncio.to_thread(merge_and_recalc_sync, df, quotes_map[ticker])

    is_triggered, reason, level, r_l, s_l = await check_signals(df)
    
    price = df['close'].iloc[-1]
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else (price * 0.05)
    stop_loss = price - (2 * atr_val)

    if not reason: reason = "手动测试 (无信号)"
    
    chart_buf = await generate_chart(df, ticker, r_l, s_l)
    msg = f"✅ `{ticker}` | {level}\n💰 `${price:.2f}`\n📝 {reason}\n🛑 Stop: `${stop_loss:.2f}`"
    try:
        f = discord.File(chart_buf, filename=f"{ticker}_test.png")
        await interaction.followup.send(content=msg, file=f)
    except Exception as e:
        logging.error(f"Send Error: {e}")
        await interaction.followup.send(f"⚠️ 发送图片失败: {e}")
    finally:
        chart_buf.close()

if __name__ == "__main__":
    if DISCORD_TOKEN: client.run(DISCORD_TOKEN)
