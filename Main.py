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
import random

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

# [配置] 扫描时间
TIME_MARKET_OPEN = time(9, 30)
TIME_MARKET_SCAN_START = time(10, 0) # 10点才开始报
TIME_MARKET_CLOSE = time(16, 0)

# --- 核心策略配置 (完全注释版) ---
CONFIG = {
    # -------------------------------------------------------------------------
    # [1] 过滤器 (Filter)：一票否决制
    # -------------------------------------------------------------------------
    "filter": {
        # [防追高] 60日涨幅限制
        # 含义：当前价格不能超过 (过去60天最低价 * 1.3)。
        "max_60d_gain": 0.3,

        # [防过热] RSI 超买限制
        # 含义：RSI(14) 指标不能超过 60。
        "max_rsi": 60,

        # [防回落] 乖离率限制
        # 含义：现价不能比 50日均线 (MA50) 高出 20%。
        "max_bias_50": 0.20,

        # [防抛压] 上影线限制
        # 含义：上影线长度不能超过 K线总长度的 40%。
        "max_upper_shadow": 0.4,

        # [防疯牛/妖股] 单日波动限制
        # 含义：当天涨跌幅绝对值不能超过 70%。
        "max_day_change": 0.7,

        # [资金门槛] 量比阈值
        # 含义：当前成交量(或预估量) 必须是 20日均量 的 1.15倍以上。
        "min_vol_ratio": 1.15,
        
        # --- 布林带挤压 (BB Squeeze) 策略参数 ---
        
        # [蓄势条件] 极度压缩阈值
        # 含义：昨日的布林带带宽 (Bandwidth) 必须小于 8%。
        "min_bb_squeeze_width": 0.08,

        # [启动条件] 开口扩张阈值
        # 含义：今日的布林带带宽必须扩大到 9.5% 以上。
        "min_bb_expand_width": 0.095,

        # [抄底位置] 底部位置分位数
        # 含义：当前价格必须处于过去60天价格区间的 30% 分位以下。
        "max_bottom_pos": 0.30,
        
        # [趋势潜能] ADX 活跃度
        # 含义：ADX (趋势强度) 必须大于 15。
        "min_adx_for_squeeze": 15
    },

    # -------------------------------------------------------------------------
    # [2] 形态识别参数 (Pattern)
    # -------------------------------------------------------------------------
    "pattern": {
        # [关键] 枢轴点窗口 (Pivot Window)
        # 数值 10：意味着这个点必须是前后10天(共21天)内的最高点，才能被连线。
        # 作用：数值越大，画出的趋势线越跨越长周期（大级别），过滤掉小杂波。
        "pivot_window": 10
    },

    # -------------------------------------------------------------------------
    # [3] 系统参数 (System)
    # -------------------------------------------------------------------------
    "system": {
        # [防刷屏] 冷却时间
        "cooldown_days": 3,

        # [防拥堵] 单次发送限制
        "max_charts_per_scan": 5,

        # [数据源] 历史数据回溯天数
        # 原因：为了画出一年前的大级别形态（如ADBE），必须请求足够长的数据。
        "history_days": 300
    },

    # -------------------------------------------------------------------------
    # [4] 打分权重 (Score) & 条件参数 (Params)
    # -------------------------------------------------------------------------
    "SCORE": { 
        # [及格线]
        "MIN_ALERT_SCORE": 70, 

        # [新增] 策略参数表：这里控制触发得分的具体条件
        "PARAMS": {
            # 1. 巨量 (HEAVY_VOLUME)
            # 含义：当前成交量是 20日均量 的多少倍才算巨量？
            "heavy_vol_multiplier": 1.55, 

            # 2. 强趋势 (STRONG_ADX)
            # 含义：ADX 大于多少才算强趋势？
            "adx_strong_threshold": 25,

            # 3. 趋势激活 (ADX_ACTIVATION)
            # 含义：之前的 ADX 必须小于多少（代表盘整），现在拐头向上才算激活？
            "adx_activation_lower": 20,

            # 4. KDJ 反击 (KDJ_REBOUND)
            # 含义：昨天的 J 线必须小于多少（超卖），今天金叉才给分？
            "kdj_j_oversold": 0,

            # 5. MACD 底背离 (MACD_DIVERGE)
            # 含义：股价必须接近过去20天最低价的多少范围？(1.02 = 2%以内)
            "divergence_price_tolerance": 1.02,
            # 含义：当前 MACD 值必须高于过去最低 MACD 的多少？(0.8 = 只要不比最低点低太多就行，或者比最低点高)
            "divergence_macd_strength": 0.8,

            # 6. OBV 资金流 (OBV_TREND_UP)
            # 含义：比较当前 OBV 和多少天前的 OBV？(确认近期是在流入)
            "obv_lookback": 5,

            # 7. 抛售高潮 (CAPITULATION)
            # 含义：量能必须大于均量的多少倍？
            "capitulation_vol_mult": 2,
            # 含义：下影线比例必须大于多少？(0.5 = 下影线占K线一半以上)
            "capitulation_pinbar": 0.5,
            # 含义：市值小于多少才算小盘股容易被操纵？(50亿)
            "capitulation_mcap": 5_000_000_000
        },

        # [加分项] 分值权重
        "WEIGHTS": {
            # --- 第一梯队：最强信号 ---
            "PATTERN_BREAK": 40,   # 形态突破
            "NX_BREAKOUT": 35,     # 均线突破
            "BB_SQUEEZE": 30,      # 布林挤压

            # --- 第二梯队：趋势确认 ---
            "GOD_TIER_NX": 20,     # 回踩支撑
            "STRONG_ADX": 20,      # 强趋势
            "ADX_ACTIVATION": 20,  # 趋势激活

            # --- 第三梯队：辅助验证 ---
            "OBV_TREND_UP": 15,    # 资金流入
            "CAPITULATION": 12,    # 抛售高潮
            "HEAVY_VOLUME": 10,    # 巨量

            # --- 第四梯队：常规指标 ---
            "MACD_ZERO_CROSS": 10, 
            "MACD_DIVERGE": 10,    
            "KDJ_REBOUND": 8,      
            "CANDLE_PATTERN": 5    
        },

        # [评级标签]
        "EMOJI": { 
            100: "TOP", 90: "HIGH", 80: "MID", 70: "LOW", 60: "TEST"
        }
    }
}

# --- 静态股票池 ---
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

    # 11. OBV (能量潮)
    obv_sign = np.sign(df['close'].diff()).fillna(0)
    df['OBV'] = (df['volume'] * obv_sign).cumsum()
    df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

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
      
    connector = aiohttp.TCPConnector(limit=3)
    semaphore = asyncio.Semaphore(3)
      
    headers = {
        "User-Agent": "Mozilla/5.0 (StockBot/1.0)",
        "Accept": "application/json"
    }

    async def fetch_single(session, sym):
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={sym}&from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
        async with semaphore:
            retries = 3
            for i in range(retries):
                try:
                    async with session.get(url, ssl=False) as response:
                        if response.status == 429:
                            wait_time = 3 * (2 ** i)
                            logging.warning(f"[429 Rate Limit] {sym}. Retry {i+1}/{retries} in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue 
                          
                        if response.status == 200:
                            data = await response.json()
                            df = await asyncio.to_thread(_safely_process_fmp_data, data, sym)
                            if df is not None and not df.empty:
                                results[sym] = df
                            else:
                                logging.warning(f"[数据为空] {sym}")
                            break 
                        else:
                            logging.error(f"[HTTP 错误] {sym} Status: {response.status}")
                            break
                except Exception as e:
                    logging.error(f"[异常] {sym}: {e}")
                    break
            await asyncio.sleep(0.5)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks_list = [fetch_single(session, sym) for sym in symbols]
        await asyncio.gather(*tasks_list)
    return results

async def fetch_realtime_quotes(symbols: list):
    if not symbols: return {}
    quotes_map = {}
      
    connector = aiohttp.TCPConnector(limit=5)
    semaphore = asyncio.Semaphore(5)
    headers = {"User-Agent": "StockBot/1.0", "Accept": "application/json"}
      
    async def fetch_single_quote(session, sym):
        url = f"https://financialmodelingprep.com/stable/quote?symbol={sym}&apikey={FMP_API_KEY}"
        async with semaphore:
            retries = 3
            for i in range(retries):
                try:
                    async with session.get(url, ssl=False) as response:
                        if response.status == 429:
                            wait_time = 2 * (2 ** i)
                            logging.warning(f"[429 Rate Limit] Quote {sym}. Retry {i+1}/{retries} in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue

                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, list):
                                for item in data:
                                    s = item.get('symbol')
                                    if s: quotes_map[s] = item
                            elif isinstance(data, dict):
                                     s = data.get('symbol')
                                     if s: quotes_map[s] = data
                            break
                        else:
                             logging.error(f"[Quote Error] {sym} Status: {response.status}")
                             break
                except Exception as e:
                    logging.error(f"[Quote Exception] {sym}: {e}")
                    break
            await asyncio.sleep(0.2)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks_list = [fetch_single_quote(session, sym) for sym in symbols]
        await asyncio.gather(*tasks_list)
    return quotes_map

async def fetch_market_index_data(days=60):
    now = datetime.now()
    from_date = (now - timedelta(days=days + 30)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")
      
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol=%5EIXIC&from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
      
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, list):
                        df = pd.DataFrame(data)
                        if 'date' in df.columns and 'price' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date').sort_index(ascending=True)
                            return df
        except Exception as e:
            logging.error(f"[Market Index Error] {e}")
    return None

def find_pivots(df, window=10):
    highs = df['high'].values
    lows = df['low'].values
    dates = df.index
      
    pivots_high = [] 
    pivots_low = []
      
    # 强制回溯 250 天
    lookback_days = 250
    start_idx = max(0, len(df) - lookback_days)
      
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
        
        # 保存 (date, price, index)
        if is_high: pivots_high.append((dates[i], current_high, i))
        if is_low: pivots_low.append((dates[i], current_low, i))
            
    return pivots_high, pivots_low

# [核心修复] 重写形态识别算法：改为“外切线”逻辑，确保不切穿中间价格
# [修改] 增加返回值 min_anchor_idx
def identify_patterns(df):
    if len(df) < 60: return None, [], [], None
    
    # 获取所有高低点 Pivot
    pivots_high, pivots_low = find_pivots(df, window=5) # 稍微缩小窗口以捕捉更多细节点
    
    res_line, sup_line = [], []
    pattern_name = None
    min_anchor_idx = None # [新增] 用于记录最早的锚点位置
    
    # 视野起点
    vis_start_idx = max(0, len(df) - 250)
    curr_idx = len(df) - 1
    
    # [修复点] 提前定义时间变量，防止后面 UnboundLocalError
    t_start = df.index[vis_start_idx]
    t_end = df.index[curr_idx]
    
    # --- 1. 阻力线 (Resistance) - 优化版：外包络逻辑 ---
    if pivots_high:
        # 步骤 A: 寻找“锚点” (Anchor)
        # 策略：在过去的数据中（排除最近15天，防止近期波动干扰锚点选择），找到最高点作为起点
        candidates_anchor = [p for p in pivots_high if p[2] < curr_idx - 15]
        
        if candidates_anchor:
            # 找到全局最高点作为锚点 (这是你画蓝线的起点)
            anchor = max(candidates_anchor, key=lambda x: x[1])
            idx_1, y1 = anchor[2], anchor[1]
            
            best_line = None
            max_slope = -float('inf') # 我们要找斜率最大（也就是最接近0，下降最慢）的线
            
            # 步骤 B: 寻找第二个点 (Target)
            # 遍历锚点之后的所有 Pivot
            for target in pivots_high:
                idx_2, y2 = target[2], target[1]
                
                if idx_2 <= idx_1 + 10: continue # 距离太近不连
                
                # 计算斜率 m 和 截距 c
                m = (y2 - y1) / (idx_2 - idx_1)
                c = y1 - m * idx_1
                
                # 只画下降趋势线 (阻力线通常向下)
                if m > 0: continue
                
                # --- 关键修正：严格的“不切穿”检查 ---
                # 检查锚点和目标点之间，是否有任何 K 线的高点显著超出了这条线
                # 只要中间有一个点“冒头”，这条线就作废
                is_valid = True
                
                # 提取中间段的 High 数据进行批量比对 (比循环快)
                check_start = idx_1 + 1
                check_end = idx_2 - 1
                
                if check_end > check_start:
                    subset_highs = df['high'].iloc[check_start:check_end+1].values
                    subset_indices = np.arange(check_start, check_end+1)
                    line_vals = m * subset_indices + c
                    
                    # [修改点] 允许 2% 的容错 (1.02)
                    if np.any(subset_highs > line_vals * 1.02):
                        is_valid = False
                
                if is_valid:
                    # 步骤 C: 优选逻辑
                    # 在所有“合法”的不切穿线中，选择斜率最大（最平缓）的那条
                    # 这样能保证线是架在所有价格之上的“最外层”
                    if m > max_slope:
                        max_slope = m
                        best_line = (m, c, idx_1, idx_2)
            
            # 步骤 D: 生成阻力线数据
            if best_line:
                m, c, idx_1, idx_2 = best_line
                
                # 延伸线段：从视野起点 到 现在的预测点
                p_start = m * vis_start_idx + c
                p_end = m * curr_idx + c
                
                res_line = [[(t_start, p_start), (t_end, p_end)]]
                
                # [新增] 更新 min_anchor_idx
                if min_anchor_idx is None or idx_1 < min_anchor_idx:
                    min_anchor_idx = idx_1

                # --- 检测突破 ---
                curr_price = df['close'].iloc[-1]
                # 突破必须是当前价格高于线
                if curr_price > p_end:
                     pattern_name = "趋势突破 (由守转攻)"

    # --- 2. 支撑线 (Support) - 保持原有逻辑 (寻找底部切线) ---
    if pivots_low:
        candidates_anchor_low = [p for p in pivots_low if p[2] < curr_idx - 15]
        if candidates_anchor_low:
            anchor_low = min(candidates_anchor_low, key=lambda x: x[1])
            lx1, ly1 = anchor_low[2], anchor_low[1]
            
            best_sup_line = None
            
            for target in pivots_low:
                lx2, ly2 = target[2], target[1]
                if lx2 <= lx1 + 10: continue
                
                m_sup = (ly2 - ly1) / (lx2 - lx1)
                c_sup = ly1 - m_sup * lx1
                
                is_valid_sup = True
                check_start = lx1 + 1
                check_end = lx2 - 1
                
                if check_end > check_start:
                    subset_lows = df['low'].iloc[check_start:check_end+1].values
                    subset_indices = np.arange(check_start, check_end+1)
                    line_vals = m_sup * subset_indices + c_sup
                    # [修改点] 允许 2% 的容错 (0.98)
                    if np.any(subset_lows < line_vals * 0.98):
                        is_valid_sup = False
                
                if is_valid_sup:
                    best_sup_line = (m_sup, c_sup)

            if best_sup_line:
                m_sup, c_sup = best_sup_line
                lp_start = m_sup * vis_start_idx + c_sup
                lp_end = m_sup * curr_idx + c_sup
                sup_line = [[(t_start, lp_start), (t_end, lp_end)]]
                
                # [新增] 更新 min_anchor_idx
                if min_anchor_idx is None or lx1 < min_anchor_idx:
                    min_anchor_idx = lx1

    return pattern_name, res_line, sup_line, min_anchor_idx

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
        # [修改] 简化名称
        patterns.append("锤子线")

    return patterns

def get_volume_projection_factor(ny_now, minutes_elapsed):
    TOTAL_MINUTES = 390
    if minutes_elapsed <= 10: return 13.0
    elif minutes_elapsed <= 60: return 13.0 - (13.0 - 8.0) * (minutes_elapsed - 10) / 50
    else: return 8.0 - (8.0 - 4.0) * (minutes_elapsed - 60) / (TOTAL_MINUTES - 60)

def calculate_risk_levels(df):
    """
    返回 (stop_loss, support)
    """
    curr_close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else curr_close * 0.05
      
    stop_loss = curr_close - (2.8 * atr)
      
    _, pivots_low = find_pivots(df, window=5)
    support = stop_loss
      
    if pivots_low:
        last_pivot_low = pivots_low[-1][1]
        if last_pivot_low < curr_close:
             support = last_pivot_low
             
    return stop_loss, support

# --- 核心信号检查函数 ---
# [修改] 增加返回值 anchor_idx
def check_signals_sync(df):
    if len(df) < 60: return False, 0, "数据不足", [], [], None
      
    last_date = df.index[-1].date()
    today_date = datetime.now(MARKET_TIMEZONE).date()
      
    if (today_date - last_date).days > 4:
        return False, 0, f"DATA_STALE: 数据严重滞后 ({last_date})", [], [], None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    score = 0
    weights = CONFIG["SCORE"]["WEIGHTS"]
    params = CONFIG["SCORE"]["PARAMS"] # [新增] 引用配置参数
    violations = [] 

    low_60 = df['low'].tail(60).min()
    high_60 = df['high'].tail(60).max()
      
    if curr['close'] > low_60 * (1 + CONFIG["filter"]["max_60d_gain"]): 
        # [修改] 名称统一为 过滤器
        violations.append("过滤器: 短期涨幅过大")
        
    prev_close_safe = prev['close'] if prev['close'] > 0 else 1.0
    day_gain = (curr['close'] - prev['close']) / prev_close_safe

    if abs(day_gain) > CONFIG["filter"]["max_day_change"]: 
        # [修改] 名称统一为 过滤器
        violations.append("过滤器: 单日波动过大")
        
    if curr['RSI'] > CONFIG["filter"]["max_rsi"]: 
        # [修改] 名称统一为 过滤器
        violations.append("过滤器: RSI严重超买")
      
    if curr['BIAS_50'] > CONFIG["filter"]["max_bias_50"]:
        # [修改] 名称统一为 过滤器
        violations.append("过滤器: 乖离率过大")

    if curr['Upper_Shadow_Ratio'] > CONFIG["filter"]["max_upper_shadow"]:
        # [修改] 名称统一为 过滤器
        violations.append("过滤器: 长上影线压力")

    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
    is_open_market = 0 < minutes_elapsed < 390
      
    is_volume_ok = False
    proj_vol_final = curr['volume']
      
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
        if curr['volume'] >= curr['Vol_MA20'] * CONFIG["filter"]["min_vol_ratio"]:
            is_volume_ok = True
            
    if not is_volume_ok:
        # [修改] 名称统一为 过滤器
        violations.append("过滤器: 量能不足")

    # [修改] 使用配置参数
    if proj_vol_final > curr['Vol_MA20'] * params["heavy_vol_multiplier"]:
        score += weights["HEAVY_VOLUME"]

    candle_patterns = detect_candle_patterns(df)
    if candle_patterns:
        triggers.append(f"K线: {', '.join(candle_patterns)}")
        score += weights["CANDLE_PATTERN"]

    bb_min_width = CONFIG["filter"]["min_bb_squeeze_width"]
    bb_target_width = CONFIG["filter"]["min_bb_expand_width"]
    max_pos = CONFIG["filter"]["max_bottom_pos"]
      
    if high_60 > low_60:
        price_pos = (curr['close'] - low_60) / (high_60 - low_60)
    else:
        price_pos = 0.5
      
    if prev['BB_Width'] < bb_min_width: 
        if curr['BB_Width'] >= bb_target_width: 
            if curr['close'] > curr['open']: 
                 if price_pos <= max_pos: 
                    triggers.append(f"BB Squeeze: 低位启动 (宽:{curr['BB_Width']:.3f}, 位:{price_pos:.2f})")
                    score += weights["BB_SQUEEZE"]

    # [修改] 使用配置参数
    is_strong_trend = curr['ADX'] > params["adx_strong_threshold"] and curr['PDI'] > curr['MDI']
    is_adx_rising = curr['ADX'] > prev['ADX']
      
    if is_strong_trend and is_adx_rising:
        score += weights["STRONG_ADX"]
        
    recent_adx_min = df['ADX'].iloc[-10:-1].min()
    # [修改] 使用配置参数
    adx_activating = (recent_adx_min < params["adx_activation_lower"]) and \
                      (df['ADX'].iloc[-1] > df['ADX'].iloc[-2] > df['ADX'].iloc[-3])
                      
    if adx_activating:
        triggers.append(f"趋势激活: 盘整结束 ADX拐头")
        score += weights["ADX_ACTIVATION"]

    had_breakout = (df['close'].tail(10) > df['Nx_Blue_UP'].tail(10)).any()
    on_support = (curr['low'] >= curr['Nx_Blue_DW'] * 0.99) and (curr['close'] > curr['Nx_Blue_DW'])
      
    if is_strong_trend and had_breakout and on_support and (curr['close'] > curr['open']):
        triggers.append("Nx 结构: 蓝梯回踩确认")
        score += weights["GOD_TIER_NX"] 

    # [修改] 接收 min_anchor_idx
    pattern_name, res_line, sup_line, anchor_idx = identify_patterns(df)
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

    # [修改] 使用配置参数
    if prev['J'] < params["kdj_j_oversold"] and curr['J'] > 0 and curr['K'] > curr['D']:
        triggers.append(f"KDJ 反击")
        score += weights["KDJ_REBOUND"]
      
    price_low_20 = df['close'].tail(20).min()
    # [修改] 使用配置参数
    price_is_low = curr['close'] <= price_low_20 * params["divergence_price_tolerance"]
    macd_low_20 = df['MACD'].tail(20).min()
    if price_is_low and curr['MACD'] < 0:
        # [修改] 使用配置参数
        if curr['MACD'] > macd_low_20 * params["divergence_macd_strength"] and curr['DIF'] > df['DIF'].tail(20).min():
             triggers.append(f"MACD 底背离")
             score += weights["MACD_DIVERGE"]

    if curr['OBV'] > curr['OBV_MA20']:
        # [修改] 使用配置参数
        obv_lookback = params["obv_lookback"]
        obv_rising = curr['OBV'] > df['OBV'].iloc[-obv_lookback]
        if obv_rising and curr['close'] > curr['open']:
             triggers.append("资金面: OBV趋势向上 (资金流入)")
             score += weights["OBV_TREND_UP"]

    pinbar_ratio = (curr['close'] - curr['low']) / (curr['high'] - curr['low'] + 1e-9)
    market_cap = df.attrs.get('marketCap', float('inf')) 
      
    if curr['low'] < curr['BB_Low']:
        # [修改] 使用配置参数
        if proj_vol_final > curr['Vol_MA20'] * params["capitulation_vol_mult"]:
            if pinbar_ratio > params["capitulation_pinbar"] and market_cap < params["capitulation_mcap"]:
                triggers.append(f"抛售高潮")
                score += weights["CAPITULATION"]

    is_triggered = (score >= CONFIG["SCORE"]["MIN_ALERT_SCORE"]) and (len(violations) == 0)
      
    final_reason_parts = triggers + violations
    final_reason = "\n".join(final_reason_parts) if final_reason_parts else "无明显信号"

    return is_triggered, score, final_reason, res_line, sup_line, anchor_idx

async def check_signals(df):
    return await asyncio.to_thread(check_signals_sync, df)

# [修改] 增加参数 anchor_idx, 增加裁剪逻辑, 增加右侧留白, 移除副图, 美化样式, 增加布林中轨
def _generate_chart_sync(df, ticker, res_line=[], sup_line=[], stop_price=None, support_price=None, anchor_idx=None):
    buf = io.BytesIO()
    
    # --- 1. 数据准备与切片 ---
    # 默认回溯天数
    default_lookback = 80
    start_idx = max(0, len(df) - default_lookback)
      
    # 如果有确切的锚点，优先使用它来定位左边界
    if anchor_idx is not None:
        start_idx = max(0, anchor_idx - 20)

    plot_df = df.iloc[start_idx:].copy()
    
    # --- 2. 右侧留白逻辑 ---
    # 生成未来 20 个工作日的时间索引
    last_date = plot_df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=20)
    
    # 创建一个全空的 DataFrame 用于占位
    future_df = pd.DataFrame(index=future_dates, columns=plot_df.columns)
    
    # 合并数据
    plot_df = pd.concat([plot_df, future_df])

    # --- 3. 辅助线数据准备 ---
    valid_len = len(df.iloc[start_idx:]) 
    total_len = len(plot_df)            
    
    if stop_price is None: stop_price = df['close'].iloc[-1] * 0.95
    if support_price is None: support_price = df['close'].iloc[-1] * 0.90
    
    stop_line_data = [stop_price] * total_len
    supp_line_data = [support_price] * total_len

    # --- 4. 裁剪趋势线 ---
    def clip_line_segments(segments):
        new_segments = []
        if not segments: return new_segments
        plot_start_date = plot_df.index[0]
        
        for seg in segments:
            d1, p1 = seg[0]
            d2, p2 = seg[1]
            if d2 < plot_start_date: continue
            if d1 < plot_start_date:
                try:
                    new_segments.append(seg) 
                except: continue
            else:
                new_segments.append(seg)
        return new_segments

    res_line_clipped = clip_line_segments(res_line)
    sup_line_clipped = clip_line_segments(sup_line)

    # --- 5. 配置 addplot (含 Nx结构, Boll三轨, 止损支撑) ---
    add_plots = [
        # Nx 结构 (实线)
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        
        # [新增] 布林带三轨 (淡紫色虚线/点线，不干扰主视线)
        mpf.make_addplot(plot_df['BB_Up'], color='#9370DB', linestyle='-.', width=0.8, alpha=0.7),
        # 中轨：使用更轻的点线，辅助判断中枢
        mpf.make_addplot(plot_df['BB_Mid'], color='#9370DB', linestyle=':', width=0.6, alpha=0.6), 
        mpf.make_addplot(plot_df['BB_Low'], color='#9370DB', linestyle='-.', width=0.8, alpha=0.7),
        
        # 止损/支撑 (延伸到未来)
        mpf.make_addplot(stop_line_data, color='red', linestyle='--', width=1.0, alpha=0.6), 
        mpf.make_addplot(supp_line_data, color='green', linestyle=':', width=1.0, alpha=0.6), 
    ]
    
    # --- 6. 审美美化 ---
    my_marketcolors = mpf.make_marketcolors(
        up='#ff333a',      
        down='#00b060',    
        edge={'up': '#ff333a', 'down': '#00b060'}, 
        wick={'up': '#ff333a', 'down': '#00b060'}, 
        volume='in'        
    )
    
    my_style = mpf.make_mpf_style(
        base_mpl_style="seaborn-v0_8-whitegrid", 
        marketcolors=my_marketcolors,
        gridstyle=':',     
        gridcolor='#e0e0e0', 
        gridaxis='both',
        facecolor='white', 
        rc={
            'font.family': 'sans-serif', 
            'axes.labelcolor': 'grey',
            'xtick.labelcolor': 'grey',
            'ytick.labelcolor': 'grey',
            'axes.edgecolor': '#f0f0f0' 
        }
    )

    # 准备趋势线
    seq_of_points = []
    if res_line_clipped:
        for line in res_line_clipped:
            seq_of_points.append([(line[0][0], float(line[0][1])), (line[1][0], float(line[1][1]))])
    if sup_line_clipped:
        for line in sup_line_clipped:
            seq_of_points.append([(line[0][0], float(line[0][1])), (line[1][0], float(line[1][1]))])

    kwargs = dict(
        type='candle', 
        style=my_style, 
        title=dict(title=f"{ticker} Analysis", color='black', fontsize=15),
        ylabel='', 
        addplot=add_plots, 
        volume=False, # 关闭成交量
        tight_layout=True, 
        datetime_format='%m-%d', 
        xrotation=0, 
        figsize=(10, 6), 
        savefig=dict(fname=buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=120)
    )
      
    if seq_of_points:
        kwargs['alines'] = dict(
            alines=seq_of_points,
            colors='gray', 
            linestyle='--', 
            linewidths=1.2,
            alpha=0.6
        )
      
    try:
        mpf.plot(plot_df, **kwargs)
        buf.seek(0)
    finally:
        plt.close('all')
        
    return buf

# [修改] 增加参数 anchor_idx
async def generate_chart(df, ticker, res_line=[], sup_line=[], stop_price=None, support_price=None, anchor_idx=None):
    return await asyncio.to_thread(_generate_chart_sync, df, ticker, res_line, sup_line, stop_price, support_price, anchor_idx)

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
            need_10d = data.get("ret_10d") is None and (today - signal_date).days > 10
            need_20d = data.get("ret_20d") is None and (today - signal_date).days > 20
            if need_1d or need_5d or need_10d or need_20d: symbols_to_check.add(ticker)
            
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
                
                # 1D
                if data.get("ret_1d") is None and len(after_signal) >= 1:
                    price_1d = after_signal.iloc[0]['close']
                    data["ret_1d"] = round(((price_1d - signal_price) / signal_price) * 100, 2)
                    updates_made = True
                
                # 5D
                if data.get("ret_5d") is None and len(after_signal) >= 5:
                    price_5d = after_signal.iloc[4]['close'] 
                    data["ret_5d"] = round(((price_5d - signal_price) / signal_price) * 100, 2)
                    updates_made = True
                    
                # 10D
                if data.get("ret_10d") is None and len(after_signal) >= 10:
                    price_10d = after_signal.iloc[9]['close'] 
                    data["ret_10d"] = round(((price_10d - signal_price) / signal_price) * 100, 2)
                    updates_made = True

                # 20D
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

def create_alert_embed(ticker, score, price, reason, stop_loss, support, df, filename):
    level_str = get_level_by_score(score)
    # [修改] 检查 "过滤器" 关键字
    if "过滤器" in reason or "STALE" in reason:
        color = 0x95a5a6 
    else:
        color = 0x00ff00 if score >= 80 else 0x3498db
      
    embed = discord.Embed(title=f"🚨{ticker} 抄底信号 | 得分 {score}", color=color)
    embed.description = f"**现价:** `${price:.2f}`"
      
    curr = df.iloc[-1]
      
    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
      
    vol_label = "**量比 (预估):**"
    vol_ratio = 0.0
      
    if 0 < minutes_elapsed < 390:
        proj_factor = get_volume_projection_factor(ny_now, max(minutes_elapsed, 1))
        projected_vol = curr['volume'] * proj_factor
        vol_ratio = projected_vol / df['Vol_MA20'].iloc[-1]
    else:
        vol_ratio = curr['volume'] / df['Vol_MA20'].iloc[-1]
      
    obv_status = "流入" if curr['OBV'] > curr['OBV_MA20'] else "流出"
    
    indicator_text = (
        f"**RSI(14):** `{curr['RSI']:.1f}`\n"
        f"**ADX:** `{curr['ADX']:.1f}`\n"
        f"{vol_label} `{vol_ratio:.1f}x`\n" 
        f"**OBV:** `{obv_status}`\n"
        f"**Bias(50):** `{curr['BIAS_50']*100:.1f}%`"
    )
    embed.add_field(name="\u200b", value=indicator_text, inline=True)
      
    risk_text = (
        f"**止损价:** `${stop_loss:.2f}`\n"
        f"**支撑位:** `${support:.2f}`\n"
    )
    embed.add_field(name="\u200b", value=risk_text, inline=True)
      
    embed.add_field(name="\u200b", value=f"```{reason}```", inline=False)
    embed.set_image(url=f"attachment://{filename}")
      
    return embed

class StockBotClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.alert_channel = None
        self.last_report_date = None

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
        
        if not self.scheduled_report.is_running():
            self.scheduled_report.start()
            
        await self.tree.sync()

    async def send_daily_stats_report(self):
        if not self.alert_channel: return
        
        logging.info("Generating daily backtest report...")
        await update_stats_data()
        load_settings()
        
        history = settings.get("signal_history", {})
        market_df = await fetch_market_index_data(days=80)

        def get_market_ret(date_str, offset_days):
            if market_df is None or market_df.empty: return None
            try:
                target_date = pd.to_datetime(date_str).normalize()
                idx = market_df.index.get_indexer([target_date], method='nearest')[0]
                
                if idx + offset_days < len(market_df):
                    p_start = market_df.iloc[idx]['price']
                    p_end = market_df.iloc[idx + offset_days]['price']
                    return ((p_end - p_start) / p_start) * 100
            except: pass
            return None

        stats_agg = {k: {"s_sum": 0.0, "s_c": 0, "m_sum": 0.0, "m_c": 0, "w": 0} for k in ["1d", "5d", "10d", "20d"]}
        
        valid_signals = []
        if history:
            sorted_dates = sorted(history.keys(), reverse=True)
            today = datetime.now().date()
            
            for date_str in sorted_dates:
                try:
                    sig_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except: continue
                if (today - sig_date).days > 25: continue 
                
                tickers_data = history[date_str]
                for ticker, data in tickers_data.items():
                    if data.get("score", 0) == 0: continue 
                    
                    score = data.get("score", 0)
                    valid_signals.append((date_str, ticker, score, data))
                    
                    for k, days_off in [("1d", 1), ("5d", 5), ("10d", 10), ("20d", 20)]:
                        m = get_market_ret(date_str, days_off)
                        if m is not None:
                            stats_agg[k]["m_sum"] += m
                            stats_agg[k]["m_c"] += 1
                        
                        r = data.get(f"ret_{k}")
                        if r is not None:
                            stats_agg[k]["s_sum"] += r
                            stats_agg[k]["s_c"] += 1
                            if r > 0: stats_agg[k]["w"] += 1
        else:
            if market_df is not None and not market_df.empty:
                pass

        embed = discord.Embed(title="回测统计", color=0x9b59b6)
        
        def mk_field(key):
            d = stats_agg[key]
            
            if d["s_c"] > 0:
                avg_stock = d["s_sum"] / d["s_c"]
                avg_stock_str = f"`{avg_stock:+.2f}%`"
                win_rate = f"`{d['w']/d['s_c']*100:.0f}%`"
            else:
                avg_stock = None
                avg_stock_str = "Wait..."
                win_rate = "-"

            if d["m_c"] > 0:
                avg_market = d["m_sum"] / d["m_c"]
                avg_market_str = f"`{avg_market:+.2f}%`"
            else:
                if d["s_c"] == 0 and market_df is not None and not market_df.empty:
                    try:
                        days_offset = int(key[:-1])
                        if len(market_df) > days_offset:
                            p_now = market_df.iloc[-1]['price']
                            p_prev = market_df.iloc[-(days_offset+1)]['price']
                            val = ((p_now - p_prev) / p_prev) * 100
                            avg_market = val
                            avg_market_str = f"`{val:+.2f}%`"
                        else:
                            avg_market = None
                            avg_market_str = "Wait..."
                    except:
                        avg_market = None
                        avg_market_str = "Wait..."
                else:
                    avg_market = None
                    avg_market_str = "Wait..."

            if avg_stock is not None and avg_market is not None and isinstance(avg_market, float):
                diff = avg_stock - avg_market
                diff_str = f"**{diff:+.2f}%**"
            else:
                diff_str = "-"
            
            return f"个股平均: {avg_stock_str}\n纳指同期: {avg_market_str}\n超额收益: {diff_str}\n个股胜率: {win_rate}"

        embed.add_field(name="1日表现", value=mk_field("1d"), inline=True)
        embed.add_field(name="5日表现", value=mk_field("5d"), inline=True)
        embed.add_field(name="10日表现", value=mk_field("10d"), inline=True)
        embed.add_field(name="20日表现", value=mk_field("20d"), inline=True)
        
        recent_list_str = []
        for date_str, ticker, score, data in valid_signals[:5]:
            r1 = data.get("ret_1d")
            r1_str = f"{r1:+.1f}%" if r1 is not None else "-"
            r5 = data.get("ret_5d")
            r5_str = f"{r5:+.1f}%" if r5 is not None else "-"
            r10 = data.get("ret_10d")
            r10_str = f"{r10:+.1f}%" if r10 is not None else "-"
            r20 = data.get("ret_20d")
            r20_str = f"{r20:+.1f}%" if r20 is not None else "-"
            
            recent_list_str.append(f"`{date_str}` **{ticker}**\n└ 1D:`{r1_str}` 5D:`{r5_str}` 10D:`{r10_str}` 20D:`{r20_str}`")
        
        if recent_list_str:
            embed.add_field(name="详细情况", value="\n".join(recent_list_str), inline=False)
        else:
            embed.add_field(name="详细情况", value="无近期信号", inline=False)

        embed.set_footer(text=f"Report generated at {datetime.now(MARKET_TIMEZONE).strftime('%H:%M:%S')} ET")
        await self.alert_channel.send(embed=embed)

    @tasks.loop(minutes=1)
    async def scheduled_report(self):
        now_et = datetime.now(MARKET_TIMEZONE)
        if now_et.hour == 16 and now_et.minute == 30:
            today_date = now_et.date()
            if self.last_report_date != today_date:
                await self.send_daily_stats_report()
                self.last_report_date = today_date

    @tasks.loop(minutes=5)
    async def monitor_stocks(self):
        if not self.alert_channel: return
        now_et = datetime.now(MARKET_TIMEZONE)
        curr_time, today_str = now_et.time(), now_et.strftime('%Y-%m-%d')
        
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
            
            if df is None or df.empty: continue

            user_ids = ticker_user_map[ticker]
            all_alerted = True
            users_to_ping = []
            for uid in user_ids:
                status_key = f"{ticker}-{today_str}"
                status = users_data[uid]['daily_status'].get(status_key, "NONE")
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
            
            # [修改] 解包 min_anchor_idx
            is_triggered, score, reason, res_line, sup_line, anchor_idx = await check_signals(df)
            
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
                
                stop_loss, support = calculate_risk_levels(df)
                
                alert_obj = {
                    "ticker": ticker,
                    "score": score, 
                    "priority": score, 
                    "price": price,
                    "reason": reason,
                    "support": support,
                    "stop_loss": stop_loss,
                    "df": df,
                    "res_line": res_line,
                    "sup_line": sup_line,
                    "anchor_idx": anchor_idx, # [新增] 保存 anchor_idx
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
                    "ret_10d": current_hist.get("ret_10d"),
                    "ret_20d": current_hist.get("ret_20d"),
                }
                
                for uid in users:
                    status_key = f"{ticker}-{today_str}"
                    users_data[uid]['daily_status'][status_key] = "MARKET_SENT"
                
                mentions = " ".join([f"<@{uid}>" for uid in users])
                
                if sent_charts < max_charts:
                    # [修改] 传递 anchor_idx
                    chart_buf = await generate_chart(
                        alert["df"], ticker, alert["res_line"], alert["sup_line"], 
                        alert["stop_loss"], alert["support"], alert["anchor_idx"]
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

@client.tree.command(name="reset_stats", description="Reset all backtest statistics")
async def reset_stats(interaction: discord.Interaction):
    global settings
    settings["signal_history"] = {}
    save_settings()
    await interaction.response.send_message("Statistics reset.", ephemeral=True)

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

@client.tree.command(name="stats", description="View historical signal accuracy (20-day window)")
async def stats_command(interaction: discord.Interaction):
    await interaction.response.defer()
    
    await update_stats_data()
    
    load_settings()
    history = settings.get("signal_history", {})
    market_df = await fetch_market_index_data(days=80)

    def get_market_ret(date_str, offset_days):
        if market_df is None or market_df.empty: return None
        try:
            target_date = pd.to_datetime(date_str).normalize()
            idx = market_df.index.get_indexer([target_date], method='nearest')[0]
            if idx + offset_days < len(market_df):
                p_start = market_df.iloc[idx]['price']
                p_end = market_df.iloc[idx + offset_days]['price']
                return ((p_end - p_start) / p_start) * 100
        except:
            pass
        return None

    stats_agg = {
        k: {"s_sum": 0.0, "s_c": 0, "m_sum": 0.0, "m_c": 0, "w": 0} 
        for k in ["1d", "5d", "10d", "20d"]
    }
    
    seen_tickers = set()
    valid_signals = []
    
    sorted_dates = sorted(history.keys(), reverse=True)
    today = datetime.now().date()
    
    for date_str in sorted_dates:
        try:
            sig_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except: continue
        
        days_diff = (today - sig_date).days
        if days_diff > 25: continue
        
        tickers_data = history[date_str]
        for ticker, data in tickers_data.items():
            if data.get("score", 0) == 0: continue

            if ticker in seen_tickers: continue
            seen_tickers.add(ticker)
            
            score = data.get("score", 0)
            valid_signals.append((date_str, ticker, score, data))
            
            for k, days_off in [("1d", 1), ("5d", 5), ("10d", 10), ("20d", 20)]:
                m = get_market_ret(date_str, days_off) 
                if m is not None:
                    stats_agg[k]["m_sum"] += m
                    stats_agg[k]["m_c"] += 1

                r = data.get(f"ret_{k}")
                if r is not None:
                    stats_agg[k]["s_sum"] += r
                    stats_agg[k]["s_c"] += 1
                    if r > 0: stats_agg[k]["w"] += 1

    embed = discord.Embed(title="回测统计", color=0x00BFFF)
    
    def mk_field(key):
        d = stats_agg[key]
        
        if d["s_c"] > 0:
            avg_stock = d["s_sum"] / d["s_c"]
            avg_stock_str = f"`{avg_stock:+.2f}%`"
            win_rate = f"`{d['w']/d['s_c']*100:.0f}%`"
        else:
            avg_stock = None
            avg_stock_str = "Wait..."
            win_rate = "-"

        if d["m_c"] > 0:
            avg_market = d["m_sum"] / d["m_c"]
            avg_market_str = f"`{avg_market:+.2f}%`"
        else:
            if d["s_c"] == 0 and market_df is not None and not market_df.empty:
                try:
                    days_offset = int(key[:-1])
                    if len(market_df) > days_offset:
                        p_now = market_df.iloc[-1]['price']
                        p_prev = market_df.iloc[-(days_offset+1)]['price']
                        val = ((p_now - p_prev) / p_prev) * 100
                        avg_market = val
                        avg_market_str = f"`{val:+.2f}%`"
                    else:
                        avg_market = None
                        avg_market_str = "Wait..."
                except:
                    avg_market = None
                    avg_market_str = "Wait..."
            else:
                avg_market = None
                avg_market_str = "Wait..."
        
        if avg_market is not None and isinstance(avg_market, float):
            avg_market_str = f"`{avg_market:+.2f}%`"
        else:
            avg_market_str = "Wait..."

        if avg_stock is not None and avg_market is not None and isinstance(avg_market, float):
            diff = avg_stock - avg_market
            diff_str = f"**{diff:+.2f}%**"
        else:
            diff_str = "-"
        
        return (
            f"个股平均: {avg_stock_str}\n"
            f"纳指同期: {avg_market_str}\n"
            f"超额收益: {diff_str}\n"
            f"个股胜率: {win_rate}"
        )

    embed.add_field(name="1日表现", value=mk_field("1d"), inline=True)
    embed.add_field(name="5日表现", value=mk_field("5d"), inline=True)
    embed.add_field(name="10日表现", value=mk_field("10d"), inline=True)
    embed.add_field(name="20日表现", value=mk_field("20d"), inline=True)

    recent_list_str = []
    for date_str, ticker, score, data in valid_signals[:10]:
        r1 = data.get("ret_1d")
        r1_str = f"{r1:+.1f}%" if r1 is not None else "-"
        r5 = data.get("ret_5d")
        r5_str = f"{r5:+.1f}%" if r5 is not None else "-"
        r10 = data.get("ret_10d")
        r10_str = f"{r10:+.1f}%" if r10 is not None else "-"
        r20 = data.get("ret_20d")
        r20_str = f"{r20:+.1f}%" if r20 is not None else "-"
        
        recent_list_str.append(f"`{date_str}` **{ticker}**\n└ 1D:`{r1_str}` 5D:`{r5_str}` 10D:`{r10_str}` 20D:`{r20_str}`")
        
    if recent_list_str:
        embed.add_field(name="详细情况", value="\n".join(recent_list_str), inline=False)
    else:
        embed.add_field(name="详细情况", value="无近期信号", inline=False)
        
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

    # [修改] 解包 anchor_idx
    is_triggered, score, reason, r_l, s_l, anchor_idx = await check_signals(df)
    
    price = df['close'].iloc[-1]
    
    stop_loss, support = calculate_risk_levels(df)

    if not reason: 
        reason = f"无明显信号 (得分: {score})"
    
    # [修改] 传递 anchor_idx
    chart_buf = await generate_chart(df, ticker, r_l, s_l, stop_loss, support, anchor_idx)
    filename = f"{ticker}_test.png"
    
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
