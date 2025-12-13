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
import warnings

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

# --- 核心策略配置 (RVOL 加强版 + 四维共振 + 动态布林) ---
CONFIG = {
    # [1] 过滤器：左侧抄底核心 (一票否决制)
    "filter": {
        "max_60d_gain": 0.3,          # [防追高] 过去60天涨幅超过 30% 则不看
        "max_rsi": 60,                # [防过热] RSI(14) 超过 60 则不看
        "max_bias_50": 0.20,          # [防回落] 现价偏离 50日均线 20% 以上不看
        "max_upper_shadow": 0.4,      # [防抛压] 上影线长度占整根K线 40% 以上不看
        "max_day_change": 0.7,        # [防妖股] 单日涨跌幅超过 70% 不看
        
        "min_rvol": 1.2,              # [核心] RVOL 必须 > 1.2 (比历史同期活跃20%以上)
        
        # [布林带动态配置 - 修改部分]
        "min_bb_squeeze_width": 0.10, # [前置条件] 昨日带宽需小于此值 (定义什么是"窄")
        "bb_expansion_rate": 1.2,     # [动态扩张] 今天带宽 / 昨天带宽 >= 1.2 (即扩大20%才算开口)
        
        "max_bottom_pos": 0.30,       # [位置] 价格在过去60天区间的位置 (0.3表示底部30%)
        "min_adx_for_squeeze": 15     # [趋势] ADX 最小门槛，确保不是死水
    },

    # [2] 形态识别
    "pattern": {
        "pivot_window": 10            # [关键点] 识别高低点的前后窗口天数
    },

    # [3] 系统设置
    "system": {
        "cooldown_days": 3,           # [防刷屏] 发出信号后的冷却天数
        "max_charts_per_scan": 5,     # [防拥堵] 每次扫描最大发送图表数量
        "history_days": 300           # [数据源] 获取历史数据的天数
    },

    # [4] 打分系统
    "SCORE": { 
        "MIN_ALERT_SCORE": 70,        # [及格线] 总分低于此值不报警
        
        # [4.1] 四维共振设置
        "RESONANCE": {
            "window_days": 5,         # [窗口] 回溯过去 5 天寻找背离信号
            "min_signals": 2,         # [阈值] 至少需要 2 个指标同时背离才算共振
            "bonus_score": 25         # [加分] 达成共振后的奖励分数
        },

        # [4.2] 策略参数
        "PARAMS": {
            "rvol_heavy": 2.0,              # [机构] RVOL > 2.0 视为机构大单扫货
            "rvol_capitulation": 2.5,       # [恐慌] 恐慌抛售时的量能要求
            
            "adx_strong_threshold": 25,     # [趋势] ADX > 25 视为强趋势
            "adx_activation_lower": 20,     # [趋势] ADX < 20 视为盘整，用于判断启动
            "kdj_j_oversold": 0,            # [超卖] KDJ.J < 0 视为超卖
            "divergence_price_tolerance": 1.02, # [背离] 价格创新低容差
            "divergence_macd_strength": 0.8,    # [背离] MACD 柱子强度的容差
            "obv_lookback": 5,              # [资金] OBV 回溯对比天数
            "capitulation_pinbar": 0.5      # [K线] 针型K线判断阈值
        },

        # [4.3] 权重 (各项得分)
        "WEIGHTS": {
            # "4D_RESONANCE": 25,   # 由 CONFIG["RESONANCE"]["bonus_score"] 控制
            
            "PATTERN_BREAK": 40,    # [形态] 旗形突破 (最重要)
            "PATTERN_SUPPORT": 20,  # [形态] 旗形支撑回踩
            "BB_SQUEEZE": 35,       # [布林] 极度压缩后的开口
            "STRONG_ADX": 20,       # [趋势] 强趋势状态
            "ADX_ACTIVATION": 25,   # [趋势] 趋势从盘整中激活
            "OBV_TREND_UP": 15,     # [资金] OBV 持续向上
            
            "CAPITULATION": 25,     # [抄底] 恐慌盘涌出 (配合 RVOL 验证)
            "HEAVY_INSTITUTIONAL": 20, # [量能] 纯粹的机构异动 (高 RVOL)
            
            "MACD_ZERO_CROSS": 10,  # [指标] MACD 0轴金叉
            "MACD_DIVERGE": 15,     # [指标] MACD 底背离 (常规)
            "KDJ_REBOUND": 10,      # [指标] KDJ 超卖反弹
            "CANDLE_PATTERN": 5     # [K线] 吞没/晨星/锤子
        },

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
# [新增] RVOL 全局缓存
rvol_baseline_cache = {} 

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

# -----------------------------------------------------------------------------
# [核心] RVOL 计算器
# -----------------------------------------------------------------------------
class RVOLCalculator:
    """
    负责在开盘前或启动时，一次性计算股票的成交量基准线。
    """
    @staticmethod
    async def precalculate_baselines(symbols):
        global rvol_baseline_cache
        logging.info(f"Start pre-calculating RVOL baselines for {len(symbols)} tickers...")
        
        start_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        connector = aiohttp.TCPConnector(limit=5)
        semaphore = asyncio.Semaphore(5)
        
        async def fetch_intraday(session, sym):
            # FMP 5分钟 历史数据
            url = f"https://financialmodelingprep.com/stable/historical-chart/5min/{sym}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
            async with semaphore:
                retries = 3
                for i in range(retries):
                    try:
                        async with session.get(url, ssl=False) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                return sym, data
                            elif resp.status == 429:
                                await asyncio.sleep(2)
                                continue
                    except: pass
                    await asyncio.sleep(0.5)
            return sym, []

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch_intraday(session, sym) for sym in symbols]
            results = await asyncio.gather(*tasks)

        count_ok = 0
        for sym, data in results:
            if not data: continue
            
            try:
                df = pd.DataFrame(data)
                if 'date' not in df.columns or 'volume' not in df.columns: continue
                
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC').dt.tz_convert(MARKET_TIMEZONE)
                df['time_str'] = df['date'].dt.strftime('%H:%M')
                df['date_only'] = df['date'].dt.date
                
                df = df[(df['time_str'] >= '09:30') & (df['time_str'] <= '16:00')]
                
                df = df.sort_values('date')
                df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
                
                baseline = df.groupby('time_str')['cum_vol'].median()
                
                rvol_baseline_cache[sym] = baseline.to_dict()
                count_ok += 1
            except Exception as e:
                logging.error(f"Error processing RVOL for {sym}: {e}")
            
        logging.info(f"RVOL Baselines calculated for {count_ok} stocks.")

    @staticmethod
    def get_current_rvol(ticker, current_cum_vol, ny_time):
        if ticker not in rvol_baseline_cache:
            return 1.0 
            
        minute = ny_time.minute
        floored_minute = (minute // 5) * 5
        time_key = f"{ny_time.hour:02d}:{floored_minute:02d}"
        
        baseline_vol = rvol_baseline_cache[ticker].get(time_key)
        
        if not baseline_vol or baseline_vol == 0:
            return 1.0
            
        return current_cum_vol / baseline_vol

# --- 核心逻辑 (指标计算) ---
def calculate_indicators(df):
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
      
    df = df[df['close'] > 0]
      
    # MACD (一次性计算，后续逻辑直接复用)
    price_col = 'close'
    exp12 = df[price_col].ewm(span=12, adjust=False).mean()
    exp26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
      
    # RSI (14)
    delta = df[price_col].diff()
    gain = (delta.clip(lower=0)).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # [新增] 辅助 RSI 用于共振计算 (RSI6 和 RSI12)
    gain6 = (delta.clip(lower=0)).rolling(window=6).mean()
    loss6 = (-delta.clip(upper=0)).rolling(window=6).mean()
    rs6 = gain6 / loss6.replace(0, 1e-9)
    df['RSI6'] = 100 - (100 / (1 + rs6))
    
    gain12 = (delta.clip(lower=0)).rolling(window=12).mean()
    loss12 = (-delta.clip(upper=0)).rolling(window=12).mean()
    rs12 = gain12 / loss12.replace(0, 1e-9)
    df['RSI12'] = 100 - (100 / (1 + rs12))
      
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
      
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # BB
    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']

    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv_denom = (high_max - low_min).replace(0, 1e-9)
    df['RSV'] = (df['close'] - low_min) / rsv_denom * 100
    df['K'] = df['RSV'].ewm(com=2).mean() 
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # ADX
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

    # Bias
    df['MA50'] = df['close'].rolling(50).mean()
    ma50_safe = df['MA50'].replace(0, np.nan) 
    df['BIAS_50'] = (df['close'] - ma50_safe) / ma50_safe

    # Shadow
    candle_range = (df['high'] - df['low']).replace(0, 1e-9)
    upper_shadow = np.where(df['close'] >= df['open'], df['high'] - df['close'], df['high'] - df['open'])
    df['Upper_Shadow_Ratio'] = upper_shadow / candle_range

    # OBV
    obv_sign = np.sign(df['close'].diff()).fillna(0)
    df['OBV'] = (df['volume'] * obv_sign).cumsum()
    df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

    # [新增] 丝带指标 (视觉用, 不参与评分)
    df['Ribbon_Fast'] = df['close'].ewm(span=21, adjust=False).mean()
    df['Ribbon_Slow'] = df['close'].ewm(span=60, adjust=False).mean()

    # [新增] CCI (14) 计算
    # TYP = (H+L+C)/3, CCI = (TYP - MA_TYP) / (0.015 * AVEDEV)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['TP_MA'] = df['TP'].rolling(window=14).mean()
    # Pandas MAD is mean absolute deviation
    df['TP_MAD'] = df['TP'].rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (df['TP'] - df['TP_MA']) / (0.015 * df['TP_MAD'].replace(0, 1e-9))
    df['CCI_MA'] = df['CCI'].rolling(window=5).mean()

    # [新增] MFI (14) 计算
    # Money Flow = TP * Vol
    df['RawMF'] = df['TP'] * df['volume']
    df['PosMF'] = np.where(df['TP'] > df['TP'].shift(1), df['RawMF'], 0)
    df['NegMF'] = np.where(df['TP'] < df['TP'].shift(1), df['RawMF'], 0)
    df['PosMF_Sum'] = df['PosMF'].rolling(window=14).sum()
    df['NegMF_Sum'] = df['NegMF'].rolling(window=14).sum()
    mf_total = df['PosMF_Sum'] + df['NegMF_Sum']
    df['MFI'] = 100 * (df['PosMF_Sum'] / mf_total.replace(0, 1e-9))
    df['MFI_MA'] = df['MFI'].rolling(window=6).mean()

    return df

def process_dataframe_sync(hist_data):
    if not hist_data: return None
    df = pd.DataFrame(hist_data)
    if 'date' not in df.columns: return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index(ascending=True)
    return calculate_indicators(df)

def merge_and_recalc_sync(df, quote):
    if df is None or quote is None: return df
    try:
        quote_time = pd.to_datetime(quote['timestamp'], unit='s').tz_localize('UTC').tz_convert(MARKET_TIMEZONE)
        quote_date = quote_time.normalize().tz_localize(None) 
        last_idx = df.index[-1]
        last_date = pd.to_datetime(last_idx).normalize()
        if last_date.tzinfo is not None: last_date = last_date.tz_localize(None)

        current_price = quote['price']
        safe_high = max(quote['dayHigh'], current_price, quote['open'])
        safe_low = min(quote['dayLow'], current_price, quote['open'])

        new_row = {
            'open': quote['open'], 'high': safe_high, 'low': safe_low, 'close': current_price,
            'volume': quote['volume'], 'date': quote_date 
        }
        df_mod = df.copy()
        if last_date == quote_date:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col == 'high': df_mod.at[last_idx, col] = max(df_mod.at[last_idx, col], new_row[col])
                elif col == 'low': df_mod.at[last_idx, col] = min(df_mod.at[last_idx, col], new_row[col])
                else: df_mod.at[last_idx, col] = new_row[col]
        elif last_date < quote_date:
            new_df = pd.DataFrame([new_row])
            new_df = new_df.set_index('date')
            df_mod = pd.concat([df_mod, new_df])
        if 'marketCap' in quote:
            df_mod.attrs['marketCap'] = quote['marketCap']
        return calculate_indicators(df_mod)
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
    headers = {"User-Agent": "Mozilla/5.0 (StockBot/1.0)", "Accept": "application/json"}

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
                            if df is not None and not df.empty: results[sym] = df
                            else: logging.warning(f"[数据为空] {sym}")
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
    lookback_days = 250
    start_idx = max(0, len(df) - lookback_days)
    for i in range(start_idx + window, len(df) - window):
        is_high = True
        is_low = True
        current_high = highs[i]
        current_low = lows[i]
        for j in range(1, window + 1):
            if highs[i-j] >= current_high or highs[i+j] > current_high: is_high = False
            if lows[i-j] <= current_low or lows[i+j] < current_low: is_low = False
        if is_high: pivots_high.append((dates[i], current_high, i))
        if is_low: pivots_low.append((dates[i], current_low, i))
    return pivots_high, pivots_low

def identify_patterns(df):
    if len(df) < 60: return None, [], [], None, None, None # Modified to return slope/intercept
    pivots_high, pivots_low = find_pivots(df, window=5)
    res_line, sup_line = [], []
    pattern_name = None
    min_anchor_idx = None
    vis_start_idx = max(0, len(df) - 250)
    curr_idx = len(df) - 1
    t_start = df.index[vis_start_idx]
    t_end = df.index[curr_idx]
    
    # [新增] 支撑线参数
    sup_slope = None
    sup_intercept = None
      
    # --- 1. 阻力线 (Resistance) ---
    if pivots_high:
        candidates_anchor = [p for p in pivots_high if p[2] < curr_idx - 15]
        if candidates_anchor:
            anchor = max(candidates_anchor, key=lambda x: x[1])
            idx_1, y1 = anchor[2], anchor[1]
            best_line = None
            max_slope = -float('inf') 
            for target in pivots_high:
                idx_2, y2 = target[2], target[1]
                if idx_2 <= idx_1 + 10: continue 
                m = (y2 - y1) / (idx_2 - idx_1)
                c = y1 - m * idx_1
                if m > 0: continue
                is_valid = True
                check_start = idx_1 + 1
                check_end = idx_2 - 1
                if check_end > check_start:
                    subset_highs = df['high'].iloc[check_start:check_end+1].values
                    subset_indices = np.arange(check_start, check_end+1)
                    line_vals = m * subset_indices + c
                    if np.any(subset_highs > line_vals * 1.02): is_valid = False
                if is_valid:
                    if m > max_slope:
                        max_slope = m
                        best_line = (m, c, idx_1, idx_2)
            if best_line:
                m, c, idx_1, idx_2 = best_line
                p_start = m * vis_start_idx + c
                p_end = m * curr_idx + c
                res_line = [[(t_start, p_start), (t_end, p_end)]]
                if min_anchor_idx is None or idx_1 < min_anchor_idx: min_anchor_idx = idx_1
                curr_price = df['close'].iloc[-1]
                if curr_price > p_end: pattern_name = "趋势突破 (由守转攻)"

    # --- 2. 支撑线 (Support) ---
    if pivots_low:
        sorted_pivots = sorted(pivots_low, key=lambda x: x[1])
        potential_anchors = sorted_pivots[:5] 
        best_sup_line = None
        max_dist = 0 
        for anchor in potential_anchors:
            lx1, ly1 = anchor[2], anchor[1]
            targets = [p for p in pivots_low if p[2] > lx1 + 5]
            for target in targets:
                lx2, ly2 = target[2], target[1]
                m_sup = (ly2 - ly1) / (lx2 - lx1)
                c_sup = ly1 - m_sup * lx1
                if abs(m_sup) > 5: continue
                is_valid_sup = True
                check_start = lx1 + 1
                check_end = lx2 - 1
                if check_end > check_start:
                    subset_lows = df['low'].iloc[check_start:check_end+1].values
                    subset_indices = np.arange(check_start, check_end+1)
                    line_vals = m_sup * subset_indices + c_sup
                    if np.any(subset_lows < line_vals * 0.97): 
                        is_valid_sup = False
                if is_valid_sup:
                    dist = lx2 - lx1
                    if dist > max_dist:
                        max_dist = dist
                        best_sup_line = (m_sup, c_sup)
        if best_sup_line:
            m_sup, c_sup = best_sup_line
            # [新增] 保存斜率和截距，供后续逻辑判断使用
            sup_slope = m_sup
            sup_intercept = c_sup
            
            lp_start = m_sup * vis_start_idx + c_sup
            lp_end = m_sup * curr_idx + c_sup
            sup_line = [[(t_start, lp_start), (t_end, lp_end)]]
            if min_anchor_idx is None: min_anchor_idx = sorted_pivots[0][2] 

    return pattern_name, res_line, sup_line, min_anchor_idx, sup_slope, sup_intercept

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
    if lower_shadow > 2 * curr_body and upper_shadow < 0.5 * curr_body: patterns.append("锤子线")
    return patterns

def calculate_risk_levels(df):
    curr_close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else curr_close * 0.05
    stop_loss = curr_close - (2.8 * atr)
    _, pivots_low = find_pivots(df, window=5)
    support = stop_loss
    if pivots_low:
        last_pivot_low = pivots_low[-1][1]
        if last_pivot_low < curr_close: support = last_pivot_low
    return stop_loss, support

# -----------------------------------------------------------------------------
# [核心修改] 整合 RVOL 的信号检查
# -----------------------------------------------------------------------------
def check_signals_sync(df, ticker): # [修改] 传入 ticker
    if len(df) < 60: return False, 0, "数据不足", [], [], None, 1.0
    last_date = df.index[-1].date()
    today_date = datetime.now(MARKET_TIMEZONE).date()
    if (today_date - last_date).days > 4: return False, 0, f"DATA_STALE: 数据严重滞后 ({last_date})", [], [], None, 1.0

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    score = 0
    weights = CONFIG["SCORE"]["WEIGHTS"]
    params = CONFIG["SCORE"]["PARAMS"]
    violations = [] 
    
    # 过滤器 (左侧交易严格执行，不再豁免)
    low_60 = df['low'].tail(60).min()
    high_60 = df['high'].tail(60).max()
    
    if curr['close'] > low_60 * (1 + CONFIG["filter"]["max_60d_gain"]): violations.append("过滤器: 短期涨幅过大")
    
    prev_close_safe = prev['close'] if prev['close'] > 0 else 1.0
    day_gain = (curr['close'] - prev['close']) / prev_close_safe
    if abs(day_gain) > CONFIG["filter"]["max_day_change"]: violations.append("过滤器: 单日波动过大")
    
    if curr['RSI'] > CONFIG["filter"]["max_rsi"]: violations.append("过滤器: RSI严重超买")
    
    if curr['BIAS_50'] > CONFIG["filter"]["max_bias_50"]: violations.append("过滤器: 乖离率过大")
    if curr['Upper_Shadow_Ratio'] > CONFIG["filter"]["max_upper_shadow"]: violations.append("过滤器: 长上影线压力")

    # ============================================================
    # [修改逻辑开始] 提前计算形态，用于量能豁免判断
    # ============================================================
    pattern_name, res_line, sup_line, anchor_idx, sup_slope, sup_intercept = identify_patterns(df)
    
    # 判断是否处于“支撑线”附近 (为了给“底部休眠”发免死金牌)
    is_structure_support = False
    if sup_slope is not None:
        curr_idx = len(df) - 1
        curr_sup_price = sup_slope * curr_idx + sup_intercept
        # 如果当前价格在支撑线附近 (0.98 ~ 1.03)
        if 0.98 <= curr['close'] / curr_sup_price <= 1.03:
            is_structure_support = True
    # ============================================================

    # --- [关键] RVOL 计算与判定 ---
    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
    is_open_market = 0 < minutes_elapsed < 390
    
    # 获取实时 RVOL
    rvol = 1.0
    is_volume_ok = False
    
    if is_open_market and minutes_elapsed > 5:
        # 使用我们的 RVOLCalculator
        rvol = RVOLCalculator.get_current_rvol(ticker, curr['volume'], ny_now)
        
        if rvol >= CONFIG["filter"]["min_rvol"]:
            is_volume_ok = True
    else:
        # 盘前盘后或刚开盘，不看 RVOL
        is_volume_ok = True 
        
    if not is_volume_ok: 
        # [修改] 结构性豁免逻辑
        if is_structure_support:
            # 如果处于支撑位，允许缩量（底部休眠），不视为违规
            pass 
        else:
            # 既没放量，又没踩支撑，视为垃圾时间
            violations.append(f"过滤器: 资金不活跃 (RVOL {rvol:.2f} < 1.2)")
    
    # [评分] 机构大单扫货信号 (纯量能加分)
    if rvol > params["rvol_heavy"]:
        triggers.append(f"机构进场: 异常放量 (RVOL {rvol:.1f}x)")
        score += weights["HEAVY_INSTITUTIONAL"]

    candle_patterns = detect_candle_patterns(df)
    if candle_patterns:
        triggers.append(f"K线: {', '.join(candle_patterns)}")
        score += weights["CANDLE_PATTERN"]

    # --- 纯粹抄底信号逻辑 ---
    
    # [A] 布林带挤压 + 低位 (修改为动态比例)
    bb_squeeze_limit = CONFIG["filter"]["min_bb_squeeze_width"]
    bb_expand_rate = CONFIG["filter"]["bb_expansion_rate"]
    max_pos = CONFIG["filter"]["max_bottom_pos"]
    price_pos = (curr['close'] - low_60) / (high_60 - low_60) if high_60 > low_60 else 0.5
    
    if prev['BB_Width'] < bb_squeeze_limit: 
        # 计算扩张比例
        prev_width_safe = prev['BB_Width'] if prev['BB_Width'] > 0 else 0.001
        width_ratio = curr['BB_Width'] / prev_width_safe
        
        if width_ratio >= bb_expand_rate: 
            if curr['close'] > curr['open']: 
                 if price_pos <= max_pos: 
                    triggers.append(f"BB Squeeze: 变盘启动 (前宽:{prev['BB_Width']:.3f}, 扩张:{width_ratio:.2f}x)")
                    score += weights["BB_SQUEEZE"]

    # [B] ADX 趋势启动
    is_strong_trend = curr['ADX'] > params["adx_strong_threshold"] and curr['PDI'] > curr['MDI']
    is_adx_rising = curr['ADX'] > prev['ADX']
    if is_strong_trend and is_adx_rising: score += weights["STRONG_ADX"]
    
    recent_adx_min = df['ADX'].iloc[-10:-1].min()
    adx_activating = (recent_adx_min < params["adx_activation_lower"]) and \
                      (df['ADX'].iloc[-1] > df['ADX'].iloc[-2] > df['ADX'].iloc[-3])
    if adx_activating:
        triggers.append(f"趋势激活: 盘整结束 ADX拐头")
        score += weights["ADX_ACTIVATION"]

    # [新增] 旗形突破与支撑逻辑
    pattern_scored = False 
    
    # 1. 优先判断突破 (40分)
    if pattern_name:
        triggers.append(pattern_name)
        score += weights["PATTERN_BREAK"]
        pattern_scored = True
    
    # 2. 如果没有突破，判断是否在支撑线附近 (20分)
    if not pattern_scored and sup_slope is not None:
        curr_idx = len(df) - 1
        def get_sup_price(idx): return sup_slope * idx + sup_intercept
        
        curr_sup = get_sup_price(curr_idx)
        is_on_support_now = (curr['close'] >= curr_sup * 0.995) and (curr['close'] <= curr_sup * 1.02)
        
        if is_on_support_now:
            # 条件A: 触底企稳
            was_touching = False
            start_check_idx = max(0, curr_idx - 4)
            for i in range(start_check_idx, curr_idx):
                sup_at_i = get_sup_price(i)
                low_at_i = df['low'].iloc[i]
                if low_at_i <= sup_at_i * 1.02:
                    was_touching = True
                    break
            
            if was_touching:
                triggers.append("旗形支撑: 触底企稳 (4日确认)")
                score += weights["PATTERN_SUPPORT"]
                pattern_scored = True
            
            # 条件B: 假摔回踩
            if not pattern_scored:
                was_broken = False
                start_check_idx = max(0, curr_idx - 6)
                for i in range(start_check_idx, curr_idx - 2): 
                    sup_at_i = get_sup_price(i)
                    if df['close'].iloc[i] < sup_at_i:
                        was_broken = True
                        break
                if was_broken:
                    triggers.append("旗形支撑: 假摔回踩 (3日确认)")
                    score += weights["PATTERN_SUPPORT"]
                    pattern_scored = True

    # [C] MACD & KDJ 反转
    is_zero_cross = prev['DIF'] < 0 and curr['DIF'] > 0 and curr['DIF'] > curr['DEA']
    if is_zero_cross:
        triggers.append(f"MACD 金叉")
        score += weights["MACD_ZERO_CROSS"]
    
    if prev['J'] < params["kdj_j_oversold"] and curr['J'] > 0 and curr['K'] > curr['D']:
        triggers.append(f"KDJ 反击")
        score += weights["KDJ_REBOUND"]
    
    # [D] 底背离 (抄底核心)
    price_low_20 = df['close'].tail(20).min()
    price_is_low = curr['close'] <= price_low_20 * params["divergence_price_tolerance"]
    macd_low_20 = df['MACD'].tail(20).min()
    if price_is_low and curr['MACD'] < 0:
        if curr['MACD'] > macd_low_20 * params["divergence_macd_strength"] and curr['DIF'] > df['DIF'].tail(20).min():
             triggers.append(f"MACD 底背离")
             score += weights["MACD_DIVERGE"]
    
    # [E] 资金面
    if curr['OBV'] > curr['OBV_MA20']:
        obv_lookback = params["obv_lookback"]
        obv_rising = curr['OBV'] > df['OBV'].iloc[-obv_lookback]
        if obv_rising and curr['close'] > curr['open']:
             triggers.append("资金面: OBV趋势向上 (资金流入)")
             score += weights["OBV_TREND_UP"]

    # [F] 抛售高潮 (结合 RVOL 验证)
    if curr['low'] < curr['BB_Low']: 
        if rvol > params["rvol_capitulation"]:
            triggers.append(f"抛售高潮: 恐慌盘涌出 (RVOL {rvol:.1f})")
            score += weights["CAPITULATION"]

    # [新增] 四维共振逻辑 (4D Resonance)
    # 检测过去 N 天内是否发生了指标的底背离信号
    # 需要满足：CROSS(指标, 信号线) 且 当前PriceLow < 上一次Cross时的PriceLow 且 当前Indicator > 上一次Cross时的Indicator
    
    res_cfg = CONFIG["SCORE"]["RESONANCE"]
    res_window = res_cfg["window_days"]
    
    def check_divergence_window(series_val, series_sig, series_low, lookback):
        df_len = len(series_val)
        for i in range(df_len - lookback, df_len):
            if i <= 20: continue 
            
            # 1. 检查金叉
            if series_val[i-1] < series_sig[i-1] and series_val[i] > series_sig[i]:
                # 2. 寻找上一次金叉
                last_cross_idx = -1
                for j in range(i - 1, max(0, i - 60), -1):
                    if series_val[j-1] < series_sig[j-1] and series_val[j] > series_sig[j]:
                        last_cross_idx = j
                        break
                
                if last_cross_idx != -1:
                    # 3. 比较背离
                    price_lower = series_low[i] < series_low[last_cross_idx]
                    ind_higher = series_val[i] > series_val[last_cross_idx]
                    
                    if price_lower and ind_higher:
                        return True
        return False

    s_low = df['low'].values
    
    # 1. MACD 背离
    div_macd = check_divergence_window(df['DIF'].values, df['DEA'].values, s_low, res_window)
    
    # 2. RSI 背离 (使用 RSI6 和 RSI12 交叉)
    div_rsi = check_divergence_window(df['RSI6'].values, df['RSI12'].values, s_low, res_window)
    
    # 3. MFI 背离
    div_mfi = check_divergence_window(df['MFI'].values, df['MFI_MA'].values, s_low, res_window)
    
    # 4. CCI 背离 (特殊: 要求发生背离时 CCI 处于低位)
    div_cci = False
    cci_val = df['CCI'].values
    cci_ma = df['CCI_MA'].values
    for i in range(len(df) - res_window, len(df)):
        if i <= 20: continue
        if cci_val[i-1] < cci_ma[i-1] and cci_val[i] > cci_ma[i]:
            is_oversold = (cci_val[i] < -100) or (cci_val[i-1] < -100)
            if is_oversold:
                last_cross_idx = -1
                for j in range(i - 1, max(0, i - 60), -1):
                    if cci_val[j-1] < cci_ma[j-1] and cci_val[j] > cci_ma[j]:
                        last_cross_idx = j
                        break
                if last_cross_idx != -1:
                    if s_low[i] < s_low[last_cross_idx] and cci_val[i] > cci_val[last_cross_idx]:
                        div_cci = True
                        break

    # 计算共振
    resonance_count = sum([div_macd, div_rsi, div_mfi, div_cci])
    if resonance_count >= res_cfg["min_signals"]:
        triggers.append(f"四维共振: {resonance_count}指标底背离")
        score += res_cfg["bonus_score"]

    is_triggered = (score >= CONFIG["SCORE"]["MIN_ALERT_SCORE"]) and (len(violations) == 0)
    final_reason_parts = triggers + violations
    final_reason = "\n".join(final_reason_parts) if final_reason_parts else "无明显信号"
    
    return is_triggered, score, final_reason, res_line, sup_line, anchor_idx, rvol

async def check_signals(df, ticker):
    return await asyncio.to_thread(check_signals_sync, df, ticker)

# -----------------------------------------------------------------------------
# 图表生成函数
# -----------------------------------------------------------------------------
def _generate_chart_sync(df, ticker, res_line=[], sup_line=[], stop_price=None, support_price=None, anchor_idx=None):
    buf = io.BytesIO()
      
    default_lookback = 100 
    start_idx = max(0, len(df) - default_lookback)
    if anchor_idx is not None: start_idx = max(0, anchor_idx - 30) 

    plot_df = df.iloc[start_idx:].copy()
    last_date = plot_df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=10) 
    future_df = pd.DataFrame(index=future_dates, columns=plot_df.columns)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        plot_df = pd.concat([plot_df, future_df])

    # --- Volume Profile (红涨绿跌适配) ---
    valid_df = plot_df.dropna(subset=['close', 'volume'])
    if not valid_df.empty:
        price_min = valid_df['low'].min()
        price_max = valid_df['high'].max()
        bins = np.linspace(price_min, price_max, 50)
        
        bull_df = valid_df[valid_df['close'] >= valid_df['open']]
        bear_df = valid_df[valid_df['close'] < valid_df['open']]
        
        vol_bull, _ = np.histogram(bull_df['close'], bins=bins, weights=bull_df['volume'])
        vol_bear, _ = np.histogram(bear_df['close'], bins=bins, weights=bear_df['volume'])
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bar_height = (bins[1] - bins[0]) * 0.8
    else:
        vol_bull, vol_bear, bin_centers, bar_height = [], [], [], 0

    total_len = len(plot_df)                 
    if stop_price is None: stop_price = df['close'].iloc[-1] * 0.95
    if support_price is None: support_price = df['close'].iloc[-1] * 0.90
    stop_line_data = [stop_price] * total_len
    supp_line_data = [support_price] * total_len

    # --- Clipping Logic ---
    def clip_line_segments(segments):
        new_segments = []
        if not segments: return new_segments
        
        plot_start_date = plot_df.index[0]
        plot_start_ts = plot_start_date.timestamp()
        
        for seg in segments:
            d1, p1 = seg[0]
            d2, p2 = seg[1]
            try:
                if isinstance(d1, pd.Timestamp): d1_ts = d1.timestamp()
                else: d1_ts = d1.timestamp()
                if isinstance(d2, pd.Timestamp): d2_ts = d2.timestamp()
                else: d2_ts = d2.timestamp()

                if d2_ts < plot_start_ts: continue 
                
                if d1_ts < plot_start_ts:
                    if d2_ts - d1_ts == 0: continue
                    slope = (p2 - p1) / (d2_ts - d1_ts)
                    new_p1 = p1 + slope * (plot_start_ts - d1_ts)
                    new_segments.append([(plot_start_date, new_p1), (d2, p2)])
                else:
                    new_segments.append(seg)
            except: continue
        return new_segments

    res_line_clipped = clip_line_segments(res_line)
    sup_line_clipped = clip_line_segments(sup_line)

    # --- 样式定制 (红涨绿跌) ---
    premium_bg_color = '#131722'
    grid_color = '#2a2e39'
    text_color = '#b2b5be'
    volume_color = '#3b404e'
    
    my_marketcolors = mpf.make_marketcolors(
        up='#d93025',   # 红色 (涨)
        down='#1db954', # 绿色 (跌)
        edge='inherit',    
        wick='inherit',    
        volume=volume_color,
        ohlc='inherit'
    )
    
    my_style = mpf.make_mpf_style(
        base_mpl_style="dark_background",
        marketcolors=my_marketcolors,
        facecolor=premium_bg_color,
        figcolor=premium_bg_color,
        gridstyle=':', gridcolor=grid_color, gridaxis='both',
        rc={
            'font.family': 'sans-serif', 'axes.labelcolor': text_color,
            'xtick.labelcolor': text_color, 'ytick.labelcolor': text_color,
            'axes.edgecolor': grid_color,
            'ytick.left': False, 'ytick.right': True,
            'ytick.labelleft': False, 'ytick.labelright': True,
            'patch.linewidth': 0, 
        }
    )

    # [丝带] 填充逻辑
    ribbon_fast = plot_df['Ribbon_Fast'].values
    ribbon_slow = plot_df['Ribbon_Slow'].values
    
    # 蓝色 (Bullish), 黄色 (Bearish), alpha=0.1
    fb_bull = dict(y1=ribbon_fast, y2=ribbon_slow, where=ribbon_fast >= ribbon_slow, color='#00BFFF', alpha=0.1)
    fb_bear = dict(y1=ribbon_fast, y2=ribbon_slow, where=ribbon_fast < ribbon_slow, color='#FFFF00', alpha=0.1)

    add_plots = [
        mpf.make_addplot(plot_df['BB_Up'], color='#9370DB', linestyle=':', width=0.6, alpha=0.5),
        mpf.make_addplot(plot_df['BB_Mid'], color='#9370DB', linestyle=':', width=0.6, alpha=0.7), 
        mpf.make_addplot(plot_df['BB_Low'], color='#9370DB', linestyle=':', width=0.6, alpha=0.5),
        mpf.make_addplot(stop_line_data, color='red', linestyle='--', width=0.8, alpha=0.6), 
        mpf.make_addplot(supp_line_data, color='green', linestyle=':', width=0.8, alpha=0.6),
        # 丝带 addplot (线宽=0)
        mpf.make_addplot(plot_df['Ribbon_Fast'], width=0, alpha=0, fill_between=fb_bull),
        mpf.make_addplot(plot_df['Ribbon_Fast'], width=0, alpha=0, fill_between=fb_bear),
    ]
      
    seq_of_points = []
    if res_line_clipped:
        for line in res_line_clipped: seq_of_points.append([(line[0][0], float(line[0][1])), (line[1][0], float(line[1][1]))])
    if sup_line_clipped:
        for line in sup_line_clipped: seq_of_points.append([(line[0][0], float(line[0][1])), (line[1][0], float(line[1][1]))])

    kwargs = dict(
        type='candle', style=my_style, 
        ylabel='', addplot=add_plots, 
        volume=True, volume_panel=1, panel_ratios=(3, 1),
        tight_layout=True, datetime_format='%m-%d', xrotation=0, figsize=(10, 6),
        returnfig=True, 
        scale_padding={'right': 1.0}
    )
      
    if seq_of_points:
        kwargs['alines'] = dict(
            alines=seq_of_points,
            colors='#d1d4dc', linewidths=0.6, linestyle='-', alpha=0.4 
        )
      
    try:
        fig, axlist = mpf.plot(plot_df, **kwargs)
        ax_main = axlist[0]
        
        ax_main.text(0.5, 0.92, ticker, 
            transform=ax_main.transAxes, 
            fontsize=60, color='white', alpha=0.05, 
            ha='center', va='top', weight='bold', zorder=0)

        if not valid_df.empty:
            ax_vp = ax_main.twiny()
            max_vol = max(vol_bull.max(), vol_bear.max()) if len(vol_bull) > 0 else 1
            ax_vp.set_xlim(0, max_vol * 4) 
            ax_vp.invert_xaxis() 
            
            ax_vp.barh(bin_centers, vol_bear, height=bar_height, color='#1db954', alpha=0.06, align='center', zorder=0)
            ax_vp.barh(bin_centers, vol_bull, height=bar_height, color='#d93025', alpha=0.06, align='center', left=vol_bear, zorder=0)
            ax_vp.axis('off')

        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
        buf.seek(0)
    except Exception as e:
        logging.error(f"Plot Error: {e}")
        plt.close('all')
        return io.BytesIO()
    finally:
        plt.close('all')
        
    return buf

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

def create_alert_embed(ticker, score, price, reason, stop_loss, support, df, filename, rvol=None, is_filtered=False):
    level_str = get_level_by_score(score)
    if "过滤器" in reason or "STALE" in reason:
        color = 0x95a5a6 
    else:
        color = 0x00ff00 if score >= 80 else 0x3498db
    
    # 标题如果包含不支持的格式字符可能显示异常，故状态在描述中体现
    title_text = f"🚨{ticker} 抄底信号 | 得分 {score}"
    if is_filtered:
        title_text = f"🚫{ticker} 信号拦截 | 得分 {score} (低分)"
        color = 0x7f8c8d
      
    embed = discord.Embed(title=title_text, color=color)
    
    # 描述部分支持 markdown 删除线
    score_display = f"~~{score}~~" if is_filtered else f"{score}"
    embed.description = f"**现价:** `${price:.2f}`\n**得分:** {score_display}"
      
    curr = df.iloc[-1]
    obv_status = "流入" if curr['OBV'] > curr['OBV_MA20'] else "流出"
    
    vol_str = f"`{rvol:.2f}x`" if rvol else "N/A"
    
    indicator_text = (
        f"**RSI(14):** `{curr['RSI']:.1f}`\n"
        f"**ADX:** `{curr['ADX']:.1f}`\n"
        f"**RVOL:** {vol_str}\n" 
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
        
        asyncio.create_task(self.initialize_baselines())

        if not self.monitor_stocks.is_running():
            self.monitor_stocks.start()
        
        if not self.scheduled_report.is_running():
            self.scheduled_report.start()
            
        await self.tree.sync()

    async def initialize_baselines(self):
        users_data = settings.get("users", {})
        all_tickers = set()
        for u in users_data.values():
            all_tickers.update(u.get("stocks", []))
        
        for pool in STOCK_POOLS.values():
            all_tickers.update(pool)
            
        if all_tickers:
            await RVOLCalculator.precalculate_baselines(list(all_tickers))

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
            
            is_triggered, score, reason, res_line, sup_line, anchor_idx, rvol = await check_signals(df, ticker)
            
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
                    "anchor_idx": anchor_idx, 
                    "users": users_to_ping,
                    "rvol": rvol 
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
                    chart_buf = await generate_chart(
                        alert["df"], ticker, alert["res_line"], alert["sup_line"], 
                        alert["stop_loss"], alert["support"], alert["anchor_idx"]
                    )
                    filename = f"{ticker}.png"
                    
                    embed = create_alert_embed(
                        ticker, score, alert['price'], alert['reason'], 
                        alert['stop_loss'], alert['support'], alert['df'], filename,
                        rvol=alert["rvol"]
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
    asyncio.create_task(RVOLCalculator.precalculate_baselines(new_list))
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
    asyncio.create_task(RVOLCalculator.precalculate_baselines(new_list))
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

    await RVOLCalculator.precalculate_baselines([ticker])

    data_map = await fetch_historical_batch([ticker])
    quotes_map = await fetch_realtime_quotes([ticker])
    
    if not data_map or ticker not in data_map:
        await interaction.followup.send(f"Failed `{ticker}` (Check logs for 403/429 or data error)")
        return
        
    df = data_map[ticker]
    if ticker in quotes_map:
        df = await asyncio.to_thread(merge_and_recalc_sync, df, quotes_map[ticker])

    is_triggered, score, reason, r_l, s_l, anchor_idx, rvol = await check_signals(df, ticker)
    
    price = df['close'].iloc[-1]
    stop_loss, support = calculate_risk_levels(df)

    if not reason: 
        reason = f"无明显信号 (得分: {score})"
    
    chart_buf = await generate_chart(df, ticker, r_l, s_l, stop_loss, support, anchor_idx)
    filename = f"{ticker}_test.png"
    
    # [修改] 判断是否是低分过滤信号
    is_filtered = score < CONFIG["SCORE"]["MIN_ALERT_SCORE"]
    
    embed = create_alert_embed(
        ticker, score, price, reason, stop_loss, support, df, filename, 
        rvol=rvol, is_filtered=is_filtered
    )

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
