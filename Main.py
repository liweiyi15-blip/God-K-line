import discord
from discord import app_commands
from discord.ext import tasks
import json
import os
from datetime import datetime, time, timedelta
import asyncio
import pandas as pd
import numpy as np
import mplfinance as mpf
import pytz
from dotenv import load_dotenv
from collections import defaultdict
from scipy.stats import linregress
import aiohttp
import io

# --- 加载环境变量 ---
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
FMP_API_KEY = os.getenv("FMP_API_KEY")
try:
    ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID"))
except (TypeError, ValueError):
    ALERT_CHANNEL_ID = 0

# --- 全局配置 ---
MARKET_TIMEZONE = pytz.timezone('America/New_York')

SETTINGS_FILE = "/app/data/settings.json"
if not os.path.exists("/app/data"):
    SETTINGS_FILE = "settings.json"

TIME_PRE_MARKET_START = time(9, 0)
TIME_MARKET_OPEN = time(9, 30)
TIME_MARKET_CLOSE = time(16, 0)

# --- 核心策略配置 ---
CONFIG = {
    "filter": {
        "max_60d_gain": 3.0,     
        "max_rsi": 85,           
        "max_day_change": 0.15,  
        "min_vol_ratio": 1.3,    
        "intraday_vol_ratio": 1.8, 
        "min_converge_angle": 0.05
    },
    "pattern": {
        "min_r2": 0.70,
        "window": 60
    },
    "system": {
        "cooldown_days": 5,
        "max_charts_per_scan": 5,
        "history_days": 400      # 保持400天以确保EMA90准确
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
    "NASDAQ_100": ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "ADBE", "COST", "PEP", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "CMCSA", "AZN", "QCOM", "TXN", "AMGN", "HON", "INTU", "SBUX", "GILD", "BKNG", "DIOD", "MDLZ", "ISRG", "REGN", "LRCX", "VRTX", "ADP", "ADI", "MELI", "KLAC", "PANW", "SNPS", "CDNS", "CHTR", "MAR", "CSX", "ORLY", "MNST", "NXPI", "CTAS", "FTNT", "WDAY", "DXCM", "PCAR", "KDP", "PAYX", "IDXX", "AEP", "LULU", "EXC", "BIIB", "ADSK", "XEL", "ROST", "MCHP", "CPRT", "SGEN", "DLTR", "EA", "FAST", "CTSH", "WBA", "VRSK", "CSGP", "ODFL", "ANSS", "EBAY", "ILMN", "GFS", "ALGN", "TEAM", "CDW", "WBD", "SIRI", "ZM", "ENPH", "JD", "PDD", "LCID", "RIVN", "ZS", "DDOG", "CRWD", "TTD", "BKR", "CEG", "GEHC", "ON", "FANG"],
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
    except:
        settings = {"users": {}, "signal_history": {}}

def save_settings():
    try:
        dir_name = os.path.dirname(SETTINGS_FILE)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

def get_user_data(user_id):
    uid_str = str(user_id)
    if "users" not in settings: settings["users"] = {}
    if uid_str not in settings["users"]:
        settings["users"][uid_str] = {"stocks": [], "daily_status": {}}
    return settings["users"][uid_str]

# --- 核心逻辑 (计算部分) ---
def calculate_nx_indicators(df):
    """计算核心指标，必须在每次更新数据后调用"""
    # 确保数值类型正确
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df['Nx_Blue_UP'] = df['high'].ewm(span=24, adjust=False).mean()
    df['Nx_Blue_DW'] = df['low'].ewm(span=23, adjust=False).mean()
    df['Nx_Yellow_UP'] = df['high'].ewm(span=89, adjust=False).mean()
    df['Nx_Yellow_DW'] = df['low'].ewm(span=90, adjust=False).mean()
    
    price_col = 'close'
    exp12 = df[price_col].ewm(span=12, adjust=False).mean()
    exp26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df

def process_dataframe_sync(hist_data):
    """处理历史数据初始加载"""
    if not hist_data: return None
    df = pd.DataFrame(hist_data)
    if 'date' not in df.columns: return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index(ascending=True)
    return calculate_nx_indicators(df)

def merge_and_recalc_sync(df, quote):
    """
    [新增] 将实时Quote缝合到历史DataFrame中，并重新计算指标
    这是日内交易的核心：历史趋势 + 实时价格
    """
    if df is None or quote is None: return df
    
    # 解析 Quote 数据
    try:
        # FMP timestamp 是 Unix timestamp (秒)
        quote_time = pd.to_datetime(quote['timestamp'], unit='s').tz_localize('UTC').tz_convert(MARKET_TIMEZONE)
        # 移除时间部分，只保留日期，用于判断是"更新今日"还是"新增今日"
        quote_date = quote_time.normalize()
        
        last_idx = df.index[-1]
        last_date = last_idx.normalize() if hasattr(last_idx, 'normalize') else pd.to_datetime(last_idx).normalize()

        new_row = {
            'open': quote['open'],
            'high': max(quote['dayHigh'], quote['price']), # 保护：有时候价格比dayHigh还高
            'low': min(quote['dayLow'], quote['price']),
            'close': quote['price'],
            'volume': quote['volume'],
            'date': quote_date # 索引使用日期
        }

        # 逻辑：如果历史数据的最后一天 == Quote的日期，说明历史数据已经包含了今天（部分），需要覆盖更新
        # 如果历史数据最后一天 < Quote的日期，说明是新的一天，追加
        
        df_mod = df.copy()
        
        if last_date == quote_date:
            # 更新最后一行
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_mod.at[last_idx, col] = new_row[col]
        elif last_date < quote_date:
            # 追加新的一行
            new_df = pd.DataFrame([new_row])
            new_df = new_df.set_index('date')
            df_mod = pd.concat([df_mod, new_df])
        
        # 必须重新计算指标，因为Close变了，EMA/RSI/MACD都会变
        return calculate_nx_indicators(df_mod)
        
    except Exception as e:
        print(f"Merge Error: {e}")
        return df

async def fetch_historical_batch(symbols: list, days=None):
    if not symbols: return {}
    if days is None: days = CONFIG["system"]["history_days"]
    
    chunk_size = 50 
    results = {}
    now = datetime.now()
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbols_str}&from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = []
                        if isinstance(data, dict):
                            if "historicalStockList" in data: items = data["historicalStockList"]
                            elif "symbol" in data and "historical" in data: items = [data]
                        elif isinstance(data, list): items = data
                        
                        for item in items:
                            sym = item.get('symbol')
                            hist = item.get('historical', [])
                            if not hist or not sym: continue
                            df = await asyncio.to_thread(process_dataframe_sync, hist)
                            if df is not None: results[sym] = df
            except Exception: pass
    return results

async def fetch_realtime_quotes(symbols: list):
    """[新增] 批量获取实时报价"""
    if not symbols: return {}
    chunk_size = 100 # Quote 接口通常支持更多
    quotes_map = {}
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            # 使用 quote 接口
            url = f"https://financialmodelingprep.com/stable/quote?symbol={symbols_str}&apikey={FMP_API_KEY}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list):
                            for item in data:
                                sym = item.get('symbol')
                                if sym: quotes_map[sym] = item
            except Exception as e:
                print(f"Quote Fetch Error: {e}")
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

def check_signals_sync(df):
    if len(df) < 60: return False, "", "NONE", [], []
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    level = "NORMAL"

    low_60 = df['low'].tail(60).min()
    if curr['close'] > low_60 * CONFIG["filter"]["max_60d_gain"]: return False, "", "RISK_FILTER", [], []
    if abs((curr['close'] - prev['close']) / prev['close']) > CONFIG["filter"]["max_day_change"]: return False, "", "RISK_FILTER", [], []
    if curr['RSI'] > CONFIG["filter"]["max_rsi"]: return False, "", "RISK_FILTER", [], []

    ny_now = datetime.now(MARKET_TIMEZONE)
    market_open = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = (ny_now - market_open).total_seconds() / 60
    
    is_open_market = 0 < minutes_elapsed < 390
    
    if is_open_market:
        safe_minutes = max(minutes_elapsed, 20) 
        projection_factor = 390 / safe_minutes
        proj_vol = curr['volume'] * projection_factor
        vol_threshold = CONFIG["filter"]["intraday_vol_ratio"]
    else:
        proj_vol = curr['volume']
        vol_threshold = CONFIG["filter"]["min_vol_ratio"]
        
    is_heavy_volume = proj_vol > curr['Vol_MA20'] * vol_threshold

    recent_10 = df.tail(10)
    had_breakout = (recent_10['close'] > recent_10['Nx_Blue_UP']).any()
    on_support = curr['close'] > curr['Nx_Blue_DW'] and curr['low'] <= curr['Nx_Blue_UP'] * 1.02
    
    if had_breakout and on_support and is_heavy_volume:
        triggers.append(f"👑 **二次起爆**: 蓝梯回踩确认 + 放量启动")
        level = "GOD_TIER"

    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name:
        if is_heavy_volume:
            triggers.append(pattern_name)
            if level != "GOD_TIER": level = "S_TIER"

    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        triggers.append(f"📈 **Nx 蓝梯突破**: 趋势转多确认")
        if level not in ["GOD_TIER", "S_TIER"]: level = "A_TIER"

    low_20 = df['low'].tail(20).min()
    price_is_low = curr['low'] <= low_20 * 1.02
    dif_20_min = df['DIF'].tail(20).min()
    divergence = (curr['DIF'] > dif_20_min) and (curr['MACD'] > prev['MACD'])
    
    if price_is_low and divergence:
        if is_downtrend or curr['RSI'] < 40:
             triggers.append(f"💎 **Cd 结构底背离**: 底部反转信号")
             if level not in ["GOD_TIER", "S_TIER", "A_TIER"]: level = "B_TIER"

    if prev['RSI'] < 30 and curr['RSI'] > 30:
        triggers.append(f"🚀 **RSI 弘历战法**: 超卖金叉")
        if level == "NORMAL": level = "C_TIER"

    if triggers:
        if is_downtrend and len(triggers) < 2 and level not in ["GOD_TIER", "S_TIER"]:
            return False, "", "WEAK_SIGNAL", [], []
        return True, "\n".join(triggers), level, res_line, sup_line
    return False, "", "NONE", [], []

async def check_signals(df):
    return await asyncio.to_thread(check_signals_sync, df)

def _generate_chart_sync(df, ticker, res_line=[], sup_line=[]):
    filename = f"{ticker}_alert.png"
    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    my_style = mpf.make_mpf_style(base_mpl_style="ggplot", marketcolors=s, gridstyle=":")
    plot_df = df.tail(80)
    add_plots = [
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['MACD'], panel=2, type='bar', color='dimgray', alpha=0.5, ylabel='MACD'),
        mpf.make_addplot(plot_df['DIF'], panel=2, color='orange'),
        mpf.make_addplot(plot_df['DEA'], panel=2, color='blue'),
    ]
    kwargs = dict(type='candle', style=my_style, title=f"{ticker} Analysis", ylabel='Price', addplot=add_plots, volume=True, panel_ratios=(6, 2, 2), savefig=filename)
    if res_line: 
        all_lines = []
        if res_line: all_lines.extend(res_line)
        if sup_line: all_lines.extend(sup_line)
        kwargs['alines'] = dict(alines=all_lines, colors='white', linewidths=1.5, linestyle='--')
    mpf.plot(plot_df, **kwargs)
    return filename

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
        print(f'Logged in as {self.user}')
        self.alert_channel = self.get_channel(ALERT_CHANNEL_ID)
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
        
        print(f"[{now_et.strftime('%H:%M')}] Scanning started...")
        users_data = settings.get("users", {})
        all_tickers = set()
        ticker_user_map = defaultdict(list)
        
        for uid, udata in users_data.items():
            for k in list(udata['daily_status'].keys()):
                if not k.endswith(today_str): del udata['daily_status'][k]
            for ticker in udata.get("stocks", []):
                all_tickers.add(ticker)
                ticker_user_map[ticker].append(uid)

        if not all_tickers: return

        # 1. 获取历史数据 (Base)
        hist_map = await fetch_historical_batch(list(all_tickers))
        
        # 2. 获取实时报价 (Intraday Update)
        quotes_map = {}
        if is_open:
            quotes_map = await fetch_realtime_quotes(list(all_tickers))

        alerts_buffer = []

        for ticker, df_hist in hist_map.items():
            # 缝合逻辑：将实时报价缝合到历史数据中
            df = df_hist
            if ticker in quotes_map:
                # 在线程池中执行缝合+重算，防止阻塞
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
            for i in range(1, cooldown_days + 1):
                past_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                if past_date in history and ticker in history[past_date]:
                    past_level = history[past_date][ticker]["level"]
                    if past_level == "GOD_TIER": 
                        in_cooldown = True
                        break 
                    
            is_triggered, reason, level, res_line, sup_line = await check_signals(df)
            
            if in_cooldown and level != "GOD_TIER": continue 

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
                    chart_file = await generate_chart(alert["df"], ticker, alert["res_line"], alert["sup_line"])
                    msg = (
                        f"{mentions}\n【{emoji} {level} 信号】\n"
                        f"🎯 **{ticker}** | 💰 `${alert['price']:.2f}`\n"
                        f"{'-'*20}\n{alert['reason']}\n{'-'*20}\n"
                        f"🛑 动态止损(2ATR): `${alert['support']:.2f}`"
                    )
                    try:
                        file = discord.File(chart_file)
                        await self.alert_channel.send(content=msg, file=file)
                        sent_charts += 1
                    except Exception as e: print(e)
                    finally:
                        if os.path.exists(chart_file): os.remove(chart_file)
                else:
                    summary_list.append(f"{emoji} **{ticker}** ({level})")

            if summary_list:
                summary_msg = f"📋 **其他触发信号 (简报)**:\n" + " | ".join(summary_list)
                try: await self.alert_channel.send(content=summary_msg)
                except: pass
            
            save_settings()
        
        print(f"[{now_et.strftime('%H:%M')}] Scan finished. Alerts: {len(alerts_buffer)}")

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
    
    # 获取历史 + 实时
    data_map = await fetch_historical_batch([ticker])
    quotes_map = await fetch_realtime_quotes([ticker])
    
    if not data_map or ticker not in data_map:
        await interaction.followup.send(f"❌ 失败 `{ticker}`")
        return
        
    df = data_map[ticker]
    # 如果有实时数据，进行缝合
    if ticker in quotes_map:
        df = await asyncio.to_thread(merge_and_recalc_sync, df, quotes_map[ticker])

    is_triggered, reason, level, r_l, s_l = await check_signals(df)
    
    price = df['close'].iloc[-1]
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else (price * 0.05)
    stop_loss = price - (2 * atr_val)

    cf = await generate_chart(df, ticker, r_l, s_l)
    msg = f"✅ `{ticker}` | {level}\n💰 `${price:.2f}`\n📝 {reason}\n🛑 Stop: `${stop_loss:.2f}`"
    try:
        f = discord.File(cf)
        await interaction.followup.send(content=msg, file=f)
    except: pass
    finally:
        if os.path.exists(cf): os.remove(cf)

if __name__ == "__main__":
    if DISCORD_TOKEN: client.run(DISCORD_TOKEN)
