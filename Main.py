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

# --- 加载环境变量 ---
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
FMP_API_KEY = os.getenv("FMP_API_KEY")
try:
    ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID"))
except (TypeError, ValueError):
    ALERT_CHANNEL_ID = 0

# --- 全局配置与常量 (对应图片建议：配置抽离) ---
MARKET_TIMEZONE = pytz.timezone('America/New_York')
SETTINGS_FILE = "/app/data/settings.json"

# 本地测试兼容
if not os.path.exists("/app/data"):
    try:
        if not os.path.exists("/app/data"): pass
    except:
        SETTINGS_FILE = "settings.json"

TIME_PRE_MARKET_START = time(9, 0)
TIME_MARKET_OPEN = time(9, 30)
TIME_MARKET_CLOSE = time(16, 0)

# --- 核心策略配置 (在此处统一调参) ---
CONFIG = {
    "filter": {
        "max_60d_gain": 1.4,       # [风控] 60天涨幅超过40%过滤
        "max_3d_gain": 0.35,       # [风控] 3天涨幅超过35%过滤 (防止追高)
        "max_day_change": 0.12,    # [风控] 单日涨跌幅超过12%过滤 (防天地板情绪过热)
        "min_vol_ratio": 1.3,      # 放量倍数
        "min_converge_angle": 0.05 # 旗形收敛角度差
    },
    "pattern": {
        "min_r2": 0.70,            # [质量] 线性回归拟合度阈值 (0.7才算有效趋势)
        "window": 60               # 扫描窗口
    },
    "emoji": {
        "GOD_TIER": "👑", 
        "S_TIER": "🔥", 
        "A_TIER": "📈", 
        "B_TIER": "💎", 
        "C_TIER": "🚀",
        "RISK": "🛡️"
    }
}

# --- 静态股票池 ---
NASDAQ_100_LIST = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "ADBE",
    "COST", "PEP", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "CMCSA", "AZN", "QCOM",
    "TXN", "AMGN", "HON", "INTU", "SBUX", "GILD", "BKNG", "DIOD", "MDLZ", "ISRG",
    "REGN", "LRCX", "VRTX", "ADP", "ADI", "MELI", "KLAC", "PANW", "SNPS", "CDNS",
    "CHTR", "MAR", "CSX", "ORLY", "MNST", "NXPI", "CTAS", "FTNT", "WDAY", "DXCM",
    "PCAR", "KDP", "PAYX", "IDXX", "AEP", "LULU", "EXC", "BIIB", "ADSK", "XEL",
    "ROST", "MCHP", "CPRT", "SGEN", "DLTR", "EA", "FAST", "CTSH", "WBA", "VRSK",
    "CSGP", "ODFL", "ANSS", "EBAY", "ILMN", "GFS", "ALGN", "TEAM", "CDW", "WBD",
    "SIRI", "ZM", "ENPH", "JD", "PDD", "LCID", "RIVN", "ZS", "DDOG", "CRWD", "TTD",
    "BKR", "CEG", "GEHC", "ON", "FANG"
]

GOD_TIER_LIST = [
    "NVDA", "AMD", "TSM", "SMCI", "AVGO", "ARM",
    "PLTR", "AI", "PATH",
    "BABA", "PDD", "BIDU", "NIO", "LI", "XPEV",
    "COIN", "MARA", "MSTR"
]

# --- 全局变量 ---
settings = {}

# --- 辅助函数 ---
def load_settings():
    global settings
    try:
        directory = os.path.dirname(SETTINGS_FILE)
        if directory and not os.path.exists(directory):
            try: os.makedirs(directory)
            except OSError: pass
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            settings = {"users": {}}
            save_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        settings = {"users": {}}

def save_settings():
    try:
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

# --- [核心优化] 异步批量获取历史数据 + Bug修复 ---
async def fetch_historical_batch(symbols: list, days=400):
    if not symbols: return {}
    
    # FMP v3 historical-price-full 支持批量，建议分片 (50-100)
    chunk_size = 50 
    results = {}
    
    now = datetime.now()
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            # 使用 v3 接口以支持批量
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbols_str}?from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
            
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # [Bug修复] 对应图片中的 FMP 返回结构判断逻辑
                        # FMP 在单股票和多股票时返回结构不同，且有时会有 Error Message
                        if isinstance(data, dict):
                            if "Error Message" in data:
                                print(f"FMP Error: {data['Error Message']}")
                                continue
                            if "historicalStockList" in data:
                                items = data["historicalStockList"]
                            elif "symbol" in data and "historical" in data:
                                items = [data] # 转成 list 统一处理
                            else:
                                items = []
                        elif isinstance(data, list):
                            items = data
                        else:
                            items = []

                        for item in items:
                            sym = item.get('symbol')
                            hist = item.get('historical', [])
                            if not hist or not sym: continue
                            
                            df = pd.DataFrame(hist)
                            # 必须确保有 date 字段
                            if 'date' not in df.columns: continue
                            
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date').sort_index(ascending=True)
                            
                            # 计算指标
                            df = calculate_nx_indicators(df)
                            results[sym] = df
            except Exception as e:
                print(f"Error fetching batch {chunk}: {e}")
                
    return results

# --- [保留] 实时价格查询 (Watchlist用) ---
async def fetch_fmp_quotes(symbols: list):
    if not symbols: return []
    chunk_size = 50
    all_quotes = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            url = f"https://financialmodelingprep.com/stable/quote?symbol={symbols_str}&apikey={FMP_API_KEY}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list): all_quotes.extend(data)
            except Exception as e:
                print(f"Error fetching quotes: {e}")
    return all_quotes

# --- 核心指标计算 ---
def calculate_nx_indicators(df):
    # 基础均线
    df['Nx_Blue_UP'] = df['high'].ewm(span=24, adjust=False).mean()
    df['Nx_Blue_DW'] = df['low'].ewm(span=23, adjust=False).mean()
    df['Nx_Yellow_UP'] = df['high'].ewm(span=89, adjust=False).mean()
    df['Nx_Yellow_DW'] = df['low'].ewm(span=90, adjust=False).mean()
    
    # MACD
    price_col = 'close'
    exp12 = df[price_col].ewm(span=12, adjust=False).mean()
    exp26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 成交量均线
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    return df

# --- [核心优化] 线性回归趋势线计算 (图片建议：信号质量升级) ---
def linreg_trend(points, min_r2):
    """
    使用线性回归计算趋势线
    返回: (slope, intercept, r_sq) 或 None
    """
    if len(points) < 4: return None
    
    # 构造 X 轴 (0, 1, 2...)
    x = np.arange(len(points))
    y = points.values
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_sq = r_value ** 2
    
    # [质量控制] 过滤拟合度太差的 (R^2 < 0.7)
    if r_sq < min_r2: return None
    
    return slope, intercept, r_sq

# --- [重写] 机构级形态识别 (引入 Scipy 线性回归) ---
def identify_patterns(df):
    window = CONFIG["pattern"]["window"]
    min_r2 = CONFIG["pattern"]["min_r2"]
    
    if len(df) < window + 5: return None, [], []
    
    recent = df.tail(window).copy()
    recent = recent.reset_index() # index 变成 0,1,2...
    
    # 寻找局部极值
    recent['pivot_high'] = recent['high'].rolling(5, center=True).max() == recent['high']
    recent['pivot_low'] = recent['low'].rolling(5, center=True).min() == recent['low']
    
    high_points = recent[recent['pivot_high']]
    low_points = recent[recent['pivot_low']]
    
    if len(high_points) >= 3 and len(low_points) >= 3:
        # 取最近的 N 个极值点进行拟合 (例如最近8个)
        h_data = high_points['high'].tail(8)
        l_data = low_points['low'].tail(8)
        
        # 使用线性回归拟合压力线和支撑线
        res_trend = linreg_trend(h_data, min_r2)
        sup_trend = linreg_trend(l_data, min_r2)
        
        if res_trend and sup_trend:
            slope_res, int_res, r2_res = res_trend
            slope_sup, int_sup, r2_sup = sup_trend
            
            # [收敛判断逻辑]
            # 1. 压力线向下 (斜率 < 0)
            # 2. 支撑线向上 (斜率 > 0) 或走平
            # 3. 确实收敛: 支撑斜率 > 压力斜率 + 阈值
            # 4. 拟合度高: R^2 > 0.7 (已经在 linreg_trend 中过滤)
            
            if slope_res < 0 and (slope_sup > slope_res + CONFIG["filter"]["min_converge_angle"]):
                
                # 计算今日理论突破位
                curr_idx = recent.index[-1]
                resistance_today = slope_res * curr_idx + int_res
                
                curr_close = recent['close'].iloc[-1]
                curr_vol = recent['volume'].iloc[-1]
                vol_ma = recent['Vol_MA20'].iloc[-1]
                
                # [突破确认]
                # 1. 前一天收盘价在压力线下方 (防止已经是突破后的行情)
                prev_idx = recent.index[-2]
                res_prev = slope_res * prev_idx + int_res
                prev_close = recent['close'].iloc[-2]
                
                if prev_close <= res_prev * 1.02: # 允许2%误差
                    # 2. 今天收盘突破 + 放量
                    if curr_close > resistance_today and curr_vol > vol_ma * CONFIG["filter"]["min_vol_ratio"]:
                        
                        # 构造画线数据 (取拟合段的起点和终点，绘制延长线)
                        start_idx = recent.index[0]
                        end_idx = recent.index[-1]
                        
                        # 转换回时间坐标
                        t1 = recent['date'].iloc[0]
                        p1 = slope_res * start_idx + int_res
                        t2 = recent['date'].iloc[-1]
                        p2 = slope_res * end_idx + int_res
                        
                        t3 = recent['date'].iloc[0]
                        p3 = slope_sup * start_idx + int_sup
                        t4 = recent['date'].iloc[-1]
                        p4 = slope_sup * end_idx + int_sup
                        
                        return "🚩 **放量旗形突破(机构算法)**", [[(t1,p1), (t2,p2)]], [[(t3,p3), (t4,p4)]]

    return None, [], []

# --- [重写] 信号检查 (严格遵循优先级表) ---
def check_signals(df):
    if len(df) < 60: return False, "", "NONE", [], []
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    level = "NORMAL"
    
    # === 优先级 1: 风控 (保命第一) ===
    
    # 1.1 60日暴涨过滤
    low_60 = df['low'].tail(60).min()
    if curr['close'] > low_60 * CONFIG["filter"]["max_60d_gain"]: 
        return False, "", "RISK_FILTER", [], []

    # 1.2 [新增] 3日短期暴涨过滤 (防止追高接盘)
    gain_3d = df['close'].pct_change(3).iloc[-1]
    if gain_3d > CONFIG["filter"]["max_3d_gain"]:
        return False, "", "RISK_FILTER", [], []
        
    # 1.3 [新增] 当日情绪过热/跌停 (天地板过滤)
    day_change = abs((curr['close'] - prev['close']) / prev['close'])
    if day_change > CONFIG["filter"]["max_day_change"]:
        return False, "", "RISK_FILTER", [], []

    # === 优先级 2: GOD_TIER (二次起爆) ===
    recent_10 = df.tail(10)
    # 过去10天曾经突破过蓝梯上沿
    had_breakout = (recent_10['close'] > recent_10['Nx_Blue_UP']).any()
    # 当前回踩蓝梯 (在蓝梯上下沿之间，或者贴近下沿)
    on_support = curr['close'] > curr['Nx_Blue_DW'] and curr['low'] <= curr['Nx_Blue_UP'] * 1.02
    # 再次放量
    re_volume = curr['volume'] > curr['Vol_MA20'] * 1.5
    
    if had_breakout and on_support and re_volume:
        triggers.append(f"👑 **二次起爆**: 蓝梯回踩确认 + 放量启动")
        level = "GOD_TIER"

    # === 优先级 3: S_TIER (旗形/楔形突破) ===
    # 只有没触发 GOD_TIER 时才判定 S_TIER，或者叠加
    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name:
        triggers.append(pattern_name)
        if level != "GOD_TIER": level = "S_TIER"

    # === 优先级 4: A_TIER (Nx 蓝梯突破) ===
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        triggers.append(f"📈 **Nx 蓝梯突破**: 趋势转多确认")
        if level not in ["GOD_TIER", "S_TIER"]: level = "A_TIER"

    # === 优先级 5: B_TIER (Cd/MACD 底背离) ===
    low_20 = df['low'].tail(20).min()
    price_is_low = curr['low'] <= low_20 * 1.01
    dif_20_min = df['DIF'].tail(20).min()
    divergence = curr['DIF'] > dif_20_min 
    momentum_turn = curr['MACD'] > prev['MACD']
    
    if price_is_low and divergence and momentum_turn:
        if is_downtrend or curr['RSI'] < 35:
             triggers.append(f"💎 **Cd 结构底背离**: 底部反转信号")
             if level not in ["GOD_TIER", "S_TIER", "A_TIER"]: level = "B_TIER"

    # === 优先级 6: C_TIER (RSI 弘历战法) ===
    if prev['RSI'] < 30 and curr['RSI'] > 30:
        triggers.append(f"🚀 **RSI 弘历战法**: 超卖金叉")
        if level == "NORMAL": level = "C_TIER" # 最低优先级

    # === 优先级 7: 尾部风控 (弱信号过滤) ===
    if triggers:
        # 如果是空头趋势，且不是神级或S级信号，必须有2个以上共振才报
        if is_downtrend and len(triggers) < 2 and level not in ["GOD_TIER", "S_TIER"]:
            return False, "", "WEAK_SIGNAL", [], []
            
        return True, "\n".join(triggers), level, res_line, sup_line

    return False, "", "NONE", [], []

# --- 异步画图 (图片建议：小优化) ---
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
    
    lines_to_draw = []
    if res_line: lines_to_draw.extend(res_line) 
    if sup_line: lines_to_draw.extend(sup_line)
    
    kwargs = dict(
        type='candle', style=my_style, title=f"{ticker} Analysis", ylabel='Price', 
        addplot=add_plots, volume=True, panel_ratios=(6, 2, 2), savefig=filename
    )
    if lines_to_draw:
        kwargs['alines'] = dict(alines=lines_to_draw, colors='white', linewidths=1.5, linestyle='--')

    mpf.plot(plot_df, **kwargs)
    return filename

async def generate_chart(df, ticker, res_line=[], sup_line=[]):
    # 放入线程池运行，避免阻塞主循环
    return await asyncio.to_thread(_generate_chart_sync, df, ticker, res_line, sup_line)

# --- Discord Client ---

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
        
        # 1. 收集所有用户关注的股票
        users_data = settings.get("users", {})
        all_tickers = set()
        ticker_user_map = defaultdict(list)
        
        for uid, udata in users_data.items():
            # 清理旧状态
            for k in list(udata['daily_status'].keys()):
                if not k.endswith(today_str): del udata['daily_status'][k]
            for ticker in udata.get("stocks", []):
                all_tickers.add(ticker)
                ticker_user_map[ticker].append(uid)

        if not all_tickers: return

        # 2. [批量获取] 使用 Async Batch 替代循环 Request
        # 这一步是性能提升的关键，瞬间获取所有数据
        data_map = await fetch_historical_batch(list(all_tickers))
        
        # 3. 处理数据
        for ticker, df in data_map.items():
            user_ids = ticker_user_map[ticker]
            
            # 检查是否每个人都推送过了
            all_alerted = True
            for uid in user_ids:
                status_key = f"{ticker}-{today_str}"
                status = users_data[uid]['daily_status'].get(status_key, "NONE")
                if is_pre and status == "NONE": all_alerted = False
                if is_open and status not in ["MARKET_SENT", "BOTH_SENT"]: all_alerted = False
            
            if all_alerted: continue

            # 信号检查
            is_triggered, reason, level, res_line, sup_line = check_signals(df)
            
            if is_triggered:
                # 异步生成图表
                chart_file = await generate_chart(df, ticker, res_line, sup_line)
                price = df['close'].iloc[-1]
                nx_support = df['Nx_Blue_DW'].iloc[-1]
                
                users_to_ping = []
                for uid in user_ids:
                    status_key = f"{ticker}-{today_str}"
                    status = users_data[uid]['daily_status'].get(status_key, "NONE")
                    should_alert = False
                    if is_pre and status == "NONE": should_alert = True
                    if is_open and status in ["NONE", "PRE_SENT"]: should_alert = True
                    
                    if should_alert:
                        users_to_ping.append(uid)
                        new_status = "PRE_SENT" if is_pre else ("BOTH_SENT" if status == "PRE_SENT" else "MARKET_SENT")
                        users_data[uid]['daily_status'][status_key] = new_status
                
                if users_to_ping:
                    save_settings()
                    mentions = " ".join([f"<@{uid}>" for uid in users_to_ping])
                    
                    # 使用配置中的 Emoji
                    emoji = CONFIG["emoji"].get(level, "🚨")
                    
                    msg = (
                        f"{mentions}\n【{emoji} {level} 信号触发】\n"
                        f"🎯 **标的**: `{ticker}` | 💰 **现价**: `${price:.2f}`\n"
                        f"{'-'*25}\n{reason}\n{'-'*25}\n"
                        f"🌊 **Nx 蓝梯下沿**: `${nx_support:.2f}`"
                    )
                    try:
                        file = discord.File(chart_file)
                        await self.alert_channel.send(content=msg, file=file)
                    except Exception as e:
                        print(f"Error sending msg: {e}")
                    finally:
                        if os.path.exists(chart_file): os.remove(chart_file)
        
        print(f"[{now_et.strftime('%H:%M')}] Scan finished.")

# --- 实例化 & 注册命令 ---

intents = discord.Intents.default()
client = StockBotClient(intents=intents)

@client.tree.command(name="import_nasdaq", description="导入纳指100")
async def import_nasdaq(interaction: discord.Interaction):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set(NASDAQ_100_LIST))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已添加 {len(new_list)} 只纳指成分股。")

@client.tree.command(name="import_gods", description="导入神级热门股")
async def import_gods(interaction: discord.Interaction):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set(GOD_TIER_LIST))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已添加神级热门股。")

@client.tree.command(name="clearstocks", description="清空关注列表")
async def clear_stocks(interaction: discord.Interaction):
    user_data = get_user_data(interaction.user.id)
    user_data["stocks"] = []
    user_data["daily_status"] = {}
    save_settings()
    await interaction.response.send_message("🗑️ 已清空。", ephemeral=True)

# --- Watch 系列命令 ---

@client.tree.command(name="watch_add", description="批量添加关注 (例如: AAPL, TSLA)")
@app_commands.describe(codes="股票代码，用逗号或空格分隔")
async def watch_add(interaction: discord.Interaction, codes: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in codes.replace(',', ' ').replace('，', ' ').split() if t.strip()]))
    
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    
    await interaction.followup.send(f"✅ 已关注: `{', '.join(new_list)}`")

@client.tree.command(name="watch_remove", description="从关注列表移除代码")
@app_commands.describe(codes="股票代码，用逗号或空格分隔")
async def watch_remove(interaction: discord.Interaction, codes: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    to_remove = set([t.strip().upper() for t in codes.replace(',', ' ').replace('，', ' ').split() if t.strip()])
    
    current_list = user_data["stocks"]
    new_list = [s for s in current_list if s not in to_remove]
    
    if len(new_list) == len(current_list):
        await interaction.followup.send("⚠️ 列表中未找到指定代码。")
    else:
        user_data["stocks"] = new_list
        save_settings()
        await interaction.followup.send(f"🗑️ 已移除: `{', '.join(to_remove)}`")

@client.tree.command(name="watch_list", description="查看我的关注列表")
async def watch_list(interaction: discord.Interaction):
    stocks = get_user_data(interaction.user.id)["stocks"]
    if len(stocks) > 60:
        display_str = ", ".join(stocks[:60]) + f"... (共 {len(stocks)} 只)"
    else:
        display_str = ", ".join(stocks) if stocks else '空'
    await interaction.response.send_message(f"📋 **当前关注**:\n`{display_str}`", ephemeral=True)

@client.tree.command(name="watch_price", description="获取关注列表的实时行情")
async def watch_price(interaction: discord.Interaction):
    stocks = get_user_data(interaction.user.id)["stocks"]
    if not stocks:
        await interaction.response.send_message("📭 关注列表为空，请先使用 `/watch_add` 添加。", ephemeral=True)
        return

    await interaction.response.defer()
    
    # 获取报价 (调用新的 fetch_fmp_quotes)
    quotes = await fetch_fmp_quotes(stocks)
    
    if not quotes:
        await interaction.followup.send("❌ 无法获取数据 (API错误或代码无效)。")
        return

    embed = discord.Embed(title="📈 实时行情 (Watchlist)", color=0x00ff00)
    embed.set_footer(text="Data provided by Financial Modeling Prep")
    
    msg_lines = []
    for q in quotes:
        symbol = q.get('symbol')
        price = q.get('price')
        change_p = q.get('changesPercentage')
        
        icon = "🟢" if change_p and change_p > 0 else "🔴"
        if change_p == 0: icon = "⚪"
        
        line = f"{icon} **{symbol}**: `${price}` ({change_p}%)"
        msg_lines.append(line)

    full_text = "\n".join(msg_lines)
    if len(full_text) > 4000:
        full_text = full_text[:4000] + "\n... (列表过长截断)"
        
    embed.description = full_text
    await interaction.followup.send(embed=embed)

@client.tree.command(name="test", description="立即测试股票")
async def test_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    
    # 测试时也复用 batch 逻辑，虽然只有一个
    data_map = await fetch_historical_batch([ticker])
    if not data_map or ticker not in data_map:
        await interaction.followup.send(f"❌ 获取 `{ticker}` 失败。")
        return
        
    df = data_map[ticker]
    is_triggered, reason, level, res_line, sup_line = check_signals(df)
    
    chart_file = await generate_chart(df, ticker, res_line, sup_line)
    last_row = df.iloc[-1]
    
    msg = (
        f"✅ **接口测试正常** | `{ticker}`\n"
        f"📊 **信号状态**: {level}\n"
        f"💰 收盘: `${last_row['close']:.2f}`\n"
        f"🌊 Nx蓝梯: `${last_row['Nx_Blue_DW']:.2f}` ~ `${last_row['Nx_Blue_UP']:.2f}`\n"
        f"📝 **触发理由**: \n{reason if reason else '无触发'}"
    )
    
    try:
        file = discord.File(chart_file)
        await interaction.followup.send(content=msg, file=file)
    except Exception as e:
        await interaction.followup.send(f"❌ 发送失败: {e}")
    finally:
        if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    if DISCORD_TOKEN:
        client.run(DISCORD_TOKEN)
