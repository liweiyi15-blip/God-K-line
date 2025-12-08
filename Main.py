import discord
from discord import app_commands
from discord.ext import tasks
import requests
import json
import os
from datetime import datetime, time, timedelta
import time as time_module
import pandas as pd
import numpy as np
import mplfinance as mpf
import pytz
from dotenv import load_dotenv
from collections import defaultdict
from scipy.stats import linregress # 新增：用于计算趋势线斜率

# --- 加载环境变量 ---
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
FMP_API_KEY = os.getenv("FMP_API_KEY")
try:
    ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID"))
except (TypeError, ValueError):
    ALERT_CHANNEL_ID = 0 

# --- 全局常量 ---
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

# --- 形态识别核心算法 (新) ---

def identify_patterns(df, window=20):
    """
    识别旗形/楔形突破和双底
    返回: (pattern_name, upper_line_points, lower_line_points)
    points 格式: [(date1, price1), (date2, price2)] 用于绘图
    """
    # 需要至少一定数量的数据
    if len(df) < 60: return None, [], []
    
    # 1. 寻找局部高点和低点 (Pivot Points)
    # 使用 rolling window 寻找局部极值
    df['max_local'] = df['high'].rolling(window=10, center=True).max()
    df['min_local'] = df['low'].rolling(window=10, center=True).min()
    
    # 提取最近的两个显著高点 (High1, High2) 和 低点 (Low1, Low2)
    # 简单逻辑：取最近 60 天内，不仅是局部最大，而且比较突出的点
    recent_df = df.tail(60).copy()
    
    # 获取高点索引
    high_idxs = recent_df[recent_df['high'] == recent_df['max_local']].index
    low_idxs = recent_df[recent_df['low'] == recent_df['min_local']].index
    
    if len(high_idxs) < 2 or len(low_idxs) < 2:
        return None, [], []
        
    # 取最后两个高点和低点
    h2_date, h1_date = high_idxs[-1], high_idxs[-2] # h2 是最新的
    l2_date, l1_date = low_idxs[-1], low_idxs[-2]
    
    h2_val, h1_val = recent_df.loc[h2_date]['high'], recent_df.loc[h1_date]['high']
    l2_val, l1_val = recent_df.loc[l2_date]['low'], recent_df.loc[l1_date]['low']
    
    # --- 策略 A: 旗形/楔形突破 (Flag/Wedge Breakout) ---
    # 条件：高点降低 (压力线下倾)，低点抬高或持平 (收敛)，且当前价格突破压力线
    
    # 计算压力线 (连接 h1 和 h2) 在“今天”的理论价格
    # y = mx + c
    # 把日期转为数字进行线性回归
    x_h = np.array([(d - h1_date).days for d in [h1_date, h2_date]])
    y_h = np.array([h1_val, h2_val])
    slope_h, intercept_h, _, _, _ = linregress(x_h, y_h)
    
    # 今天的 X 坐标
    today_date = df.index[-1]
    days_diff = (today_date - h1_date).days
    resistance_price_today = slope_h * days_diff + intercept_h
    
    current_close = df['close'].iloc[-1]
    
    # 判定 1: 压力线必须是向下倾斜的 (slope_h < 0) 或者是平的
    # 判定 2: 当前价格 突破了 压力线
    if slope_h < 0 and current_close > resistance_price_today * 1.005: # 突破 0.5%
        return "🚩 **旗形/楔形突破**", [(h1_date, h1_val), (today_date, resistance_price_today)], []

    # --- 策略 B: 双底回踩不破 (Double Bottom Support) ---
    # 条件: l1 和 l2 价格接近 (5%以内)，且当前价格在 l2 附近
    # l1 是左底，l2 是右底
    if abs(l1_val - l2_val) / l1_val < 0.05:
        # 当前价格距离右底不远 (比如 3% 以内)，且是红盘(涨势)或缩量
        if current_close > l2_val and (current_close - l2_val)/l2_val < 0.03:
             return "⚓ **双底支撑回踩**", [], [(l1_date, l1_val), (l2_date, l2_val)]

    return None, [], []

# --- 核心指标计算 ---
def calculate_nx_indicators(df):
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
    
    return df

def check_signals(df):
    if len(df) < 60: return False, "", "NONE", [], []
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    triggers = []
    level = "NORMAL"
    
    # 基础过滤
    low_60 = df['low'].tail(60).min()
    if curr['close'] > low_60 * 1.4: return False, "", "NONE", [], []

    # 1. 识别形态 (新增)
    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name:
        triggers.append(pattern_name)
        level = "S_TIER" # 形态突破通常很重要

    # 2. Nx 趋势
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    is_shrinking_vol = curr['volume'] < (curr['Vol_MA20'] * 0.7)
    
    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        triggers.append(f"📈 **Nx 突破**: 站稳蓝色牛熊线")
            
    # 3. 缩量回踩
    dist_to_support = abs(curr['close'] - curr['Nx_Blue_DW']) / curr['Nx_Blue_DW']
    if curr['close'] > curr['Nx_Yellow_UP'] and dist_to_support < 0.015 and curr['close'] < prev['close']:
        if is_shrinking_vol:
            triggers.append(f"🛡️ **缩量回踩**: 回调至蓝色支撑且缩量")
            level = "S_TIER"

    # 4. Cd/MACD
    low_20 = df['low'].tail(20).min()
    price_is_low = curr['low'] <= low_20 * 1.01
    dif_20_min = df['DIF'].tail(20).min()
    divergence = curr['DIF'] > dif_20_min
    momentum_turn = curr['MACD'] > prev['MACD']
    
    if price_is_low and divergence and momentum_turn:
        if is_downtrend or curr['RSI'] < 35:
             triggers.append(f"💎 **Cd 底背离**: 股价新低指标背离")

    # 5. 弘历
    if prev['RSI'] < 30 and curr['RSI'] > 30:
        if is_downtrend and "Cd" not in str(triggers):
            triggers.append(f"⚠️ **RSI 反弹**: 趋势仍偏空")
        else:
            triggers.append(f"🚀 **弘历战法**: RSI金叉")
            
    if triggers:
        # 如果只有普通信号且在跌势中，过滤
        if is_downtrend and len(triggers) < 2 and "Cd" not in str(triggers) and "突破" not in str(triggers):
            return False, "", "NONE", [], []
        return True, "\n".join(triggers), level, res_line, sup_line

    return False, "", "NONE", [], []

def generate_chart(df, ticker, res_line=[], sup_line=[]):
    filename = f"{ticker}_alert.png"
    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    my_style = mpf.make_mpf_style(base_mpl_style="ggplot", marketcolors=s, gridstyle=":")
    
    plot_df = df.tail(80)
    
    # 基础指标线
    add_plots = [
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['MACD'], panel=2, type='bar', color='dimgray', alpha=0.5, ylabel='MACD'),
    ]
    
    # 如果检测到形态，把趋势线画上去 (TrendSpider 风格)
    # alines 需要 list of list of tuples: [[(t1,p1), (t2,p2)], ...]
    lines_to_draw = []
    if res_line: lines_to_draw.append(res_line)
    if sup_line: lines_to_draw.append(sup_line)
    
    kwargs = dict(
        type='candle', 
        style=my_style, 
        title=f"{ticker} Analysis", 
        ylabel='Price', 
        addplot=add_plots, 
        volume=True, 
        panel_ratios=(6, 2, 2), 
        savefig=filename
    )
    
    if lines_to_draw:
        # 添加趋势线
        kwargs['alines'] = dict(alines=lines_to_draw, colors=['white'], linewidths=1.5, linestyle='-')

    mpf.plot(plot_df, **kwargs)
    return filename

# --- 数据获取 (400天) ---

def get_stock_data(ticker, days=200):
    now = datetime.now()
    end_date_str = now.strftime("%Y-%m-%d")
    start_date_str = (now - timedelta(days=400)).strftime("%Y-%m-%d")
    
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={ticker}&from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
    )
    
    print(f"🔍 [Debug] Requesting {ticker}...")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return None
        data = response.json()
        if not data: return None
            
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'historical' in data:
            df = pd.DataFrame(data['historical'])
        else:
            return None

        if df.empty: return None

        df = df.set_index('date').sort_index(ascending=True)
        df.index = pd.to_datetime(df.index)
        return calculate_nx_indicators(df)
    except Exception as e:
        print(f"❌ [Exception] {e}")
        return None

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
        curr_time = now_et.time()
        
        is_pre = TIME_PRE_MARKET_START <= curr_time < TIME_MARKET_OPEN
        is_open = TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE
        
        if not (is_pre or is_open): return
        
        print(f"[{now_et.strftime('%H:%M')}] Scanning...")
        ticker_user_map = defaultdict(list)
        users_data = settings.get("users", {})
        today_str = now_et.strftime('%Y-%m-%d')
        
        for uid, udata in users_data.items():
            for k in list(udata['daily_status'].keys()):
                if not k.endswith(today_str): del udata['daily_status'][k]
            for ticker in udata.get("stocks", []):
                ticker_user_map[ticker].append(uid)

        for ticker, user_ids in ticker_user_map.items():
            # 状态检查
            all_alerted = True
            for uid in user_ids:
                status_key = f"{ticker}-{today_str}"
                status = users_data[uid]['daily_status'].get(status_key, "NONE")
                if is_pre and status == "NONE": all_alerted = False
                if is_open and status not in ["MARKET_SENT", "BOTH_SENT"]: all_alerted = False
            
            if all_alerted: continue

            df = get_stock_data(ticker)
            if df is None:
                time_module.sleep(1)
                continue

            # 这里的 check_signals 返回值增加了 res_line, sup_line
            is_triggered, reason, level, res_line, sup_line = check_signals(df)
            
            if is_triggered:
                # 传入画线数据
                chart_file = generate_chart(df, ticker, res_line, sup_line)
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
                    header = "【🚨 神级K线系统】" + (" 🔥 S级信号" if level == "S_TIER" else "")
                    msg = (
                        f"{mentions}\n{header}\n"
                        f"🎯 **标的**: `{ticker}` | 💰 **现价**: `${price:.2f}`\n"
                        f"{'-'*25}\n{reason}\n{'-'*25}\n"
                        f"🌊 **Nx 蓝梯下沿**: `${nx_support:.2f}`"
                    )
                    try:
                        file = discord.File(chart_file)
                        await self.alert_channel.send(content=msg, file=file)
                    except Exception as e:
                        print(f"Error: {e}")
                    finally:
                        if os.path.exists(chart_file): os.remove(chart_file)
            time_module.sleep(1.2)

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

@client.tree.command(name="addstocks", description="添加关注股票")
async def add_stocks(interaction: discord.Interaction, tickers: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in tickers.replace(',', ' ').split() if t.strip()]))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已添加！新增: `{', '.join(new_list)}`")

@client.tree.command(name="liststocks", description="查看关注列表")
async def list_stocks(interaction: discord.Interaction):
    stocks = get_user_data(interaction.user.id)["stocks"]
    if len(stocks) > 60:
        display_str = ", ".join(stocks[:60]) + f"... (共 {len(stocks)} 只)"
    else:
        display_str = ", ".join(stocks) if stocks else '空'
    await interaction.response.send_message(f"📋 **关注列表**:\n`{display_str}`", ephemeral=True)

@client.tree.command(name="clearstocks", description="清空关注列表")
async def clear_stocks(interaction: discord.Interaction):
    user_data = get_user_data(interaction.user.id)
    user_data["stocks"] = []
    user_data["daily_status"] = {}
    save_settings()
    await interaction.response.send_message("🗑️ 已清空。", ephemeral=True)

@client.tree.command(name="test", description="立即测试股票")
async def test_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    df = get_stock_data(ticker)
    
    if df is None:
        await interaction.followup.send(f"❌ 获取 `{ticker}` 失败。")
        return
        
    # 测试命令也要画线，所以要调用 check_signals 获取坐标
    _, _, _, res_line, sup_line = check_signals(df)
    
    chart_file = generate_chart(df, ticker, res_line, sup_line)
    last_row = df.iloc[-1]
    
    msg = (
        f"✅ **接口测试正常** | `{ticker}`\n"
        f"💰 收盘: `${last_row['close']:.2f}`\n"
        f"🌊 Nx蓝梯: `${last_row['Nx_Blue_DW']:.2f}` ~ `${last_row['Nx_Blue_UP']:.2f}`\n"
        f"📉 RSI: `{last_row['RSI']:.2f}`"
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
