import discord
from discord import app_commands
from discord.ext import tasks
import requests
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
import aiohttp # [新增] 用于异步请求 FMP 价格

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

# --- [新增] FMP 实时价格查询工具 ---
async def fetch_fmp_quotes(symbols: list):
    """批量获取 FMP 实时报价"""
    if not symbols: return []
    
    # FMP 支持逗号分隔，建议一次不要超过 50-100 个，这里做简单分片处理
    chunk_size = 50
    all_quotes = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}?apikey={FMP_API_KEY}"
            
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list):
                            all_quotes.extend(data)
            except Exception as e:
                print(f"Error fetching quotes: {e}")
                
    return all_quotes

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

# --- [重写] 机构级形态识别算法 ---
def identify_patterns(df, window=60):
    """
    严格版形态识别：旗形/楔形突破
    修正：使用索引进行线性回归，避免日期非线性导致的误差
    """
    if len(df) < window + 5: return None, [], []
    
    recent = df.tail(window).copy()
    # 重置索引为 0, 1, 2... 以进行准确的线性回归
    recent = recent.reset_index() 
    
    # 寻找局部极值
    recent['pivot_high'] = recent['high'].rolling(5, center=True).max() == recent['high']
    recent['pivot_low'] = recent['low'].rolling(5, center=True).min() == recent['low']
    
    high_points = recent[recent['pivot_high']]
    low_points = recent[recent['pivot_low']]
    
    # --- 策略: 旗形/楔形收敛突破 ---
    if len(high_points) >= 3 and len(low_points) >= 3:
        # 取最后3个点来判断趋势更稳
        h_last = high_points.iloc[-1]
        h_prev = high_points.iloc[-3] # 取间隔一个高点，跨度更大
        l_last = low_points.iloc[-1]
        l_prev = low_points.iloc[-3]
        
        # 1. 压力线向下倾斜 (高点降低)
        if h_last['high'] < h_prev['high']:
            # 计算压力线斜率 (y = mx + b)
            slope_res = (h_last['high'] - h_prev['high']) / (h_last.name - h_prev.name)
            intercept_res = h_prev['high'] - slope_res * h_prev.name
            
            # 计算支撑线斜率
            slope_sup = (l_last['low'] - l_prev['low']) / (l_last.name - l_prev.name)
            
            # 2. 收敛形态: 支撑线斜率 > 压力线斜率 (这就构成了收敛)
            # 且收敛角度不能太小
            if slope_sup > slope_res and (slope_sup - slope_res) > 0.05:
                
                # 3. 计算今天的理论阻力位
                curr_idx = recent.index[-1]
                resistance_today = slope_res * curr_idx + intercept_res
                
                curr_close = recent['close'].iloc[-1]
                curr_vol = recent['volume'].iloc[-1]
                vol_ma = recent['Vol_MA20'].iloc[-1]
                
                # 4. 突破前一根K线必须在通道内 (防止已经是突破后的第N天)
                prev_close = recent['close'].iloc[-2]
                prev_idx = recent.index[-2]
                resistance_prev = slope_res * prev_idx + intercept_res
                
                if prev_close <= resistance_prev:
                    # 5. 突破 + 放量
                    if curr_close > resistance_today and curr_vol > vol_ma * 1.3:
                        # 转换回原始 DataFrame 的时间索引用于画图
                        t1 = recent['date'].iloc[h_prev.name]
                        p1 = h_prev['high']
                        t2 = recent['date'].iloc[-1]
                        p2 = resistance_today
                        
                        t3 = recent['date'].iloc[l_prev.name]
                        p3 = l_prev['low']
                        t4 = recent['date'].iloc[l_last.name]
                        p4 = l_last['low']
                        
                        # 返回两根线：压力线(白) 和 支撑线(辅助)
                        # 格式: [[(d1,p1), (d2,p2)], [(d3,p3), (d4,p4)]]
                        return "🚩 **放量旗形突破**: 机构级信号 (收敛+放量)", [[(t1,p1), (t2,p2)]], [[(t3,p3), (t4,p4)]]

    return None, [], []

def check_signals(df):
    if len(df) < 60: return False, "", "NONE", [], []
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    triggers = []
    level = "NORMAL"
    
    # 基础过滤: 剔除已暴涨股 (60日涨幅过大)
    low_60 = df['low'].tail(60).min()
    if curr['close'] > low_60 * 1.4: return False, "", "NONE", [], []

    # --- 1. 识别形态 (新增优化) ---
    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name:
        triggers.append(pattern_name)
        level = "S_TIER" # 机构级形态 S 级

    # --- 2. [新增] 突破后回踩不破 (二次确认神级策略) ---
    # 逻辑：过去 10 天曾经突破过蓝色梯子，但最近几天回调到了蓝色梯子附近，且今天再次放量上涨
    # 这是一个非常棒的“上车点”
    recent_10 = df.tail(10)
    # 检查是否有某天收盘 > 上沿
    had_breakout = (recent_10['close'] > recent_10['Nx_Blue_UP']).any()
    
    # 当前刚好在梯子附近 (支撑位)
    on_support = curr['close'] > curr['Nx_Blue_DW'] and curr['low'] <= curr['Nx_Blue_UP'] * 1.02
    
    # 再次放量启动
    re_volume = curr['volume'] > curr['Vol_MA20'] * 1.5
    
    if had_breakout and on_support and re_volume:
        triggers.append(f"🚀 **二次起爆**: 突破回踩确认支撑，放量拉升！")
        level = "GOD_TIER" # 比 S 还高一级

    # --- 3. Nx 趋势 (基础) ---
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        triggers.append(f"📈 **Nx 突破**: 站稳蓝色牛熊线")
            
    # --- 4. Cd/MACD 底背离 ---
    low_20 = df['low'].tail(20).min()
    price_is_low = curr['low'] <= low_20 * 1.01
    dif_20_min = df['DIF'].tail(20).min()
    divergence = curr['DIF'] > dif_20_min 
    momentum_turn = curr['MACD'] > prev['MACD']
    
    if price_is_low and divergence and momentum_turn:
        if is_downtrend or curr['RSI'] < 35:
             triggers.append(f"💎 **Cd 结构底背离**: 股价新低但指标背离")

    # --- 5. 弘历直接买 ---
    if prev['RSI'] < 30 and curr['RSI'] > 30:
        if is_downtrend and "Cd" not in str(triggers):
            triggers.append(f"⚠️ **RSI 超卖反弹**: 趋势仍偏空")
        else:
            triggers.append(f"🚀 **弘历战法**: RSI金叉")
            
    if triggers:
        # 过滤弱信号
        if is_downtrend and len(triggers) < 2 and "S_TIER" not in level and "GOD_TIER" not in level:
            return False, "", "NONE", [], []
        return True, "\n".join(triggers), level, res_line, sup_line

    return False, "", "NONE", [], []

def generate_chart(df, ticker, res_line=[], sup_line=[]):
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
    
    # 修复画线逻辑：确保传入的是 list of lists of tuples
    # 并且只在 plot_df 范围内画，虽然 mplfinance 会自动裁剪，但为了安全
    
    lines_to_draw = []
    if res_line: lines_to_draw.extend(res_line) # res_line 本身已经是 [[(t1,p1), (t2,p2)]] 格式
    if sup_line: lines_to_draw.extend(sup_line)
    
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
        # 正确写法：list of lists
        kwargs['alines'] = dict(alines=lines_to_draw, colors='white', linewidths=1.5, linestyle='--')

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
            
        if isinstance(data, list):
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
        curr_time, today_str = now_et.time(), now_et.strftime('%Y-%m-%d')
        
        is_pre = TIME_PRE_MARKET_START <= curr_time < TIME_MARKET_OPEN
        is_open = TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE
        
        if not (is_pre or is_open): return
        
        print(f"[{now_et.strftime('%H:%M')}] Scanning...")
        ticker_user_map = defaultdict(list)
        users_data = settings.get("users", {})
        
        for uid, udata in users_data.items():
            for k in list(udata['daily_status'].keys()):
                if not k.endswith(today_str): del udata['daily_status'][k]
            for ticker in udata.get("stocks", []):
                ticker_user_map[ticker].append(uid)

        for ticker, user_ids in ticker_user_map.items():
            all_alerted = True
            for uid in user_ids:
                status_key = f"{ticker}-{today_str}"
                status = users_data[uid]['daily_status'].get(status_key, "NONE")
                if is_pre and status == "NONE": all_alerted = False
                if is_open and status not in ["MARKET_SENT", "BOTH_SENT"]: all_alerted = False
            
            if all_alerted: continue

            df = get_stock_data(ticker)
            if df is None:
                # 核心修复：使用异步 sleep 防止卡死
                await asyncio.sleep(1)
                continue

            is_triggered, reason, level, res_line, sup_line = check_signals(df)
            
            if is_triggered:
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
                    emoji = "👑" if level == "GOD_TIER" else ("🔥" if level == "S_TIER" else "🚨")
                    header = f"【{emoji} 神级K线系统】"
                    
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
            
            # 核心修复：异步 sleep
            await asyncio.sleep(1.2)

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

@client.tree.command(name="addstocks", description="添加关注股票 (legacy)")
async def add_stocks(interaction: discord.Interaction, tickers: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in tickers.replace(',', ' ').split() if t.strip()]))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已添加！新增: `{', '.join(new_list)}`")

@client.tree.command(name="liststocks", description="查看关注列表 (legacy)")
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

# --- [新增] Watch 系列命令 (操作同样的数据源) ---

@client.tree.command(name="watch_add", description="批量添加关注 (例如: AAPL, TSLA)")
@app_commands.describe(codes="股票代码，用逗号或空格分隔")
async def watch_add(interaction: discord.Interaction, codes: str):
    # 复用 addstocks 的逻辑，保持数据一致
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in codes.replace(',', ' ').replace('，', ' ').split() if t.strip()]))
    
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    
    await interaction.followup.send(f"✅ 已关注: `{', '.join(new_list)}` (同时也加入了自动监控队列)")

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
    # 复用 list_stocks 逻辑
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
    
    # 获取报价
    quotes = await fetch_fmp_quotes(stocks)
    
    if not quotes:
        await interaction.followup.send("❌ 无法获取数据 (API错误或代码无效)。")
        return

    # 构建 Embed 表格
    embed = discord.Embed(title="📈 实时行情 (Watchlist)", color=0x00ff00)
    embed.set_footer(text="Data provided by Financial Modeling Prep")
    
    # 简单的文本排版
    msg_lines = []
    for q in quotes:
        symbol = q.get('symbol')
        price = q.get('price')
        change_p = q.get('changesPercentage')
        
        # 图标逻辑
        icon = "🟢" if change_p and change_p > 0 else "🔴"
        if change_p == 0: icon = "⚪"
        
        # 格式化
        line = f"{icon} **{symbol}**: `${price}` ({change_p}%)"
        msg_lines.append(line)

    # Discord Embed 有 4096 字符限制，如果太长需要截断
    full_text = "\n".join(msg_lines)
    if len(full_text) > 4000:
        full_text = full_text[:4000] + "\n... (列表过长截断)"
        
    embed.description = full_text
    await interaction.followup.send(embed=embed)

@client.tree.command(name="test", description="立即测试股票")
async def test_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    df = get_stock_data(ticker)
    
    if df is None:
        await interaction.followup.send(f"❌ 获取 `{ticker}` 失败。")
        return
        
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
