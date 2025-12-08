import discord
from discord import app_commands
from discord.ext import tasks
import requests
import json
import os
from datetime import datetime, time
import time as time_module
import pandas as pd
import numpy as np
import mplfinance as mpf
import pytz
from dotenv import load_dotenv

# --- 加载环境变量 ---
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
FMP_API_KEY = os.getenv("FMP_API_KEY")
try:
    ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID"))
except (TypeError, ValueError):
    ALERT_CHANNEL_ID = 0 

# --- 全局常量 ---
SETTINGS_FILE = "settings.json"
MARKET_TIMEZONE = pytz.timezone('America/New_York')

# 定义时间点 (纽约当地时间)
TIME_PRE_MARKET_START = time(9, 0)
TIME_MARKET_OPEN = time(9, 30)
TIME_MARKET_CLOSE = time(16, 0)

# --- 全局变量 ---
settings = {}

# --- 辅助函数：设置持久化 ---
def load_settings():
    global settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            settings = {"MONITORED_STOCKS": [], "DAILY_STATUS": {}}
            save_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        settings = {"MONITORED_STOCKS": [], "DAILY_STATUS": {}}

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

# --- 核心指标计算算法 ---

def calculate_nx_indicators(df):
    """
    计算 Nx (牛熊分界) 和 Cd (MACD背离) 指标
    """
    # 1. Nx 指标计算 (EMA 通道) 
    # 蓝色梯子 (短期) TF1=24
    df['Nx_Blue_UP'] = df['high'].ewm(span=24, adjust=False).mean()
    df['Nx_Blue_DW'] = df['low'].ewm(span=23, adjust=False).mean() # TF1-1
    
    # 黄色梯子 (长期) TF2=90
    df['Nx_Yellow_UP'] = df['high'].ewm(span=89, adjust=False).mean() # TF2-1
    df['Nx_Yellow_DW'] = df['low'].ewm(span=90, adjust=False).mean()

    # 2. Cd 指标核心: MACD [cite: 15]
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # 3. RSI 计算 (用于弘历战法)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def check_signals(df):
    """
    综合判断信号 (Nx + Cd + Hongli)
    返回: (Is_Trigger, Reason_String, Signal_Level)
    """
    if len(df) < 30:
        return False, "", "NONE"

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    triggers = []
    level = "NORMAL"

    # --- 1. Nx 趋势判断 (权重 80%)  ---
    # 突破蓝色梯子上沿 (最佳买点)
    nx_breakout = prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']
    is_above_blue = curr['close'] > curr['Nx_Blue_DW']
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] # 完全在蓝色梯子下方
    
    if nx_breakout:
        triggers.append(f"🔥 **Nx 突破**: 收盘价突破蓝色牛熊分界线 (${curr['Nx_Blue_UP']:.2f}) -> **加仓信号**")
        level = "S_TIER" # 神级信号

    # --- 2. Cd/MACD 背离判断 (抄底)  ---
    # 简化逻辑：股价创新低(近20天)，但 DIF 没创新低
    # 仅当不在严重下跌趋势中，或出现严重超卖时才提示
    low_20 = df['low'].tail(20).min()
    dif_20 = df['DIF'].tail(20).min()
    
    is_price_low = curr['low'] <= low_20
    is_dif_higher = curr['DIF'] > dif_20
    macd_gold = prev['DIF'] < prev['DEA'] and curr['DIF'] > curr['DEA'] # 金叉

    cd_divergence = is_price_low and is_dif_higher and macd_gold
    
    if cd_divergence:
        # 过滤：如果还是并排向下 (is_downtrend)，除非 RSI 极低否则不报 
        if is_downtrend and curr['RSI'] > 25:
             pass # 忽略无效抄底
        else:
             triggers.append(f"💎 **Cd 抄底**: MACD 底背离且金叉 (RSI: {curr['RSI']:.1f})")

    # --- 3. 弘历直接买 (RSI < 30 反转) ---
    rsi_buy = prev['RSI'] < 30 and curr['RSI'] > 30
    if rsi_buy:
        if is_downtrend and not cd_divergence:
            triggers.append(f"⚠️ **RSI 反弹**: 超卖反弹 (趋势仍偏空，注意风险)")
        else:
            triggers.append(f"🚀 **弘历战法**: RSI(14) 从超卖区金叉向上")

    # --- 综合决策 ---
    if triggers:
        # 如果是 S_TIER (Nx 突破) 直接发
        if level == "S_TIER":
            return True, "\n".join(triggers), "S_TIER"
        
        # 如果只是普通信号，确保不是在深跌中接飞刀
        # 只有 "Cd背离" 或者 "RSI反转" 且 "不在深跌" 或 "双重共振" 才发
        if is_downtrend and len(triggers) < 2:
            return False, "", "NONE" # 过滤掉
            
        return True, "\n".join(triggers), "NORMAL"

    return False, "", "NONE"

# --- 绘图函数 (包含 Nx 梯子) ---

def generate_chart(df, ticker):
    filename = f"{ticker}_analysis.png"
    
    # 设置样式
    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    my_style = mpf.make_mpf_style(base_mpl_style="seaborn", marketcolors=s, gridstyle=":")

    # 构建绘图数据 (最近 60 天)
    plot_df = df.tail(60)

    # 添加 Nx 通道和 MACD
    add_plots = [
        # Nx Blue Ladder (Short Term) 
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        # Nx Yellow Ladder (Long Term) [cite: 99]
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        # MACD Panel
        mpf.make_addplot(plot_df['MACD'], panel=2, type='bar', color='dimgray', alpha=0.5, ylabel='MACD'),
        mpf.make_addplot(plot_df['DIF'], panel=2, color='orange'),
        mpf.make_addplot(plot_df['DEA'], panel=2, color='blue'),
    ]

    title = f"{ticker} God-Tier Analysis (Nx Trend + Cd Signal)"
    
    mpf.plot(
        plot_df,
        type='candle',
        style=my_style,
        title=title,
        ylabel='Price ($)',
        addplot=add_plots,
        volume=True,
        panel_ratios=(6, 2, 2), # K线:成交量:MACD
        savefig=filename
    )
    return filename

# --- 数据获取 ---

def get_stock_data(ticker, days=200):
    # 需要足够的数据来计算 EMA90
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-chart/daily/{ticker}"
        f"?apikey={FMP_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if not data: return None

        df = pd.DataFrame(data).set_index('date').sort_index(ascending=True)
        df.index = pd.to_datetime(df.index)
        df = df.tail(days)
        
        return calculate_nx_indicators(df)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# --- Discord Bot Logic (保持之前的时间控制) ---

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
        today_str = now_et.strftime('%Y-%m-%d')

        is_pre = TIME_PRE_MARKET_START <= curr_time < TIME_MARKET_OPEN
        is_open = TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE

        if not (is_pre or is_open): return

        print(f"[{now_et.strftime('%H:%M')}] Scanning markets...")
        
        stocks = settings.get("MONITORED_STOCKS", [])
        daily_status = settings.get("DAILY_STATUS", {})
        
        # 清理旧状态
        for k in list(daily_status.keys()):
            if not k.endswith(today_str): del daily_status[k]

        for ticker in stocks:
            status_key = f"{ticker}-{today_str}"
            status = daily_status.get(status_key, "NONE")

            # 频率控制: 盘前1次，盘中1次
            if is_pre and status != "NONE": continue
            if is_open and status in ["MARKET_SENT", "BOTH_SENT"]: continue

            df = get_stock_data(ticker)
            if df is None: 
                time_module.sleep(1)
                continue

            # Check Logic
            triggered, reason, level = check_signals(df)

            if triggered:
                chart_file = generate_chart(df, ticker)
                price = df['close'].iloc[-1]
                
                # 构造消息
                header = "【🚨 🚨🚨神级K线分析系统】"
                if level == "S_TIER": header += " 🔥 趋势突破!"
                
                msg = (
                    f"{header}\n"
                    f"🎯 **标的**: `{ticker}`\n"
                    f"💰 **现价**: `${price:.2f}`\n"
                    f"------------------------\n"
                    f"{reason}\n"
                    f"------------------------\n"
                    f"📚 **操作指引**:\n"
                    f"1. 若提示 **Nx 突破**，收盘确认为最佳买点 (加仓)。\n"
                    f"2. 若提示 **Cd 抄底**，仅在股价靠近蓝色梯子或 RSI 极低时操作。\n"
                    f"3. 蓝色梯子下沿: ${df['Nx_Blue_DW'].iloc[-1]:.2f} (跌破注意风控)"
                )

                try:
                    with discord.File(chart_file) as file:
                        await self.alert_channel.send(content=msg, file=file)
                    
                    # 更新状态
                    new_status = "PRE_SENT" if is_pre else ("BOTH_SENT" if status == "PRE_SENT" else "MARKET_SENT")
                    settings["DAILY_STATUS"][status_key] = new_status
                    save_settings()
                    print(f"Alert sent for {ticker}")
                except Exception as e:
                    print(f"Error sending {ticker}: {e}")
                finally:
                    if os.path.exists(chart_file): os.remove(chart_file)
            
            time_module.sleep(1.5)

    @self.tree.command(name="addstocks", description="添加监控股票")
    async def add_stocks(self, interaction: discord.Interaction, tickers: str):
        await interaction.response.defer()
        s_list = list(set([t.strip().upper() for t in tickers.replace(',', ' ').split() if t.strip()]))
        settings["MONITORED_STOCKS"] = s_list
        settings["DAILY_STATUS"] = {} 
        save_settings()
        await interaction.followup.send(f"✅ 已更新神级监控列表: {len(s_list)} 只股票")

if __name__ == "__main__":
    if DISCORD_TOKEN:
        StockBotClient(intents=discord.Intents.default()).run(DISCORD_TOKEN)
