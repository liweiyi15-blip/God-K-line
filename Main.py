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
    # 如果环境变量未设置，设为 0，并在运行时打印警告
    ALERT_CHANNEL_ID = 0 

# --- 全局常量 ---
SETTINGS_FILE = "settings.json"
# 自动处理冬夏令时的纽约时间
MARKET_TIMEZONE = pytz.timezone('America/New_York')

# 定义时间点 (纽约当地时间)
TIME_PRE_MARKET_START = time(9, 0)  # 盘前监控开始
TIME_MARKET_OPEN = time(9, 30)      # 开盘
TIME_MARKET_CLOSE = time(16, 0)     # 收盘

# --- 全局变量 ---
settings = {}

# --- 辅助函数：设置持久化 ---

def load_settings():
    """从文件中加载设置，如果文件不存在则创建默认设置"""
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
    """将当前设置保存到文件"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

# --- 核心指标计算算法 (Nx + Cd + Hongli) ---

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

    # 2. Cd 指标核心: MACD
    # 使用 close 计算
    price_col = 'close' 
    exp12 = df[price_col].ewm(span=12, adjust=False).mean()
    exp26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # 3. RSI 计算 (用于弘历战法)
    delta = df[price_col].diff()
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

    # --- 1. Nx 趋势判断 (权重 80%) ---
    # 逻辑：收盘价从下方向上突破蓝色梯子上沿 -> 最佳买点
    nx_breakout = prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] # 完全在蓝色梯子下方
    
    if nx_breakout:
        triggers.append(f"🔥 **Nx 突破**: 收盘价突破蓝色牛熊分界线 (${curr['Nx_Blue_UP']:.2f}) -> **加仓信号**")
        level = "S_TIER" # 神级信号

    # --- 2. Cd/MACD 背离判断 (抄底) ---
    # 逻辑：股价创新低(近20天)，但 DIF 没创新低
    low_20 = df['low'].tail(20).min()
    dif_20 = df['DIF'].tail(20).min()
    
    # 判定股价接近新低 (容差 1%)
    is_price_low = curr['low'] <= low_20 * 1.01 
    # 判定指标没有新低 (底背离)
    is_dif_higher = curr['DIF'] > dif_20
    # 判定金叉 (趋势转折)
    macd_gold = prev['DIF'] < prev['DEA'] and curr['DIF'] > curr['DEA']

    cd_divergence = is_price_low and is_dif_higher and macd_gold
    
    if cd_divergence:
        # 过滤：如果处于下跌趋势中(梯子下方)，且 RSI 还不够低(>30)，则过滤掉弱背离
        if is_downtrend and curr['RSI'] > 30:
             pass 
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
        
        # 如果是下跌趋势中，必须有两个以上信号或者是背离信号才发，防止接飞刀
        if is_downtrend and len(triggers) < 2 and "Cd 抄底" not in str(triggers):
            return False, "", "NONE"
            
        return True, "\n".join(triggers), "NORMAL"

    return False, "", "NONE"

# --- 绘图函数 (包含 Nx 梯子) ---

def generate_chart(df, ticker):
    filename = f"{ticker}_analysis.png"
    
    # 设置样式
    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    my_style = mpf.make_mpf_style(base_mpl_style="seaborn", marketcolors=s, gridstyle=":")

    # 构建绘图数据 (最近 60 天，让图表更清晰)
    plot_df = df.tail(60)

    # 添加 Nx 通道和 MACD
    add_plots = [
        # Nx Blue Ladder (Short Term) - 蓝色梯子
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        # Nx Yellow Ladder (Long Term) - 黄色梯子
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
        panel_ratios=(6, 2, 2), # K线:成交量:MACD 高度比例
        savefig=filename
    )
    return filename

# --- 数据获取 (FMP Stable Interface) ---

def get_stock_data(ticker, days=200):
    """
    使用 FMP Stable 接口获取标准日线数据
    """
    # 修正点：使用 /stable/ 路径，而非 /api/v3/
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-full/{ticker}"
        f"?apikey={FMP_API_KEY}"
    )
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # historical-price-full 返回的数据在 'historical' 键中
        if not data or 'historical' not in data: 
            return None

        # FMP 返回的数据通常是按日期倒序的，需要正序排列以计算指标
        df = pd.DataFrame(data['historical'])
        df = df.set_index('date').sort_index(ascending=True)
        df.index = pd.to_datetime(df.index)
        
        # 只取需要的长度进行计算 (EMA90 需要较长数据)
        df = df.tail(days)
        
        # 确保数据不为空
        if df.empty:
            return None

        return calculate_nx_indicators(df)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# --- Discord Bot Logic ---

class StockBotClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.alert_channel = None

    async def on_ready(self):
        load_settings()
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        
        self.alert_channel = self.get_channel(ALERT_CHANNEL_ID)
        if not self.alert_channel:
             print(f"⚠️ 警告: 无法找到 ID 为 {ALERT_CHANNEL_ID} 的频道。请检查环境变量。")
        
        # 启动定时监控
        self.monitor_stocks.start()
        
        # 同步斜杠命令
        await self.tree.sync()
        print("Slash commands synced.")

    @tasks.loop(minutes=5)
    async def monitor_stocks(self):
        if not self.alert_channel: return

        # 获取纽约时间
        now_et = datetime.now(MARKET_TIMEZONE)
        curr_time = now_et.time()
        today_str = now_et.strftime('%Y-%m-%d')

        # 判断时间段
        is_pre = TIME_PRE_MARKET_START <= curr_time < TIME_MARKET_OPEN
        is_open = TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE

        # 非监控时间段直接返回
        if not (is_pre or is_open): return

        print(f"[{now_et.strftime('%H:%M')}] Scanning markets for signals...")
        
        stocks = settings.get("MONITORED_STOCKS", [])
        daily_status = settings.get("DAILY_STATUS", {})
        
        # 清理旧日期的状态
        for k in list(daily_status.keys()):
            if not k.endswith(today_str): del daily_status[k]

        for ticker in stocks:
            status_key = f"{ticker}-{today_str}"
            status = daily_status.get(status_key, "NONE")

            # 频率控制: 盘前1次，盘中1次
            if is_pre and status != "NONE": continue
            if is_open and status in ["MARKET_SENT", "BOTH_SENT"]: continue

            # 获取数据
            df = get_stock_data(ticker)
            if df is None or df.empty: 
                time_module.sleep(1) # 避免请求过快
                continue

            # 核心策略判断
            triggered, reason, level = check_signals(df)

            if triggered:
                # 生成图表
                chart_file = generate_chart(df, ticker)
                price = df['close'].iloc[-1]
                
                # 构造消息
                header = "【🚨 🚨🚨神级K线分析系统】"
                if level == "S_TIER": header += " 🔥 趋势突破!"
                
                # 获取 Nx 梯子下沿价格作为参考
                nx_support = df['Nx_Blue_DW'].iloc[-1]

                msg = (
                    f"{header}\n"
                    f"🎯 **标的**: `{ticker}`\n"
                    f"💰 **现价**: `${price:.2f}`\n"
                    f"------------------------\n"
                    f"{reason}\n"
                    f"------------------------\n"
                    f"📚 **操作指引**:\n"
                    f"1. **Nx 突破**: 属于加仓/买入信号 (站稳蓝色梯子)。\n"
                    f"2. **Cd 抄底**: 仅建议在支撑位附近或极度超卖时尝试。\n"
                    f"3. **风控参考**: 蓝色梯子下沿支撑位 `${nx_support:.2f}`"
                )

                try:
                    with discord.File(chart_file) as file:
                        await self.alert_channel.send(content=msg, file=file)
                    
                    # 更新状态
                    if is_pre:
                        new_status = "PRE_SENT"
                    else:
                        # 如果盘前发过，现在是盘中，标记为 BOTH；否则标记为 MARKET
                        new_status = "BOTH_SENT" if status == "PRE_SENT" else "MARKET_SENT"
                    
                    settings["DAILY_STATUS"][status_key] = new_status
                    save_settings()
                    print(f"Alert sent for {ticker}")
                except Exception as e:
                    print(f"Error sending alert for {ticker}: {e}")
                finally:
                    # 删除临时图片
                    if os.path.exists(chart_file): os.remove(chart_file)
            
            # 避免触发 API 速率限制
            time_module.sleep(1.5)

    @self.tree.command(name="addstocks", description="添加监控股票 (用空格分隔)")
    @app_commands.describe(tickers="例如: AAPL NVDA TSLA")
    async def add_stocks(self, interaction: discord.Interaction, tickers: str):
        await interaction.response.defer()
        
        # 处理输入：去空格、大写、去重
        s_list = list(set([t.strip().upper() for t in tickers.replace(',', ' ').split() if t.strip()]))
        
        settings["MONITORED_STOCKS"] = s_list
        # 重置今日状态，以便新添加的股票能立即被扫描
        settings["DAILY_STATUS"] = {} 
        save_settings()
        
        await interaction.followup.send(f"✅ 已更新监控列表，当前监控 {len(s_list)} 只股票。")

# --- 启动 ---
if __name__ == "__main__":
    if DISCORD_TOKEN:
        # 实例化并运行
        intents = discord.Intents.default()
        client = StockBotClient(intents=intents)
        client.run(DISCORD_TOKEN)
    else:
        print("❌ 错误: 未找到 DISCORD_TOKEN。请检查环境变量设置。")
