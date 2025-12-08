import discord
from discord import app_commands
from discord.ext import tasks
import requests
import json
import os
from datetime import datetime, time, timedelta
import time as time_module
import pandas as pd
import mplfinance as mpf
import pytz
from dotenv import load_dotenv
from collections import defaultdict

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
    return df

def check_signals(df):
    if len(df) < 30: return False, "", "NONE"
    curr, prev = df.iloc[-1], df.iloc[-2]
    triggers, level = [], "NORMAL"
    
    if prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']:
        triggers.append(f"🔥 **Nx 突破**: 突破蓝色牛熊分界线 (${curr['Nx_Blue_UP']:.2f})")
        level = "S_TIER"
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    
    low_20, dif_20 = df['low'].tail(20).min(), df['DIF'].tail(20).min()
    if (curr['low'] <= low_20 * 1.01) and (curr['DIF'] > dif_20) and (prev['DIF'] < prev['DEA'] and curr['DIF'] > curr['DEA']):
        if not (is_downtrend and curr['RSI'] > 30):
             triggers.append(f"💎 **Cd 抄底**: MACD 底背离且金叉 (RSI: {curr['RSI']:.1f})")
             
    if prev['RSI'] < 30 and curr['RSI'] > 30:
        if is_downtrend and "Cd 抄底" not in str(triggers):
            triggers.append(f"⚠️ **RSI 反弹**: 超卖反弹 (趋势仍偏空)")
        else:
            triggers.append(f"🚀 **弘历战法**: RSI(14) 从超卖区金叉向上")
            
    if triggers:
        if level == "S_TIER": return True, "\n".join(triggers), "S_TIER"
        if is_downtrend and len(triggers) < 2: return False, "", "NONE"
        return True, "\n".join(triggers), "NORMAL"
    return False, "", "NONE"

def generate_chart(df, ticker):
    filename = f"{ticker}_alert.png"
    s = mpf.make_marketcolors(up='r', down='g', inherit=True)
    
    # 🔴 关键修复：将 "seaborn" 改为 "ggplot" (或者 "seaborn-v0_8")
    # "ggplot" 是一个稳定且兼容性极好的样式，避免了新版 Matplotlib 找不到 seaborn 样式的问题
    my_style = mpf.make_mpf_style(base_mpl_style="ggplot", marketcolors=s, gridstyle=":")
    
    plot_df = df.tail(60)
    add_plots = [
        mpf.make_addplot(plot_df['Nx_Blue_UP'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Blue_DW'], color='dodgerblue', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_UP'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['Nx_Yellow_DW'], color='gold', width=1.0),
        mpf.make_addplot(plot_df['MACD'], panel=2, type='bar', color='dimgray', alpha=0.5, ylabel='MACD'),
        mpf.make_addplot(plot_df['DIF'], panel=2, color='orange'),
        mpf.make_addplot(plot_df['DEA'], panel=2, color='blue'),
    ]
    title = f"{ticker} God-Tier Analysis"
    mpf.plot(plot_df, type='candle', style=my_style, title=title, ylabel='Price ($)', addplot=add_plots, volume=True, panel_ratios=(6, 2, 2), savefig=filename)
    return filename

# --- 数据获取 ---

def get_stock_data(ticker, days=200):
    now = datetime.now()
    end_date_str = now.strftime("%Y-%m-%d")
    start_date_str = (now - timedelta(days=400)).strftime("%Y-%m-%d")
    
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={ticker}&from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
    )
    
    print(f"🔍 [Debug] Requesting {ticker} ({start_date_str} to {end_date_str})...")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ [API Error] HTTP {response.status_code}: {response.text}")
            return None
            
        data = response.json()
        
        if not data:
            print(f"❌ [API Error] Empty response for {ticker}")
            return None
            
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'historical' in data:
            df = pd.DataFrame(data['historical'])
        else:
            print(f"❌ [API Error] Unexpected data format: {type(data)}")
            return None

        df = df.set_index('date').sort_index(ascending=True)
        df.index = pd.to_datetime(df.index)
        
        print(f"✅ [Success] Loaded {len(df)} rows for {ticker}")
        
        if len(df) < 90:
            print(f"⚠️ [Warning] Not enough data for {ticker} (only {len(df)} rows). Indicators may be inaccurate.")
            
        return calculate_nx_indicators(df)
        
    except Exception as e:
        print(f"❌ [Exception] Error fetching {ticker}: {e}")
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
        print("Slash commands synced!")

    @tasks.loop(minutes=5)
    async def monitor_stocks(self):
        if not self.alert_channel: return
        now_et = datetime.now(MARKET_TIMEZONE)
        curr_time, today_str = now_et.time(), now_et.strftime('%Y-%m-%d')
        
        is_pre = TIME_PRE_MARKET_START <= curr_time < TIME_MARKET_OPEN
        is_open = TIME_MARKET_OPEN <= curr_time <= TIME_MARKET_CLOSE
        
        if not (is_pre or is_open): return
        
        print(f"[{now_et.strftime('%H:%M')}] Scanning markets...")
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
            if df is None or df.empty:
                time_module.sleep(1)
                continue

            triggered, reason, level = check_signals(df)
            if triggered:
                chart_file = generate_chart(df, ticker)
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
                    header = "【🚨 神级K线系统】" + (" 🔥 趋势突破!" if level == "S_TIER" else "")
                    msg = (
                        f"{mentions}\n{header}\n"
                        f"🎯 **标的**: `{ticker}` | 💰 **现价**: `${price:.2f}`\n"
                        f"{'-'*25}\n{reason}\n{'-'*25}\n"
                        f"🌊 **Nx 蓝梯下沿**: `${nx_support:.2f}`"
                    )
                    try:
                        with discord.File(chart_file) as file:
                            await self.alert_channel.send(content=msg, file=file)
                    finally:
                        if os.path.exists(chart_file): os.remove(chart_file)
            time_module.sleep(1.5)

# --- 实例化 & 注册命令 ---

intents = discord.Intents.default()
client = StockBotClient(intents=intents)

@client.tree.command(name="addstocks", description="[个人] 添加关注股票")
async def add_stocks(interaction: discord.Interaction, tickers: str):
    await interaction.response.defer()
    user_data = get_user_data(interaction.user.id)
    new_list = list(set([t.strip().upper() for t in tickers.replace(',', ' ').split() if t.strip()]))
    current_set = set(user_data["stocks"])
    current_set.update(new_list)
    user_data["stocks"] = list(current_set)
    save_settings()
    await interaction.followup.send(f"✅ 已添加！新增: `{', '.join(new_list)}`")

@client.tree.command(name="liststocks", description="[个人] 查看我的股票")
async def list_stocks(interaction: discord.Interaction):
    stocks = get_user_data(interaction.user.id)["stocks"]
    await interaction.response.send_message(f"📋 **关注列表**:\n`{', '.join(stocks) if stocks else '空'}`", ephemeral=True)

@client.tree.command(name="clearstocks", description="[个人] 清空我的股票")
async def clear_stocks(interaction: discord.Interaction):
    user_data = get_user_data(interaction.user.id)
    user_data["stocks"] = []
    user_data["daily_status"] = {}
    save_settings()
    await interaction.response.send_message("🗑️ 已清空。", ephemeral=True)

@client.tree.command(name="test", description="[测试] 立即测试股票")
async def test_command(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    df = get_stock_data(ticker)
    
    if df is None:
        await interaction.followup.send("❌ 获取失败，请检查 Railway 日志。")
        return
        
    chart_file = generate_chart(df, ticker)
    last_row = df.iloc[-1]
    
    msg = (
        f"✅ **接口测试正常** | `{ticker}`\n"
        f"数据来源: `historical-price-eod` (近400天)\n"
        f"• 日期: `{df.index[-1].strftime('%Y-%m-%d')}`\n"
        f"• 收盘: `{last_row['close']:.2f}`\n"
        f"• Nx蓝梯上沿: `{last_row['Nx_Blue_UP']:.2f}`\n"
        f"• RSI(14): `{last_row['RSI']:.2f}`"
    )
    
    try:
        with discord.File(chart_file) as file:
            await interaction.followup.send(content=msg, file=file)
    finally:
        if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    if DISCORD_TOKEN:
        client.run(DISCORD_TOKEN)
