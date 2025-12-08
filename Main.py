import discord
from discord import app_commands
from discord.ext import commands
import aiohttp
import os
import json
import asyncio
from datetime import datetime, timedelta

# 从环境变量获取 Token 和 API Key
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
FMP_API_KEY = os.getenv('FMP_API_KEY')

# 数据文件路径
DATA_FILE = 'watchlist.json'

class StockBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)
        self.watchlist = self.load_watchlist()

    def load_watchlist(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_watchlist(self):
        with open(DATA_FILE, 'w') as f:
            json.dump(self.watchlist, f)

    async def setup_hook(self):
        await self.tree.sync()
        print(f"Logged in as {self.user}")

client = StockBot()

async def fetch_stock_history(symbol: str):
    """
    使用 FMP Stable 接口获取历史数据
    URL: https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL&...
    """
    # 动态设定日期范围（例如过去30天）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # --- 修正重点：使用 Stable 接口，symbol 作为参数 ---
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    
    params = {
        "symbol": symbol.upper(),
        "from": start_date,
        "to": end_date,
        "apikey": FMP_API_KEY
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if not data:
                    return None
                # stable 接口直接返回列表，取最新的一条
                # 注意：具体返回结构可能需要根据实际数据微调，通常是 list[dict]
                latest = data[0] if isinstance(data, list) and len(data) > 0 else None
                return latest
            else:
                print(f"API Error {response.status}: {await response.text()}")
                return None

@client.tree.command(name="add_list", description="批量添加股票到关注列表 (用逗号或空格分隔)")
async def add_list(interaction: discord.Interaction, symbols: str):
    user_id = str(interaction.user.id)
    
    # 分割并清理输入的代码
    new_symbols = [s.upper().strip() for s in symbols.replace(',', ' ').split() if s.strip()]
    
    if user_id not in client.watchlist:
        client.watchlist[user_id] = []
    
    added = []
    for s in new_symbols:
        if s not in client.watchlist[user_id]:
            client.watchlist[user_id].append(s)
            added.append(s)
    
    client.save_watchlist()
    
    if added:
        await interaction.response.send_message(f"已添加: {', '.join(added)}")
    else:
        await interaction.response.send_message("所有输入的代码已在列表中。")

@client.tree.command(name="remove_list", description="从关注列表中批量移除股票")
async def remove_list(interaction: discord.Interaction, symbols: str):
    user_id = str(interaction.user.id)
    
    if user_id not in client.watchlist or not client.watchlist[user_id]:
        await interaction.response.send_message("你的关注列表是空的。")
        return

    targets = [s.upper().strip() for s in symbols.replace(',', ' ').split() if s.strip()]
    removed = []
    
    for s in targets:
        if s in client.watchlist[user_id]:
            client.watchlist[user_id].remove(s)
            removed.append(s)
            
    client.save_watchlist()
    
    if removed:
        await interaction.response.send_message(f"已移除: {', '.join(removed)}")
    else:
        await interaction.response.send_message("列表中未找到指定的代码。")

@client.tree.command(name="list", description="查看我的关注列表及最新价格")
async def list_stocks(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    
    if user_id not in client.watchlist or not client.watchlist[user_id]:
        await interaction.response.send_message("你的关注列表是空的。请使用 /add_list 添加。")
        return

    await interaction.response.defer()
    
    message_lines = ["**我的关注列表 (Latest EOD):**"]
    
    for symbol in client.watchlist[user_id]:
        data = await fetch_stock_history(symbol)
        if data:
            # 根据 stable 接口返回字段适配，通常包含 date, close, volume 等
            price = data.get('close') or data.get('adjClose')
            date = data.get('date')
            message_lines.append(f"• **{symbol}**: ${price} ({date})")
        else:
            message_lines.append(f"• **{symbol}**: 获取失败")
            
    await interaction.followup.send("\n".join(message_lines))

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
