def check_signals(df):
    """
    严格遵循机构级漏斗筛选逻辑 (Priority 1 -> 7)
    """
    if df is None or len(df) < 60: return False, "", "NONE", [], []
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    triggers = []
    # 默认级别
    current_level_score = 0 # 用于比较信号强度
    final_level = "NONE" 
    
    # === Priority 1: 风控第一 (60日暴涨过滤) ===
    # 逻辑：如果过去60天最低价到现在涨幅超过 60%~80%，这波鱼尾我就不吃了，给别人吃。
    low_60 = df['low'].tail(60).min()
    if low_60 > 0 and curr['close'] > low_60 * 1.8: # 这里的系数1.8可微调，越小越安全
        # 直接返回，不看后面任何信号，保命要紧
        return False, "❌ 风控拦截: 短期涨幅过大", "RISK_CONTROL", [], []

    # === Priority 2: 二次起爆 (GOD_TIER) - 盈亏比之王 ===
    # 逻辑：突破过蓝梯 -> 回调蓝梯附近获得支撑 -> 再次放量
    recent_15 = df.tail(15)
    had_breakout = (recent_15['close'] > recent_15['Nx_Blue_UP']).any() # 近期有过突破
    # 股价在蓝梯附近 (下沿之上，上沿上方一点点以内)
    in_support_zone = curr['close'] > curr['Nx_Blue_DW'] and curr['low'] <= curr['Nx_Blue_UP'] * 1.05
    # 放量确认
    re_volume = curr['volume'] > curr['Vol_MA20'] * 1.3
    
    if had_breakout and in_support_zone and re_volume:
        triggers.append(f"👑 **二次起爆 (God Tier)**: 突破回踩蓝梯确认，黄金买点！")
        if 5 > current_level_score: 
            final_level = "GOD_TIER"
            current_level_score = 5

    # === Priority 3: 旗形/楔形突破 (S_TIER) - 爆发力最强 ===
    pattern_name, res_line, sup_line = identify_patterns(df)
    if pattern_name:
        triggers.append(pattern_name)
        if 4 > current_level_score:
            final_level = "S_TIER"
            current_level_score = 4

    # === Priority 4: Nx蓝梯突破 (A_TIER) - 趋势确立 ===
    # 逻辑：昨天在梯子下/里，今天站稳梯子上沿
    nx_breakout = prev['close'] < prev['Nx_Blue_UP'] and curr['close'] > curr['Nx_Blue_UP']
    if nx_breakout:
        triggers.append(f"🚀 **Nx趋势突破 (A Tier)**: 站上蓝色牛熊线，趋势转多")
        if 3 > current_level_score:
            final_level = "A_TIER"
            current_level_score = 3

    # === Priority 5: Cd/MACD底背离 (B_TIER) - 底部反转 ===
    # 逻辑：股价新低 + DIF没新低 + 金叉/拐头
    low_20 = df['low'].tail(20).min()
    price_is_low = curr['low'] <= low_20 * 1.02 # 接近新低
    dif_20_min = df['DIF'].tail(20).min()
    divergence = curr['DIF'] > dif_20_min 
    momentum_turn = curr['MACD'] > prev['MACD']
    
    is_downtrend = curr['close'] < curr['Nx_Blue_DW'] 
    
    if price_is_low and divergence and momentum_turn:
        # 如果是下跌趋势中，这个信号比较重要
        triggers.append(f"💎 **底背离 (B Tier)**: 股价新低指标背离，潜在反转")
        if 2 > current_level_score:
            final_level = "B_TIER"
            current_level_score = 2

    # === Priority 6: RSI弘历战法 (C_TIER) - 辅助反弹 ===
    # 逻辑：超卖区金叉
    rsi_buy = prev['RSI'] < 30 and curr['RSI'] > 30
    if rsi_buy:
        triggers.append(f"⚠️ **RSI反弹 (C Tier)**: 超卖反弹，仅限短线")
        if 1 > current_level_score:
            final_level = "C_TIER"
            current_level_score = 1

    # === Priority 7: 逆势过滤 (最后一道防线) ===
    # 如果处于下跌趋势 (收盘 < 蓝梯下沿)，且信号强度不够强 (只有C级或没信号)
    # 必须过滤掉，防止在下跌中途接飞刀
    if triggers:
        # 只有 GOD, S, A, B 级信号允许在某种程度逆势(比如底背离本身就是逆势)
        # 但如果是单纯的 C级 RSI反弹 且 趋势极差，建议过滤
        
        # 严格规则：如果在下跌趋势中，且只有 C_TIER 信号，过滤掉
        if is_downtrend and final_level == "C_TIER":
             return False, "", "NONE", [], []
             
        # 返回结果
        return True, "\n".join(triggers), final_level, res_line, sup_line

    return False, "", "NONE", [], []
