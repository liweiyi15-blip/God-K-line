import requests
import os
import json

# 默认使用您提供的 API Key，生产环境建议配置在环境变量 FMP_API_KEY 中
FMP_API_KEY = os.environ.get("FMP_API_KEY", "9puU4iHVeTJvu7txYpokXLagbu2vucLK")

def get_historical_stock_data(symbol, start_date, end_date):
    """
    获取股票历史数据
    使用 stable 接口: /stable/historical-price-eod/full
    """
    # 修改点：使用 stable 接口，symbol 变为查询参数
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    
    params = {
        "symbol": symbol,
        "from": start_date,
        "to": end_date,
        "apikey": FMP_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # 检查 HTTP 错误
        
        data = response.json()
        
        # 简单校验返回数据是否包含错误信息
        if isinstance(data, dict) and "Error Message" in data:
            raise Exception(f"FMP API Error: {data['Error Message']}")
            
        return data

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # 测试代码
    print("Fetching TSLA data...")
    result = get_historical_stock_data("TSLA", "2024-11-03", "2025-12-08")
    
    if result:
        print(f"Successfully fetched {len(result)} records.")
        # 打印前两条数据用于验证
        print(json.dumps(result[:2], indent=2))
    else:
        print("Failed to fetch data.")
