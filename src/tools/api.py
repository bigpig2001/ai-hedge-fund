import datetime
import os
import pandas as pd
import requests
import akshare as ak

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date) and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data if (start_date is None or news["date"] >= start_date) and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)


def get_prices_akshare(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from akshare for Chinese stocks."""
    try:
        # Convert dates to 'YYYYMMDD' format for akshare
        ak_start_date = start_date.replace("-", "")
        ak_end_date = end_date.replace("-", "")

        # Fetch historical price data
        # akshare might require specific ticker formats, e.g., 'sh600000' or 'sz000001'
        # For now, assuming the ticker is passed in the correct format.
        stock_data_df = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=ak_start_date, end_date=ak_end_date, adjust="")

        if stock_data_df is None or stock_data_df.empty:
            return []

        prices = []
        for _, row in stock_data_df.iterrows():
            price_obj = Price(
                open=row["开盘"],
                close=row["收盘"],
                high=row["最高"],
                low=row["最低"],
                volume=int(row["成交量"]), # Ensure volume is an integer
                time=pd.to_datetime(row["日期"]).strftime('%Y-%m-%d') # Standardize date format
            )
            prices.append(price_obj)
        
        return prices
    except Exception as e:
        print(f"Error fetching price data from akshare for {ticker}: {e}")
        return []


def get_financial_metrics_akshare(ticker: str, end_date: str) -> list[FinancialMetrics]:
    """Fetch financial metrics from akshare for Chinese stocks."""
    try:
        # Fetch financial analysis indicators
        # The 'end_date' equivalent in akshare for financial indicators is often implicit (latest)
        # or might be part of how you select data if multiple periods are returned.
        # ak.stock_financial_analysis_indicator usually returns a DataFrame with many metrics.
        financial_data_df = ak.stock_financial_analysis_indicator(symbol=ticker)

        if financial_data_df is None or financial_data_df.empty:
            return []

        # akshare often returns metrics with the most recent period last.
        # We need to select the most recent available data.
        # The DataFrame is indexed by date or report period, so we can pick the last row.
        latest_metrics_series = financial_data_df.iloc[-1]

        # Helper to safely get float values, returning None if key is missing or value is not convertible
        def get_float_metric(series: pd.Series, key: str) -> float | None:
            try:
                return float(series.get(key))
            except (TypeError, ValueError):
                return None

        # Mapping fields from akshare (Chinese) to FinancialMetrics model (English)
        # This requires knowing the exact column names from ak.stock_financial_analysis_indicator()
        # and matching them to the FinancialMetrics model.
        # Example mapping (actual keys from akshare might differ and need verification):
        # Note: 'report_period' and 'period' might need specific handling based on akshare's output.
        # 'currency' is typically CNY for Chinese stocks.
        
        # For 'report_period', akshare might provide a '报告日历' or similar.
        # For 'period', it could be '季度', '年度', 'TTM'. We'll assume TTM or latest annual if possible.
        # This example assumes we get the latest report period.
        # The date from the index of the series will be used as the report_period
        report_period_str = str(latest_metrics_series.name) # Assuming the index is the date/period

        metrics_obj = FinancialMetrics(
            ticker=ticker,
            # akshare's stock_financial_analysis_indicator index is usually the date of the report.
            # We need to format it correctly.
            report_period=pd.to_datetime(report_period_str).strftime('%Y-%m-%d'),
            period="annual", # This is an assumption, akshare might specify or it might be inferred
            currency="CNY", # Assuming Chinese Yuan
            
            market_cap=get_float_metric(latest_metrics_series, "总市值"), # Example key
            enterprise_value=get_float_metric(latest_metrics_series, "企业价值"), # Placeholder, need actual key
            price_to_earnings_ratio=get_float_metric(latest_metrics_series, "市盈率TTM"), # Example key
            price_to_book_ratio=get_float_metric(latest_metrics_series, "市净率"), # Example key
            price_to_sales_ratio=get_float_metric(latest_metrics_series, "市销率TTM"), # Example key
            enterprise_value_to_ebitda_ratio=get_float_metric(latest_metrics_series, "EV/EBITDA"), # Placeholder
            enterprise_value_to_revenue_ratio=get_float_metric(latest_metrics_series, "EV/收入"), # Placeholder
            free_cash_flow_yield=None, # May not be directly available, might need calculation
            peg_ratio=get_float_metric(latest_metrics_series, "PEG"), # Example key
            gross_margin=get_float_metric(latest_metrics_series, "销售毛利率"), # Example key
            operating_margin=get_float_metric(latest_metrics_series, "营业利润率"), # Example key
            net_margin=get_float_metric(latest_metrics_series, "销售净利率"), # Example key
            return_on_equity=get_float_metric(latest_metrics_series, "净资产收益率ROE"), # Example key
            return_on_assets=get_float_metric(latest_metrics_series, "总资产报酬率ROA"), # Example key
            return_on_invested_capital=get_float_metric(latest_metrics_series, "投入资本回报率ROIC"), # Example key
            asset_turnover=get_float_metric(latest_metrics_series, "总资产周转率"), # Example key
            inventory_turnover=get_float_metric(latest_metrics_series, "存货周转率"), # Example key
            receivables_turnover=get_float_metric(latest_metrics_series, "应收账款周转率"), # Example key
            days_sales_outstanding=None, # May not be directly available
            operating_cycle=None, # May not be directly available
            working_capital_turnover=None, # May not be directly available
            current_ratio=get_float_metric(latest_metrics_series, "流动比率"), # Example key
            quick_ratio=get_float_metric(latest_metrics_series, "速动比率"), # Example key
            cash_ratio=None, # May not be directly available
            operating_cash_flow_ratio=None, # May not be directly available
            debt_to_equity=get_float_metric(latest_metrics_series, "产权比率"), # Example key (Debt/Equity)
            debt_to_assets=get_float_metric(latest_metrics_series, "资产负债率"), # Example key
            interest_coverage=get_float_metric(latest_metrics_series, "利息保障倍数"), # Example key
            revenue_growth=get_float_metric(latest_metrics_series, "营业收入同比增长率"), # Example key
            earnings_growth=get_float_metric(latest_metrics_series, "净利润同比增长率"), # Example key
            book_value_growth=None, # May not be directly available
            earnings_per_share_growth=get_float_metric(latest_metrics_series, "基本每股收益同比增长率"), # Example key
            free_cash_flow_growth=None, # May not be directly available
            operating_income_growth=get_float_metric(latest_metrics_series, "营业利润同比增长率"), # Example key
            ebitda_growth=None, # May not be directly available
            payout_ratio=get_float_metric(latest_metrics_series, "股息支付率"), # Example key
            earnings_per_share=get_float_metric(latest_metrics_series, "基本每股收益"), # Example key
            book_value_per_share=get_float_metric(latest_metrics_series, "每股净资产"), # Example key
            free_cash_flow_per_share=None # May not be directly available
        )
        return [metrics_obj]
    except Exception as e:
        print(f"Error fetching financial metrics from akshare for {ticker}: {e}")
        return []


def get_company_news_akshare(ticker: str, limit: int = 20) -> list[CompanyNews]:
    """Fetch company news from akshare for Chinese stocks."""
    try:
        # Fetch company news using stock_news_em
        # The 'limit' parameter might not be directly supported by stock_news_em in the same way.
        # It usually returns recent news, often a fixed number of articles or paginated.
        # We'll fetch the default and then slice if necessary.
        news_df = ak.stock_news_em(stock=ticker) # Eastmoney news for a specific stock

        if news_df is None or news_df.empty:
            return []

        news_items = []
        for _, row in news_df.head(limit).iterrows(): # Apply limit after fetching
            # Mapping fields from akshare to CompanyNews model
            # 'stock_news_em' columns are typically: "code", "title", "content", "public_time", "source"
            # Need to verify actual column names from akshare output.
            
            # The 'date' needs to be parsed and formatted correctly.
            # 'public_time' from ak.stock_news_em is usually a string like "YYYY-MM-DD HH:MM:SS"
            news_date_str = pd.to_datetime(row["public_time"]).strftime('%Y-%m-%d')

            news_obj = CompanyNews(
                ticker=ticker, # or row["code"] if it's more reliable
                title=row["title"],
                author=row.get("source", "N/A"), # 'source' can be used as author if no specific author field
                source=row.get("source", "Eastmoney"), # Default to Eastmoney if not specified
                date=news_date_str,
                url=row.get("url", row.get("article_url", "#")), # Check for URL fields, provide placeholder if none
                sentiment=None # akshare news functions usually don't provide sentiment
            )
            news_items.append(news_obj)
        
        return news_items
    except Exception as e:
        print(f"Error fetching company news from akshare for {ticker}: {e}")
        return []

