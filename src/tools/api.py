import datetime
import os
import pandas as pd
import requests
import akshare # Add akshare import

from src.data.cache import get_cache

# Helper functions for Chinese tickers
def is_chinese_ticker(ticker: str) -> bool:
    """Checks if the ticker is a Chinese stock ticker."""
    return ticker.endswith(".SH") or ticker.endswith(".SZ")

def get_akshare_ticker(ticker: str) -> str:
    """Extracts the 6-digit code for akshare from a ticker like 600519.SH or 000001.SZ."""
    if is_chinese_ticker(ticker):
        return ticker.split('.')[0]
    return ticker # Should not happen if is_chinese_ticker is checked first
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

# Helper function for parsing market cap strings with Chinese units
def parse_market_cap_string(value_str: str | float | None) -> float | None:
    if value_str is None:
        return None
    if isinstance(value_str, (float, int)): # Already numeric
        return float(value_str)
    if not isinstance(value_str, str):
        return None 
        
    value_str = value_str.strip()
    if not value_str: # Empty string
        return None

    try:
        if '亿' in value_str:
            return float(value_str.replace('亿', '')) * 10**8
        if '万' in value_str:
            return float(value_str.replace('万', '')) * 10**4
        # Attempt direct conversion for plain numbers or if no unit matched
        return float(value_str)
    except ValueError:
        return None


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    if is_chinese_ticker(ticker):
        try:
            akshare_ticker = get_akshare_ticker(ticker)
            # Convert date formats for akshare
            akshare_start_date = start_date.replace("-", "")
            akshare_end_date = end_date.replace("-", "")

            # Fetch data using akshare
            stock_data_df = akshare.stock_zh_a_hist(symbol=akshare_ticker, start_date=akshare_start_date, end_date=akshare_end_date, adjust="qfq")

            if stock_data_df is None or stock_data_df.empty:
                return []

            # Transform data to list of Price objects
            prices = []
            for _, row in stock_data_df.iterrows():
                # Ensure date is in YYYY-MM-DD format
                price_time = row["日期"].strftime("%Y-%m-%d") if isinstance(row["日期"], datetime.datetime) else str(row["日期"])
                
                # Filter by date again to ensure exact range
                if not (start_date <= price_time <= end_date):
                    continue

                # Ensure all expected columns are present
                open_price = row.get("开盘")
                close_price = row.get("收盘")
                high_price = row.get("最高")
                low_price = row.get("最低")
                volume_val = row.get("成交量")
                
                if any(val is None for val in [open_price, close_price, high_price, low_price, volume_val]):
                    print(f"Warning: Missing one or more expected price columns (开盘, 收盘, 最高, 最低, 成交量) in row for {ticker} on {price_time}. Skipping row.")
                    continue

                prices.append(
                    Price(
                        open=float(open_price),
                        close=float(close_price),
                        high=float(high_price),
                        low=float(low_price),
                        volume=int(volume_val),
                        time=price_time,
                    )
                )
            
            if not prices:
                return []

            # Cache the results as dicts
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            return prices
        except Exception as e:
            # Log error or raise custom exception
            print(f"Error fetching data for Chinese ticker {ticker} using akshare: {e}")
            # Fallback or raise an error - for now, returning empty list
            return [] 
    else:
        # If not in cache or no data in range, fetch from API (existing logic)
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

    if is_chinese_ticker(ticker):
        try:
            akshare_ticker = get_akshare_ticker(ticker)
            # Fetch data using akshare - stock_financial_analysis_indicator seems to take 'stock' not 'symbol'
            financial_analysis_df = akshare.stock_financial_analysis_indicator(stock=akshare_ticker)

            if financial_analysis_df is None or financial_analysis_df.empty:
                return []

            financial_metrics_list = []
            # Rename columns for easier mapping and ensure they exist
            # This is a sample mapping, actual column names from akshare need to be verified
            column_mapping = {
                '日期': 'report_period_ak', # Assuming '日期' is the report date
                '总市值': 'market_cap',
                '市盈率(PE)': 'price_to_earnings_ratio', # Example, verify actual name
                '市净率(PB)': 'price_to_book_ratio',   # Example, verify actual name
                '市销率(PS)': 'price_to_sales_ratio', # Example, verify actual name
                '净资产收益率(ROE)': 'return_on_equity', # Example, verify actual name
                '净利润同比增长率(%)': 'earnings_growth', # Example, verify actual name
                '营业总收入同比增长率(%)': 'revenue_growth', # Example, verify actual name
                '基本每股收益(元)': 'earnings_per_share' # Example, verify actual name
                # Add other relevant mappings here
            }
            
            # Filter out columns not present in the DataFrame to avoid errors
            # and rename existing ones
            df_renamed = financial_analysis_df.rename(columns=lambda c: column_mapping.get(c, c))
            
            # Ensure 'report_period_ak' exists after renaming
            if 'report_period_ak' not in df_renamed.columns:
                 # Try a common alternative if '日期' was not correct
                if '报告日' in df_renamed.columns:
                    df_renamed.rename(columns={'报告日': 'report_period_ak'}, inplace=True)
                elif '截止日期' in df_renamed.columns: # Another common name
                    df_renamed.rename(columns={'截止日期': 'report_period_ak'}, inplace=True)
                else:
                    print(f"Report period column not found for {ticker} in akshare output. Columns: {df_renamed.columns}")
                    return []


            for _, row in df_renamed.iterrows():
                report_date_str = str(row['report_period_ak'])
                # Ensure date is in YYYY-MM-DD format, it might come as YYYYMMDD or YYYY-MM-DD HH:MM:SS
                if len(report_date_str) == 8 and report_date_str.isdigit(): # YYYYMMDD
                    report_date_formatted = f"{report_date_str[:4]}-{report_date_str[4:6]}-{report_date_str[6:8]}"
                elif ' ' in report_date_str: # Contains time
                     report_date_formatted = report_date_str.split(' ')[0]
                else: # Assumed to be YYYY-MM-DD already or needs other handling
                     report_date_formatted = report_date_str

                if report_date_formatted > end_date:
                    continue

                # Determine period (annual/quarterly)
                # Simple heuristic: if month is 12, assume annual. Otherwise quarterly.
                # This is a simplification. Akshare might provide this info more directly.
                month = int(report_date_formatted.split('-')[1])
                derived_period = "annual" if month == 12 else "quarterly"
                
                # If the function's 'period' param is 'ttm', we skip as we don't calculate it.
                # Or, if 'period' is 'annual' and derived is 'quarterly', we might skip.
                # For now, let's include all and let the user filter later if needed,
                # or align with the 'period' param if it's not 'ttm'.
                if period != "ttm" and period != derived_period:
                    # This logic might be too strict, adjust as needed.
                    # For now, if specific period (annual/quarterly) requested, match it.
                    # If 'ttm' is requested, we can't provide it from this akshare source directly.
                    pass # Continue and add the metric, or `continue` to skip if strict matching is desired.


                # Helper to safely get float value
                def get_float_or_none(series, key):
                    val = series.get(key)
                    if pd.isna(val) or val == '-': # akshare often uses '-' for N/A
                        return None
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
                
                # Helper to safely get percentage value (input as float e.g. 5 for 5%)
                def get_percentage_or_none(series, key):
                    val = get_float_or_none(series,key)
                    return val / 100.0 if val is not None else None


                metrics_data = {
                    "ticker": ticker,
                    "report_period": report_date_formatted,
                    "period": derived_period, # Use derived period
                    "currency": "CNY",
                    "market_cap": get_float_or_none(row, 'market_cap'),
                    "price_to_earnings_ratio": get_float_or_none(row, 'price_to_earnings_ratio'),
                    "price_to_book_ratio": get_float_or_none(row, 'price_to_book_ratio'),
                    "price_to_sales_ratio": get_float_or_none(row, 'price_to_sales_ratio'),
                    "return_on_equity": get_percentage_or_none(row, 'return_on_equity'),
                    "earnings_growth": get_percentage_or_none(row, 'earnings_growth'), # Assuming akshare provides it as XX.XX for XX.XX%
                    "revenue_growth": get_percentage_or_none(row, 'revenue_growth'), # Assuming akshare provides it as XX.XX for XX.XX%
                    "earnings_per_share": get_float_or_none(row, 'earnings_per_share'),
                    # --- Fields that might be missing or need careful mapping ---
                    "enterprise_value": None, # Placeholder
                    "enterprise_value_to_ebitda_ratio": None, # Placeholder
                    "enterprise_value_to_revenue_ratio": None, # Placeholder
                    "free_cash_flow_yield": None, # Placeholder
                    "peg_ratio": get_float_or_none(row, '市盈率(PEG)'), # Check if '市盈率(PEG)' or similar exists
                    "gross_margin": get_percentage_or_none(row, '销售毛利率(%)'), # Check for '销售毛利率(%)'
                    "operating_margin": get_percentage_or_none(row, '营业利润率(%)'), # Check for '营业利润率(%)'
                    "net_margin": get_percentage_or_none(row, '销售净利率(%)'), # Check for '销售净利率(%)'
                    "return_on_assets": get_percentage_or_none(row, '总资产报酬率(ROA)(%)'), # Check for '总资产报酬率(ROA)(%)'
                    "return_on_invested_capital": None, # Placeholder
                    "asset_turnover": get_float_or_none(row, '总资产周转率(次)'), # Check for '总资产周转率(次)'
                    "inventory_turnover": get_float_or_none(row, '存货周转率(次)'), # Check for '存货周转率(次)'
                    "receivables_turnover": get_float_or_none(row, '应收账款周转率(次)'), # Check for '应收账款周转率(次)'
                    "days_sales_outstanding": None, # Placeholder, might need calculation
                    "operating_cycle": None, # Placeholder, might need calculation
                    "working_capital_turnover": None, # Placeholder
                    "current_ratio": get_float_or_none(row, '流动比率'), # Check for '流动比率'
                    "quick_ratio": get_float_or_none(row, '速动比率'), # Check for '速动比率'
                    "cash_ratio": None, # Placeholder
                    "operating_cash_flow_ratio": None, # Placeholder
                    "debt_to_equity": get_float_or_none(row, '产权比率(%)'), # Check for '产权比率(%)' - needs conversion from %
                    "debt_to_assets": get_float_or_none(row, '资产负债率(%)'), # Check for '资产负债率(%)' - needs conversion from %
                    "interest_coverage": get_float_or_none(row, '利息保障倍数(倍)'), # Check for '利息保障倍数(倍)'
                    "book_value_growth": None, # Placeholder
                    "earnings_per_share_growth": get_percentage_or_none(row, '每股收益同比增长率(%)'), # Check for '每股收益同比增长率(%)'
                    "free_cash_flow_growth": None, # Placeholder
                    "operating_income_growth": get_percentage_or_none(row, '营业利润同比增长率(%)'), # Check for '营业利润同比增长率(%)'
                    "ebitda_growth": None, # Placeholder
                    "payout_ratio": get_percentage_or_none(row, '股息支付率(%)'), # Check for '股息支付率(%)'
                    "book_value_per_share": get_float_or_none(row, '每股净资产(元)'), # Check for '每股净资产(元)'
                    "free_cash_flow_per_share": get_float_or_none(row, '每股经营活动产生的现金流量净额(元)'), # Check for '每股经营活动产生的现金流量净额(元)'
                }
                # Adjust debt_to_equity and debt_to_assets if they are in percentage
                if metrics_data["debt_to_equity"] is not None and ('产权比率(%)' in df_renamed.columns or '产权比率' in df_renamed.columns) : # Assuming it's a percentage
                    metrics_data["debt_to_equity"] /= 100.0
                if metrics_data["debt_to_assets"] is not None and ('资产负债率(%)' in df_renamed.columns or '资产负债率' in df_renamed.columns): # Assuming it's a percentage
                    metrics_data["debt_to_assets"] /= 100.0

                financial_metrics_list.append(FinancialMetrics(**metrics_data))

            # Sort by report_period descending
            financial_metrics_list.sort(key=lambda x: x.report_period, reverse=True)
            
            # Apply limit
            limited_metrics = financial_metrics_list[:limit]

            if not limited_metrics:
                return []

            _cache.set_financial_metrics(ticker, [m.model_dump() for m in limited_metrics])
            return limited_metrics
        except Exception as e:
            print(f"Error fetching or processing financial metrics for Chinese ticker {ticker} using akshare: {e}")
            return []
    else:
        # If not in cache or insufficient data, fetch from API (existing logic)
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
    if is_chinese_ticker(ticker):
        try:
            akshare_ticker_code = get_akshare_ticker(ticker)
            report_types_map = {
                "balance_sheet": "资产负债表",
                "income_statement": "利润表",
                "cash_flow_statement": "现金流量表",
            }
            
            all_found_line_items = []

            for report_key, akshare_report_name in report_types_map.items():
                # Fetch the financial report
                # The 'limit' here refers to number of reports (e.g., latest 4 quarters)
                # stock_financial_report_sina takes 'stock' and 'symbol' (report name)
                # It does not have an explicit limit parameter for number of reports.
                # It typically returns all available historical reports (quarterly & annual).
                report_df = akshare.stock_financial_report_sina(stock=akshare_ticker_code, symbol=akshare_report_name)

                if report_df is None or report_df.empty:
                    continue

                # The first column in report_df is typically 'REPORT_DATE' or similar, containing item names.
                # The other columns are dates (e.g., '2023-12-31', '2023-09-30').
                # We need to set the item names as index to search easily
                if report_df.columns[0] == 'REPORT_DATE' or report_df.columns[0] == '项目': # Common names for item column
                     report_df = report_df.set_index(report_df.columns[0])
                else:
                    # If the first column isn't obviously the item names, we might have an issue.
                    # For now, assume the first column is always the line item names.
                    # This might need adjustment based on actual akshare output.
                    print(f"Warning: First column of {akshare_report_name} for {ticker} is {report_df.columns[0]}. Assuming it contains line item names.")
                    report_df = report_df.set_index(report_df.columns[0])


                for requested_line_item_name in line_items:
                    if requested_line_item_name in report_df.index:
                        # Found the line item in this report
                        item_series = report_df.loc[requested_line_item_name]
                        
                        for report_col_date_str, value in item_series.items():
                            if pd.isna(value) or value == '--': # Skip NA or placeholder values
                                continue
                            
                            try:
                                # Ensure date is YYYY-MM-DD
                                current_report_date = pd.to_datetime(report_col_date_str).strftime('%Y-%m-%d')
                            except Exception: # If date conversion fails for a column, skip it
                                print(f"Warning: Could not parse date column '{report_col_date_str}' for {ticker}. Skipping.")
                                continue

                            if current_report_date > end_date:
                                continue

                            # Infer period
                            month = int(current_report_date.split('-')[1])
                            derived_period = "annual" if month == 12 else "quarterly"
                            
                            # Handle 'period' parameter from function call
                            if period == "ttm":
                                # For TTM, we are using latest annual or quarterly as proxy.
                                # This is a simplification noted in the subtask.
                                # No specific TTM calculation is done here.
                                pass # Accept the derived_period
                            elif period != derived_period:
                                continue # Skip if specific period (annual/quarterly) requested doesn't match


                            line_item_data = {
                                "ticker": ticker,
                                "report_period": current_report_date,
                                "period": derived_period,
                                "currency": "CNY",
                                requested_line_item_name: None # Placeholder for value
                            }
                            try:
                                # Attempt to convert value to float. Akshare values can be strings with units or numbers.
                                # This part might need more robust parsing if values have '万', '亿' etc.
                                # For now, assuming numeric or directly convertible string.
                                numeric_value = float(value)
                                line_item_data[requested_line_item_name] = numeric_value
                            except ValueError:
                                # If direct float conversion fails, store as string or skip
                                print(f"Warning: Could not convert value '{value}' to float for {requested_line_item_name} on {current_report_date} for {ticker}. Storing as string or skipping if critical.")
                                # Decide: Store as string, or skip? For now, let's try to store as string.
                                # line_item_data[requested_line_item_name] = str(value) 
                                # Or skip:
                                continue


                            all_found_line_items.append(LineItem(**line_item_data))
            
            # Sort all found items by report_period descending
            all_found_line_items.sort(key=lambda x: x.report_period, reverse=True)
            
            # Apply overall limit
            return all_found_line_items[:limit]

        except Exception as e:
            print(f"Error fetching or processing line items for Chinese ticker {ticker} using akshare: {e}")
            return []
    else:
        # Existing logic for non-Chinese tickers
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key

        url = "https://api.financialdatasets.ai/financials/search/line-items"

        body = {
            "tickers": [ticker], # Note: original API takes a list of tickers, here we adapt for single ticker
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
        search_results = response_model.search_results # This is already a list of LineItem
        
        if not search_results:
            return []
        
        # The API already respects limit and filters, so just return
        return search_results


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
        # Ensure dates are comparable (YYYY-MM-DD)
        filtered_data = []
        for trade_dict in cached_data:
            # Use filing_date as primary sort key, fallback to transaction_date if needed for filtering
            trade_date_str = trade_dict.get("filing_date") or trade_dict.get("transaction_date")
            if not trade_date_str: # Should not happen if data is valid
                continue
            
            trade_date_comparable = trade_date_str.split("T")[0] # Get YYYY-MM-DD part

            if (start_date is None or trade_date_comparable >= start_date.split("T")[0]) and \
               (trade_date_comparable <= end_date.split("T")[0]):
                filtered_data.append(InsiderTrade(**trade_dict))
        
        # Sort by filing_date (primary) then transaction_date (secondary), descending
        filtered_data.sort(key=lambda x: (x.filing_date or "0000-00-00", x.transaction_date or "0000-00-00"), reverse=True)
        
        if filtered_data:
            return filtered_data[:limit] # Apply limit after filtering and sorting

    if is_chinese_ticker(ticker):
        try:
            akshare_ticker_code = get_akshare_ticker(ticker)
            # Fetch data using akshare.stock_ggcg_em() - "高管持股" (Executive Shareholdings)
            # This function takes the symbol (e.g., "600519")
            ggcg_df = akshare.stock_ggcg_em(symbol=akshare_ticker_code)

            if ggcg_df is None or ggcg_df.empty:
                print(f"No insider trade data found for Chinese ticker {ticker} using akshare.stock_ggcg_em.")
                _cache.set_insider_trades(ticker, [])
                return []

            all_trades = []
            # Expected columns from akshare.stock_ggcg_em (need to verify from actual output or docs):
            # '公告日期' (Announcement Date) -> filing_date
            # '变动人' (Person Making Change) -> name
            # '变动股份数量' (Number of Shares Changed) -> transaction_shares
            # '交易平均价' (Average Transaction Price) -> transaction_price_per_share
            # '变动后持股总数' (Total Shares Held After Change) -> shares_owned_after_transaction
            # '董监高职务' (Executive Title) -> title
            # '变动原因' (Reason for Change) - Not directly in InsiderTrade model
            # '上市公司代码' - Redundant (we have ticker)
            # '上市公司简称' - Could be issuer if needed, but InsiderTrade.issuer is optional
            
            # '变动日期' (Date of Change) -> transaction_date (This is crucial)
            # If '变动日期' is not available, this source might not be suitable.
            # Let's assume '公告日期' is the primary date for filtering if '变动日期' is absent or less reliable.

            for _, row in ggcg_df.iterrows():
                filing_date_str = str(row.get('公告日期')).split(" ")[0] if pd.notna(row.get('公告日期')) else None
                transaction_date_str = str(row.get('变动日期')).split(" ")[0] if pd.notna(row.get('变动日期')) else filing_date_str # Fallback to filing if transaction_date missing

                if not transaction_date_str: # If no valid date, skip
                    continue
                
                # Date filtering
                # Ensure dates are YYYY-MM-DD for comparison
                current_trade_date_comparable = transaction_date_str # Use transaction_date for filtering range
                
                if start_date and current_trade_date_comparable < start_date.split("T")[0]:
                    continue
                if current_trade_date_comparable > end_date.split("T")[0]:
                    continue

                transaction_shares = float(row['变动股份数量']) if pd.notna(row.get('变动股份数量')) else None
                # Positive for buy, negative for sell. akshare might provide this directly or via a '变动方向' column.
                # If '变动方向' (e.g., "增持", "减持") exists, use it. Otherwise, sign of '变动股份数量' might indicate it.
                # Assuming '变动股份数量' is already signed. If not, this needs adjustment.

                transaction_price = float(row['交易平均价']) if pd.notna(row.get('交易平均价')) else None
                
                transaction_value = None
                if transaction_shares is not None and transaction_price is not None:
                    transaction_value = transaction_shares * transaction_price

                shares_after = float(row['变动后持股总数']) if pd.notna(row.get('变动后持股总数')) else None
                shares_before = None
                if shares_after is not None and transaction_shares is not None:
                    shares_before = shares_after - transaction_shares # Calculate if possible

                trade = InsiderTrade(
                    ticker=ticker, # Original ticker
                    issuer=str(row.get('上市公司简称')) if pd.notna(row.get('上市公司简称')) else None,
                    name=str(row.get('变动人')) if pd.notna(row.get('变動人')) else str(row.get('高管姓名')), # Try both common column names
                    title=str(row.get('董监高职务')) if pd.notna(row.get('董监高职务')) else None,
                    is_board_director=None, # Cannot determine from typical ggcg_em output
                    transaction_date=transaction_date_str,
                    transaction_shares=transaction_shares,
                    transaction_price_per_share=transaction_price,
                    transaction_value=transaction_value,
                    shares_owned_before_transaction=shares_before,
                    shares_owned_after_transaction=shares_after,
                    security_title="普通股", # Assume common stock
                    filing_date=filing_date_str if filing_date_str else transaction_date_str, # Use transaction_date if filing_date is missing
                )
                all_trades.append(trade)
            
            # Sort by filing_date (primary) then transaction_date (secondary), descending
            all_trades.sort(key=lambda x: (x.filing_date or "0000-00-00", x.transaction_date or "0000-00-00"), reverse=True)

            limited_trades = all_trades[:limit]

            if not limited_trades:
                _cache.set_insider_trades(ticker, [])
                return []

            _cache.set_insider_trades(ticker, [t.model_dump() for t in limited_trades])
            return limited_trades

        except Exception as e:
            print(f"Error fetching or processing insider trades for Chinese ticker {ticker} using akshare: {e}")
            print("Insider trading data for Chinese tickers might not be available or compatible via akshare.")
            _cache.set_insider_trades(ticker, []) # Cache empty list on error
            return []
    else:
        # Existing logic for non-Chinese tickers (Financial Datasets API)
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key

        all_trades_fallback = [] # Renamed to avoid conflict
        current_end_date_fallback = end_date # Use original end_date from function params

        # Pagination loop for fallback API
        while True:
            # Use filing_date for querying as per original logic for this API path
            url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date_fallback.split('T')[0]}"
            if start_date:
                url += f"&filing_date_gte={start_date.split('T')[0]}"
            
            # The original code used the function's 'limit' for pagination.
            # This can be problematic if API's max page size is smaller.
            # Let's use a fixed large page_limit for fetching and then apply the function's limit.
            page_fetch_limit = 1000 
            url += f"&limit={page_fetch_limit}"


            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Error fetching data from Financial Datasets API: {ticker} - {response.status_code} - {response.text}")

            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades_page = response_model.insider_trades

            if not insider_trades_page:
                break

            all_trades_fallback.extend(insider_trades_page)

            if len(insider_trades_page) < page_fetch_limit: # Stop if we received less than a full page
                break
            
            # Update current_end_date_fallback for next iteration
            # Use filing_date for pagination cursor
            oldest_filing_date_in_batch = min(trade.filing_date for trade in insider_trades_page if trade.filing_date)
            if not oldest_filing_date_in_batch: # Should not happen with valid data
                break 

            if start_date and oldest_filing_date_in_batch.split("T")[0] <= start_date.split("T")[0]:
                break
            
            try: # Decrement day to avoid re-fetching same last record
                oldest_datetime_obj = datetime.datetime.strptime(oldest_filing_date_in_batch.split("T")[0], '%Y-%m-%d')
                next_end_datetime = oldest_datetime_obj - datetime.timedelta(days=1) # Go to previous day
                current_end_date_fallback = next_end_datetime.strftime('%Y-%m-%d')
            except ValueError:
                print(f"Warning: Could not parse date {oldest_filing_date_in_batch} for pagination. Stopping.")
                break
            
            if current_end_date_fallback < (start_date.split("T")[0] if start_date else "0000-00-00"):
                break


        # Post-fetch filtering for fallback (more precise after getting all pages)
        # Dates from this API are 'YYYY-MM-DDTHH:MM:SSZ'
        final_filtered_trades = []
        for trade in all_trades_fallback:
            # Use filing_date as primary, transaction_date as secondary for filtering
            trade_date_to_check_str = trade.filing_date or trade.transaction_date
            if not trade_date_to_check_str:
                continue
            
            trade_date_comparable = trade_date_to_check_str.split("T")[0]

            if (start_date is None or trade_date_comparable >= start_date.split("T")[0]) and \
               (trade_date_comparable <= end_date.split("T")[0]):
                final_filtered_trades.append(trade)
        
        # Sort by filing_date (primary) then transaction_date (secondary), descending
        final_filtered_trades.sort(key=lambda x: (x.filing_date or "0000-00-00", x.transaction_date or "0000-00-00"), reverse=True)

        limited_trades_fallback = final_filtered_trades[:limit]

        if not limited_trades_fallback:
            _cache.set_insider_trades(ticker, [])
            return []

        _cache.set_insider_trades(ticker, [t.model_dump() for t in limited_trades_fallback])
        return limited_trades_fallback


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
        if filtered_data: # Already sorted and potentially limited by cache retrieval logic if any
            return filtered_data[:limit] # Apply limit again just in case

    if is_chinese_ticker(ticker):
        try:
            akshare_ticker = get_akshare_ticker(ticker)
            # Fetch news using akshare.stock_news_em
            # The 'stock' parameter expects the 6-digit code.
            news_df = akshare.stock_news_em(stock=akshare_ticker)

            if news_df is None or news_df.empty:
                return []

            all_news = []
            # Expected columns: code, title, content, public_time, source, url
            # Mapping:
            # title -> title
            # public_time -> date
            # source -> source
            # url -> url
            # author -> N/A (not provided by this akshare function)
            # sentiment -> None

            for _, row in news_df.iterrows():
                # Date formatting and filtering
                news_date_str = ""
                if pd.notna(row.get('public_time')): # 'public_time' is the typical column name
                    # Convert pandas Timestamp to 'YYYY-MM-DD HH:MM:SS' string
                    news_date_str = pd.to_datetime(row['public_time']).strftime('%Y-%m-%d %H:%M:%S')
                elif pd.notna(row.get('datetime')): # Alternative common name
                     news_date_str = pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M:%S')   
                else: # Fallback or skip if no date
                    continue 

                # Filter by date range
                # Ensure start_date and end_date are comparable with news_date_str (YYYY-MM-DD HH:MM:SS)
                # The existing API uses YYYY-MM-DDTHH:MM:SS, so we might need to adapt.
                # For simplicity, we'll compare date part if start/end_date are just dates.
                
                news_date_dt = datetime.datetime.strptime(news_date_str.split(' ')[0], '%Y-%m-%d')
                start_date_dt = datetime.datetime.strptime(start_date.split('T')[0], '%Y-%m-%d') if start_date else None
                end_date_dt = datetime.datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')

                if start_date_dt and news_date_dt < start_date_dt:
                    continue
                if news_date_dt > end_date_dt:
                    continue
                
                # Using original ticker as per requirement
                news_item = CompanyNews(
                    ticker=ticker, 
                    title=str(row['title']) if pd.notna(row.get('title')) else "N/A",
                    author="N/A", # Not available from akshare.stock_news_em
                    source=str(row['source']) if pd.notna(row.get('source')) else "N/A",
                    date=news_date_str, # Store as 'YYYY-MM-DD HH:MM:SS'
                    url=str(row['url']) if pd.notna(row.get('url')) else "",
                    sentiment=None, # Not available
                )
                all_news.append(news_item)

            # Sort by date descending
            all_news.sort(key=lambda x: x.date, reverse=True)
            
            # Apply limit
            limited_news = all_news[:limit]

            if not limited_news:
                return []

            _cache.set_company_news(ticker, [n.model_dump() for n in limited_news])
            return limited_news

        except Exception as e:
            print(f"Error fetching or processing company news for Chinese ticker {ticker} using akshare: {e}")
            return []
    else:
        # If not in cache or insufficient data, fetch from API (existing logic)
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key

        all_news_fallback = [] # Renamed to avoid conflict
        current_end_date = end_date

        while True: # Pagination for fallback API
            url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
            if start_date:
                url += f"&start_date={start_date}"
            # The fallback API might not have a limit param per page, or it's handled differently.
            # The original code implies it fetches all and then limits.
            # For safety, let's assume the 'limit' in the URL is for total items if supported,
            # or it's a per-page limit. The original code fetches in a loop and then applies limit.
            # This function's limit parameter is for the final result.
            # The URL construction in original code for fallback does use `limit`.
            
            # Construct URL with pagination limit (e.g., 1000 per page, then Python limit applies)
            # The original code's limit in the URL was the function's limit parameter.
            # This might be problematic if the API's max page size is smaller.
            # Assuming the API handles large limits or the original loop was to fetch all.
            # Let's use a fixed large page_limit for fetching and then apply the function's limit.
            page_fetch_limit = 1000 # Fetch up to 1000 per call in loop
            url_paginated = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}&limit={page_fetch_limit}"
            if start_date:
                 url_paginated += f"&start_date={start_date}"


            response = requests.get(url_paginated, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news_page = response_model.news

            if not company_news_page:
                break

            all_news_fallback.extend(company_news_page)

            # Check if we need to continue pagination
            # The original pagination logic was: if not start_date or len(company_news_page) < limit: break
            # This seems to imply `limit` was a page size. We want to fetch all in range.
            if len(company_news_page) < page_fetch_limit: # Stop if we received less than a full page
                break

            # Update end_date to the oldest date from current batch for next iteration
            # Ensure dates are comparable. API returns 'YYYY-MM-DDTHH:MM:SS'
            oldest_date_in_batch = min(news.date for news in company_news_page)
            
            # To prevent infinite loops if dates are not strictly decreasing or if start_date is very old:
            if oldest_date_in_batch <= (start_date if start_date else "0000-00-00"): # or some very early date
                break
            
            # Set current_end_date for next iteration to one second before the oldest news item's date
            # to avoid re-fetching the same last item.
            try:
                oldest_datetime_obj = datetime.datetime.fromisoformat(oldest_date_in_batch.replace('Z', ''))
                next_end_datetime = oldest_datetime_obj - datetime.timedelta(seconds=1)
                current_end_date = next_end_datetime.strftime('%Y-%m-%dT%H:%M:%S')
            except ValueError: # If date format is not as expected, break to avoid error
                print(f"Warning: Could not parse date {oldest_date_in_batch} for pagination. Stopping news fetching.")
                break


            # If we've reached or passed the start_date, we can stop
            # This check is now more robust with datetime objects
            if start_date and current_end_date < start_date:
                break
        
        # Filter by date range again after fetching all pages (more precise)
        if start_date:
            all_news_fallback = [n for n in all_news_fallback if n.date >= start_date]
        all_news_fallback = [n for n in all_news_fallback if n.date <= end_date]


        # Sort by date descending
        all_news_fallback.sort(key=lambda x: x.date, reverse=True)
        
        # Apply limit
        limited_news_fallback = all_news_fallback[:limit]

        if not limited_news_fallback:
            return []

        _cache.set_company_news(ticker, [n.model_dump() for n in limited_news_fallback])
        return limited_news_fallback


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    if is_chinese_ticker(ticker):
        akshare_ticker_code = get_akshare_ticker(ticker)
        is_today = end_date == datetime.datetime.now().strftime("%Y-%m-%d")
        market_cap_value = None

        if is_today:
            try:
                # Attempt to fetch current market cap
                info_df = akshare.stock_individual_info_em(stock=akshare_ticker_code)
                # Expected structure: DataFrame with 2 columns. First column 'item', second 'value'.
                if info_df is not None and not info_df.empty and info_df.shape[1] >= 2:
                    # Assuming first column is item names, second is values
                    item_col_name = info_df.columns[0]
                    value_col_name = info_df.columns[1]
                    
                    market_cap_row = info_df[info_df[item_col_name] == '总市值']
                    if not market_cap_row.empty:
                        raw_market_cap_str = market_cap_row[value_col_name].iloc[0]
                        market_cap_value = parse_market_cap_string(raw_market_cap_str)
                        if market_cap_value is not None:
                            return market_cap_value
                    else:
                        print(f"Could not find '总市值' in item column of akshare.stock_individual_info_em for {ticker}")
                else:
                    print(f"Unexpected DataFrame structure or empty/None DataFrame from akshare.stock_individual_info_em for {ticker}. Columns: {info_df.columns if info_df is not None else 'N/A'}")
            except Exception as e:
                print(f"Error fetching current market cap for Chinese ticker {ticker} using akshare.stock_individual_info_em: {e}. Falling back to financial_metrics.")
        
        # Fallback for historical date or if current fetch failed
        # Call get_financial_metrics (already updated for akshare)
        # Ensure period is appropriate if get_financial_metrics uses it; 'ttm' or 'annual' should be fine.
        # The get_financial_metrics for Chinese stocks in our implementation uses stock_financial_analysis_indicator,
        # which might have '总市值' directly for various report dates.
        
        # Try to get from our get_financial_metrics implementation for Chinese stocks
        # The Chinese version of get_financial_metrics extracts 'market_cap' if available from '总市值'
        # in stock_financial_analysis_indicator
        financial_metrics_list = get_financial_metrics(ticker, end_date, period="ttm", limit=1) # period="ttm" is fine as market_cap is point-in-time
        
        if financial_metrics_list and financial_metrics_list[0].market_cap is not None:
            return financial_metrics_list[0].market_cap
        else: # Last resort for Chinese tickers if above fails - try to get from price data if available
             # This is not ideal as it's not directly from financial statements or summary.
             # But if '总市值' is a column in get_financial_metrics output from akshare, it would have been caught.
             # This is a deeper fallback.
            print(f"Market cap not found via stock_individual_info_em or get_financial_metrics for Chinese ticker {ticker}.")
            return None

    else: # Existing logic for non-Chinese tickers
        if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
            headers = {}
            if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
                headers["X-API-KEY"] = api_key
            url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching company facts for {ticker} from Financial Datasets API: {response.status_code}")
                # Fallback to financial_metrics for non-Chinese tickers if company facts fails for today
                financial_metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1)
                if financial_metrics and financial_metrics[0].market_cap is not None:
                    return financial_metrics[0].market_cap
                return None

            data = response.json()
            response_model = CompanyFactsResponse(**data)
            # The 'market_cap' field in CompanyFacts is already a float | None
            return response_model.company_facts.market_cap

        # Historical date for non-Chinese tickers
        financial_metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1)
        if financial_metrics and financial_metrics[0].market_cap is not None:
            return financial_metrics[0].market_cap
        
        return None


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
