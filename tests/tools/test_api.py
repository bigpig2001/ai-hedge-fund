import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import datetime

# Assuming src.tools.api and src.data.models are in PYTHONPATH
from src.tools.api import (
    is_chinese_ticker,
    get_akshare_ticker,
    parse_market_cap_string,
    get_prices,
    # get_financial_metrics, # Add other functions as they are tested
    # search_line_items,
    # get_insider_trades,
    # get_company_news,
    # get_market_cap
)
from src.data.models import Price, PriceResponse # Add other models as needed
from src.data.cache import Cache # For type hinting mock cache

# Mock the global _cache object if it's directly used and needs control
# For this example, we'll patch specific methods on it where get_prices uses it.

class TestApiHelperFunctions(unittest.TestCase):

    def test_is_chinese_ticker(self):
        self.assertTrue(is_chinese_ticker("600519.SH"))
        self.assertTrue(is_chinese_ticker("000001.SZ"))
        self.assertFalse(is_chinese_ticker("AAPL.US"))
        self.assertFalse(is_chinese_ticker("MSFT"))
        self.assertFalse(is_chinese_ticker("000001.HK"))
        self.assertFalse(is_chinese_ticker("SH600519")) # Missing dot

    def test_get_akshare_ticker(self):
        self.assertEqual(get_akshare_ticker("600519.SH"), "600519")
        self.assertEqual(get_akshare_ticker("000001.SZ"), "000001")
        # Test with non-Chinese ticker (should return original, though typically guarded by is_chinese_ticker)
        self.assertEqual(get_akshare_ticker("AAPL.US"), "AAPL.US")

    def test_parse_market_cap_string(self):
        self.assertEqual(parse_market_cap_string("100亿"), 100 * 10**8)
        self.assertEqual(parse_market_cap_string("50.5万"), 50.5 * 10**4)
        self.assertEqual(parse_market_cap_string("12345.67"), 12345.67)
        self.assertEqual(parse_market_cap_string(12345.67), 12345.67)
        self.assertEqual(parse_market_cap_string(None), None)
        self.assertEqual(parse_market_cap_string("invalid_string"), None)
        self.assertEqual(parse_market_cap_string(""), None)
        self.assertEqual(parse_market_cap_string("100亿万"), None) # Invalid format
        self.assertEqual(parse_market_cap_string("一百亿"), None) # Non-numeric before unit


class TestGetPrices(unittest.TestCase):
    # We will patch _cache.get_prices and _cache.set_prices within each test method
    # where direct interaction with cache methods is tested.

    @patch('src.tools.api.akshare.stock_zh_a_hist')
    @patch('src.tools.api._cache.set_prices')
    @patch('src.tools.api._cache.get_prices')
    def test_get_prices_chinese_ticker_akshare_success(self, mock_cache_get, mock_cache_set, mock_akshare_hist):
        # 1. Setup: Cache miss, akshare returns data
        mock_cache_get.return_value = None

        sample_akshare_data = {
            '日期': [datetime.date(2023, 1, 3), datetime.date(2023, 1, 4), datetime.date(2023, 1, 5)],
            '开盘': [10.0, 10.2, 10.5],
            '收盘': [10.1, 10.4, 10.3],
            '最高': [10.3, 10.5, 10.6],
            '最低': [9.9, 10.1, 10.2],
            '成交量': [1000, 1200, 1100]
        }
        mock_df = pd.DataFrame(sample_akshare_data)
        mock_akshare_hist.return_value = mock_df

        # 2. Call the function
        ticker = "600519.SH"
        start_date = "2023-01-01"
        end_date = "2023-01-05" # Includes all sample data
        prices = get_prices(ticker, start_date, end_date)

        # 3. Assertions
        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_hist.assert_called_once_with(symbol="600519", start_date="20230101", end_date="20230105", adjust="qfq")
        
        self.assertEqual(len(prices), 3)
        self.assertIsInstance(prices[0], Price)
        self.assertEqual(prices[0].open, 10.0)
        self.assertEqual(prices[0].close, 10.1)
        self.assertEqual(prices[0].time, "2023-01-03")
        self.assertEqual(prices[1].volume, 1200)
        self.assertEqual(prices[1].time, "2023-01-04")
        self.assertEqual(prices[2].low, 10.2)
        self.assertEqual(prices[2].time, "2023-01-05")

        # Check that set_prices was called with the correct data
        expected_cache_data = [p.model_dump() for p in prices]
        mock_cache_set.assert_called_once_with(ticker, expected_cache_data)

    @patch('src.tools.api.akshare.stock_zh_a_hist')
    @patch('src.tools.api._cache.set_prices') # Still need to mock set_prices as it might be called with []
    @patch('src.tools.api._cache.get_prices')
    def test_get_prices_chinese_ticker_akshare_error(self, mock_cache_get, mock_cache_set, mock_akshare_hist):
        # 1. Setup: Cache miss, akshare raises an error
        mock_cache_get.return_value = None
        mock_akshare_hist.side_effect = Exception("AKShare API error")

        # 2. Call the function
        ticker = "600519.SH"
        start_date = "2023-01-01"
        end_date = "2023-01-05"
        prices = get_prices(ticker, start_date, end_date)

        # 3. Assertions
        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_hist.assert_called_once_with(symbol="600519", start_date="20230101", end_date="20230105", adjust="qfq")
        self.assertEqual(prices, [])
        mock_cache_set.assert_not_called() # Should not attempt to cache on error before data transformation

    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_prices')
    @patch('src.tools.api._cache.get_prices')
    @patch('src.tools.api.akshare.stock_zh_a_hist') # Also patch akshare to ensure it's not called
    def test_get_prices_non_chinese_ticker_fallback_success(self, mock_akshare_hist, mock_cache_get, mock_cache_set, mock_requests_get):
        # 1. Setup: Cache miss, requests.get returns data
        mock_cache_get.return_value = None
        
        mock_response = MagicMock()
        sample_api_data = {
            "ticker": "AAPL.US",
            "prices": [
                {"open": 150.0, "close": 151.0, "high": 152.0, "low": 149.0, "volume": 10000, "time": "2023-01-03T00:00:00"},
                {"open": 151.0, "close": 150.5, "high": 151.5, "low": 150.0, "volume": 12000, "time": "2023-01-04T00:00:00"}
            ]
        }
        mock_response.json.return_value = sample_api_data
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        # 2. Call the function
        ticker = "AAPL.US"
        start_date = "2023-01-01"
        end_date = "2023-01-05"
        prices = get_prices(ticker, start_date, end_date)

        # 3. Assertions
        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_hist.assert_not_called() # Ensure akshare was NOT called
        mock_requests_get.assert_called_once() # Check that requests.get was called
        
        self.assertEqual(len(prices), 2)
        self.assertIsInstance(prices[0], Price)
        self.assertEqual(prices[0].open, 150.0)
        # The Price model expects 'time' as str, the API returns it as str
        self.assertEqual(prices[0].time, "2023-01-03T00:00:00") 

        expected_cache_data = [p.model_dump() for p in prices]
        mock_cache_set.assert_called_once_with(ticker, expected_cache_data)

    @patch('src.tools.api.akshare.stock_zh_a_hist')
    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_prices')
    @patch('src.tools.api._cache.get_prices')
    def test_get_prices_cache_hit(self, mock_cache_get, mock_cache_set, mock_requests_get, mock_akshare_hist):
        # 1. Setup: Cache hit
        cached_price_data = [
            Price(open=10.0, close=10.1, high=10.3, low=9.9, volume=1000, time="2023-01-03").model_dump(),
            Price(open=10.2, close=10.4, high=10.5, low=10.1, volume=1200, time="2023-01-04").model_dump(),
            Price(open=10.5, close=10.3, high=10.6, low=10.2, volume=1100, time="2023-01-05").model_dump() # Outside range
        ]
        mock_cache_get.return_value = cached_price_data

        # 2. Call the function
        ticker = "600519.SH" # Could be any ticker, path chosen by cache
        start_date = "2023-01-01"
        end_date = "2023-01-04" # Filter to exclude the last item
        prices = get_prices(ticker, start_date, end_date)

        # 3. Assertions
        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_hist.assert_not_called()
        mock_requests_get.assert_not_called()
        mock_cache_set.assert_not_called()

        self.assertEqual(len(prices), 2)
        self.assertEqual(prices[0].time, "2023-01-03")
        self.assertEqual(prices[1].time, "2023-01-04")
        
    @patch('src.tools.api.akshare.stock_zh_a_hist')
    @patch('src.tools.api._cache.set_prices')
    @patch('src.tools.api._cache.get_prices')
    def test_get_prices_chinese_ticker_akshare_empty_df(self, mock_cache_get, mock_cache_set, mock_akshare_hist):
        # 1. Setup: Cache miss, akshare returns empty DataFrame
        mock_cache_get.return_value = None
        mock_akshare_hist.return_value = pd.DataFrame() # Empty DataFrame

        # 2. Call the function
        ticker = "600519.SH"
        start_date = "2023-01-01"
        end_date = "2023-01-05"
        prices = get_prices(ticker, start_date, end_date)

        # 3. Assertions
        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_hist.assert_called_once_with(symbol="600519", start_date="20230101", end_date="20230105", adjust="qfq")
        self.assertEqual(prices, [])
        mock_cache_set.assert_not_called() # No data to set

    @patch('src.tools.api.akshare.stock_zh_a_hist')
    @patch('src.tools.api._cache.set_prices')
    @patch('src.tools.api._cache.get_prices')
    def test_get_prices_chinese_ticker_akshare_data_out_of_range(self, mock_cache_get, mock_cache_set, mock_akshare_hist):
        # 1. Setup: Cache miss, akshare returns data but it's all outside the requested date range
        mock_cache_get.return_value = None

        sample_akshare_data = {
            '日期': [datetime.date(2022, 12, 30), datetime.date(2022, 12, 31)],
            '开盘': [10.0, 10.2],
            '收盘': [10.1, 10.4],
            '最高': [10.3, 10.5],
            '最低': [9.9, 10.1],
            '成交量': [1000, 1200]
        }
        mock_df = pd.DataFrame(sample_akshare_data)
        mock_akshare_hist.return_value = mock_df
        
        # 2. Call the function
        ticker = "600519.SH"
        start_date = "2023-01-01"
        end_date = "2023-01-05"
        prices = get_prices(ticker, start_date, end_date)

        # 3. Assertions
        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_hist.assert_called_once_with(symbol="600519", start_date="20230101", end_date="20230105", adjust="qfq")
        self.assertEqual(prices, []) # Should be empty as all data is out of range
        mock_cache_set.assert_not_called() # Or called with empty list, depending on implementation detail of get_prices when no valid prices are transformed. Current code returns [] before caching.


# Ensure all necessary models and functions are imported.
from src.tools.api import (
    is_chinese_ticker,
    get_akshare_ticker,
    parse_market_cap_string,
    get_prices,
    get_financial_metrics, 
    get_company_news,
    get_insider_trades,
    search_line_items,
    get_market_cap
)
from src.data.models import (
    Price, PriceResponse, FinancialMetrics, CompanyNews, InsiderTrade, LineItem,
    FinancialMetricsResponse, CompanyNewsResponse, InsiderTradeResponse, LineItemResponse # Added Response models for fallback tests
)
# The rest of the file content (helper function tests, TestGetPrices) remains the same.
# ... (existing TestApiHelperFunctions and TestGetPrices code) ...

if __name__ == '__main__':
    unittest.main()


class TestGetFinancialMetrics(unittest.TestCase):

    @patch('src.tools.api.akshare.stock_financial_analysis_indicator')
    @patch('src.tools.api._cache.set_financial_metrics')
    @patch('src.tools.api._cache.get_financial_metrics')
    def test_get_financial_metrics_chinese_ticker_akshare_success(self, mock_cache_get, mock_cache_set, mock_akshare_indicator):
        mock_cache_get.return_value = None
        sample_akshare_data = {
            '日期': ["2023-12-31", "2023-09-30"],
            '总市值': [2.01e+12, 1.95e+12], '市盈率(PE)': [30.5, 29.0], '市净率(PB)': [5.6, 5.4],
            '净资产收益率(ROE)': [18.5, 17.9], '净利润同比增长率(%)': [10.2, 9.5],
            '营业总收入同比增长率(%)': [5.5, 5.0], '基本每股收益(元)': [3.50, 3.40],
            '销售毛利率(%)': [60.1, 59.8], '流动比率': [1.2, 1.15], '每股净资产(元)': [20.50, 20.00],
            '市盈率(PEG)': [1.5, 1.4], '营业利润率(%)': [25.5, 25.0], '销售净利率(%)': [15.5, 15.0],
            '总资产报酬率(ROA)(%)': [8.5, 8.0], '总资产周转率(次)': [0.5, 0.45],
            '存货周转率(次)': [3.0, 2.8], '应收账款周转率(次)': [4.0, 3.8], '速动比率': [0.8, 0.75],
            '产权比率(%)': [120.0, 118.0], '资产负债率(%)': [55.0, 54.0],
            '利息保障倍数(倍)': [10.0, 9.5], '每股收益同比增长率(%)': [10.0, 9.0],
            '营业利润同比增长率(%)': [12.0, 11.0], '股息支付率(%)': [30.0, 28.0],
            '每股经营活动产生的现金流量净额(元)': [5.0, 4.5]
        }
        mock_df = pd.DataFrame(sample_akshare_data)
        mock_akshare_indicator.return_value = mock_df

        ticker = "600519.SH"
        end_date = "2023-12-31"
        metrics = get_financial_metrics(ticker, end_date, limit=5)

        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_indicator.assert_called_once_with(stock="600519")
        self.assertEqual(len(metrics), 2)
        metric1 = metrics[0]
        self.assertEqual(metric1.ticker, ticker)
        self.assertEqual(metric1.report_period, "2023-12-31")
        self.assertEqual(metric1.period, "annual")
        self.assertAlmostEqual(metric1.market_cap, 2.01e+12)
        self.assertAlmostEqual(metric1.return_on_equity, 0.185)
        self.assertAlmostEqual(metric1.debt_to_equity, 1.20)
        self.assertAlmostEqual(metric1.debt_to_assets, 0.55)
        expected_cache_data = [m.model_dump() for m in metrics]
        mock_cache_set.assert_called_once_with(ticker, expected_cache_data)

    @patch('src.tools.api.akshare.stock_financial_analysis_indicator')
    @patch('src.tools.api._cache.set_financial_metrics')
    @patch('src.tools.api._cache.get_financial_metrics')
    def test_get_financial_metrics_chinese_ticker_akshare_error(self, mock_cache_get, mock_cache_set, mock_akshare_indicator):
        mock_cache_get.return_value = None
        mock_akshare_indicator.side_effect = Exception("AKShare API error")
        metrics = get_financial_metrics("600519.SH", "2023-12-31")
        self.assertEqual(metrics, [])
        mock_cache_set.assert_not_called()

    @patch('src.tools.api.akshare.stock_financial_analysis_indicator')
    @patch('src.tools.api._cache.set_financial_metrics')
    @patch('src.tools.api._cache.get_financial_metrics')
    def test_get_financial_metrics_chinese_ticker_akshare_empty(self, mock_cache_get, mock_cache_set, mock_akshare_indicator):
        mock_cache_get.return_value = None
        mock_akshare_indicator.return_value = pd.DataFrame()
        metrics = get_financial_metrics("600519.SH", "2023-12-31")
        self.assertEqual(metrics, [])
        mock_cache_set.assert_not_called()

    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_financial_metrics')
    @patch('src.tools.api._cache.get_financial_metrics')
    @patch('src.tools.api.akshare.stock_financial_analysis_indicator')
    def test_get_financial_metrics_non_chinese_ticker_fallback(self, mock_akshare_indicator, mock_cache_get, mock_cache_set, mock_requests_get):
        mock_cache_get.return_value = None
        mock_response = MagicMock()
        # Using a simplified FinancialMetrics model for fallback test for brevity
        sample_fm_data = FinancialMetrics(ticker="AAPL.US", report_period="2023-12-31", period="annual", currency="USD", market_cap=3.0e12, price_to_earnings_ratio=30.0, enterprise_value=3.1e12, price_to_book_ratio=40.0, price_to_sales_ratio=8.0, enterprise_value_to_ebitda_ratio=25.0, enterprise_value_to_revenue_ratio=7.0, free_cash_flow_yield=0.03, peg_ratio=2.0, gross_margin=0.45, operating_margin=0.30, net_margin=0.25, return_on_equity=0.50, return_on_assets=0.20, return_on_invested_capital=0.28, asset_turnover=0.7, inventory_turnover=30.0, receivables_turnover=10.0, days_sales_outstanding=36.5, operating_cycle=48.5, working_capital_turnover=5.0, current_ratio=1.5, quick_ratio=1.0, cash_ratio=0.5, operating_cash_flow_ratio=0.4, debt_to_equity=1.2, debt_to_assets=0.5, interest_coverage=15.0, revenue_growth=0.05, earnings_growth=0.10, book_value_growth=0.08, earnings_per_share_growth=0.09, free_cash_flow_growth=0.12, operating_income_growth=0.11, ebitda_growth=0.10, payout_ratio=0.20, earnings_per_share=6.0, book_value_per_share=35.0, free_cash_flow_per_share=5.0)
        sample_api_data = FinancialMetricsResponse(financial_metrics=[sample_fm_data]).model_dump()
        mock_response.json.return_value = sample_api_data
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        metrics = get_financial_metrics("AAPL.US", "2023-12-31", limit=1)
        mock_akshare_indicator.assert_not_called()
        mock_requests_get.assert_called_once()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].ticker, "AAPL.US")
        self.assertEqual(metrics[0].market_cap, 3.0e12)
        expected_cache_data = [m.model_dump() for m in metrics]
        mock_cache_set.assert_called_once_with("AAPL.US", expected_cache_data)

    @patch('src.tools.api.akshare.stock_financial_analysis_indicator')
    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_financial_metrics')
    @patch('src.tools.api._cache.get_financial_metrics')
    def test_get_financial_metrics_cache_hit(self, mock_cache_get, mock_cache_set, mock_requests_get, mock_akshare_indicator):
        cached_metrics_data = [FinancialMetrics(ticker="MSFT.US", report_period="2023-06-30", period="annual", currency="USD", market_cap=2.5e12, price_to_earnings_ratio=35.0, enterprise_value=2.4e12, price_to_book_ratio=12.0, price_to_sales_ratio=11.0, enterprise_value_to_ebitda_ratio=22.0, enterprise_value_to_revenue_ratio=10.0, free_cash_flow_yield=0.025, peg_ratio=2.2, gross_margin=0.68, operating_margin=0.40, net_margin=0.33, return_on_equity=0.40, return_on_assets=0.18, return_on_invested_capital=0.25, asset_turnover=0.6, inventory_turnover=20.0, receivables_turnover=8.0, days_sales_outstanding=45.0, operating_cycle=60.0, working_capital_turnover=4.0, current_ratio=2.0, quick_ratio=1.5, cash_ratio=1.0, operating_cash_flow_ratio=0.35, debt_to_equity=0.5, debt_to_assets=0.3, interest_coverage=20.0, revenue_growth=0.15, earnings_growth=0.18, book_value_growth=0.12, earnings_per_share_growth=0.16, free_cash_flow_growth=0.20, operating_income_growth=0.17, ebitda_growth=0.16, payout_ratio=0.25, earnings_per_share=10.0, book_value_per_share=80.0, free_cash_flow_per_share=8.0).model_dump()]
        mock_cache_get.return_value = cached_metrics_data
        
        metrics = get_financial_metrics("MSFT.US", "2023-06-30", limit=1)
        mock_akshare_indicator.assert_not_called()
        mock_requests_get.assert_not_called()
        mock_cache_set.assert_not_called()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].report_period, "2023-06-30")

    @patch('src.tools.api.akshare.stock_financial_analysis_indicator')
    @patch('src.tools.api._cache.set_financial_metrics')
    @patch('src.tools.api._cache.get_financial_metrics')
    def test_get_financial_metrics_chinese_ticker_date_filtering(self, mock_cache_get, mock_cache_set, mock_akshare_indicator):
        mock_cache_get.return_value = None
        sample_akshare_data = {
            '日期': ["2024-01-15", "2023-12-31", "2023-09-30"],
            '总市值': [2.1e+12, 2.0e+12, 1.9e+12], '市盈率(PE)': [31.0, 30.0, 29.0],
            '净资产收益率(ROE)': [19.0, 18.0, 17.0]
            # Other fields will be None due to current placeholder logic if not in column_mapping or explicitly set here
        }
        mock_df = pd.DataFrame(sample_akshare_data)
        mock_akshare_indicator.return_value = mock_df

        ticker = "600519.SH"
        end_date = "2023-12-31" 
        metrics = get_financial_metrics(ticker, end_date, limit=5)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].report_period, "2023-12-31")
        self.assertEqual(metrics[1].report_period, "2023-09-30")


class TestGetCompanyNews(unittest.TestCase):

    @patch('src.tools.api.akshare.stock_news_em')
    @patch('src.tools.api._cache.set_company_news')
    @patch('src.tools.api._cache.get_company_news')
    def test_get_company_news_chinese_ticker_akshare_success(self, mock_cache_get, mock_cache_set, mock_akshare_news):
        mock_cache_get.return_value = None
        sample_akshare_data = {
            'code': ["600519", "600519"], 'title': ["News Title 1", "News Title 2"],
            'content': ["Content 1...", "Content 2..."], 
            'public_time': [datetime.datetime(2023, 1, 16, 11, 0, 0), datetime.datetime(2023, 1, 15, 10, 30, 0)], # Note: order for sorting test
            'source': ["Source B", "Source A"], 'url': ["http://example.com/news2", "http://example.com/news1"]
        }
        mock_df = pd.DataFrame(sample_akshare_data)
        mock_akshare_news.return_value = mock_df

        ticker = "600519.SH"
        # Dates in YYYY-MM-DDTHH:MM:SS format for comparison with CompanyNews.date from fallback API
        end_date = "2023-01-20T00:00:00" 
        start_date = "2023-01-15T00:00:00"
        
        news_list = get_company_news(ticker, end_date, start_date=start_date, limit=5)

        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_news.assert_called_once_with(stock="600519")
        
        self.assertEqual(len(news_list), 2)
        # Data is sorted by date descending in the function
        self.assertEqual(news_list[0].title, "News Title 1") 
        self.assertEqual(news_list[0].date, "2023-01-16 11:00:00") # Akshare date format
        self.assertEqual(news_list[0].source, "Source B")
        self.assertEqual(news_list[1].title, "News Title 2") 
        self.assertEqual(news_list[1].date, "2023-01-15 10:30:00")
        self.assertEqual(news_list[1].author, "N/A") # Default value

        expected_cache_data = [n.model_dump() for n in news_list]
        mock_cache_set.assert_called_once_with(ticker, expected_cache_data)

    @patch('src.tools.api.akshare.stock_news_em')
    @patch('src.tools.api._cache.set_company_news')
    @patch('src.tools.api._cache.get_company_news')
    def test_get_company_news_chinese_ticker_akshare_error(self, mock_cache_get, mock_cache_set, mock_akshare_news):
        mock_cache_get.return_value = None
        mock_akshare_news.side_effect = Exception("AKShare API error")
        news_list = get_company_news("600519.SH", "2023-01-20T00:00:00")
        self.assertEqual(news_list, [])
        mock_cache_set.assert_not_called()

    @patch('src.tools.api.akshare.stock_news_em')
    @patch('src.tools.api._cache.set_company_news')
    @patch('src.tools.api._cache.get_company_news')
    def test_get_company_news_chinese_ticker_akshare_empty(self, mock_cache_get, mock_cache_set, mock_akshare_news):
        mock_cache_get.return_value = None
        mock_akshare_news.return_value = pd.DataFrame()
        news_list = get_company_news("600519.SH", "2023-01-20T00:00:00")
        self.assertEqual(news_list, [])
        mock_cache_set.assert_not_called()

    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_company_news')
    @patch('src.tools.api._cache.get_company_news')
    @patch('src.tools.api.akshare.stock_news_em')
    def test_get_company_news_non_chinese_ticker_fallback(self, mock_akshare_news, mock_cache_get, mock_cache_set, mock_requests_get):
        mock_cache_get.return_value = None
        mock_response = MagicMock()
        # Fallback API returns dates in ISO format e.g. "2023-01-15T10:00:00Z"
        sample_api_data = CompanyNewsResponse(news=[CompanyNews(ticker="AAPL.US", title="Apple News 1", author="Reporter X", source="Tech News", date="2023-01-15T10:00:00Z", url="http://example.com/aapl1", sentiment="positive")]).model_dump()
        mock_response.json.return_value = sample_api_data
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        news_list = get_company_news("AAPL.US", "2023-01-20T00:00:00", start_date="2023-01-15T00:00:00")
        
        mock_akshare_news.assert_not_called()
        mock_requests_get.assert_called_once() 
        self.assertEqual(len(news_list), 1)
        self.assertEqual(news_list[0].title, "Apple News 1")
        self.assertEqual(news_list[0].date, "2023-01-15T10:00:00Z") # Check date format from fallback
        
        expected_cache_data = [CompanyNews(**n).model_dump() for n in sample_api_data['news']]
        mock_cache_set.assert_called_once_with("AAPL.US", expected_cache_data)

    @patch('src.tools.api.akshare.stock_news_em')
    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_company_news')
    @patch('src.tools.api._cache.get_company_news')
    def test_get_company_news_cache_hit(self, mock_cache_get, mock_cache_set, mock_requests_get, mock_akshare_news):
        cached_news_data = [ CompanyNews(ticker="MSFT.US", title="MSFT News Cached", author="N/A", source="Cache", date="2023-01-18T12:00:00Z", url="http://cached.com", sentiment=None).model_dump() ]
        mock_cache_get.return_value = cached_news_data
        
        # Dates here are for filtering the cached data
        news_list = get_company_news("MSFT.US", "2023-01-20T00:00:00", start_date="2023-01-15T00:00:00")
        
        mock_akshare_news.assert_not_called()
        mock_requests_get.assert_not_called()
        mock_cache_set.assert_not_called()
        self.assertEqual(len(news_list), 1)
        self.assertEqual(news_list[0].title, "MSFT News Cached")
        self.assertEqual(news_list[0].date, "2023-01-18T12:00:00Z") # Preserves original date format


class TestGetInsiderTrades(unittest.TestCase):

    @patch('src.tools.api.akshare.stock_ggcg_em')
    @patch('src.tools.api._cache.set_insider_trades')
    @patch('src.tools.api._cache.get_insider_trades')
    def test_get_insider_trades_chinese_ticker_akshare_success(self, mock_cache_get, mock_cache_set, mock_akshare_ggcg):
        mock_cache_get.return_value = None
        sample_akshare_data = {
            '公告日期': [datetime.date(2023, 1, 10), datetime.date(2023, 1, 5)],
            '变动日期': [datetime.date(2023, 1, 8), datetime.date(2023, 1, 3)], # transaction_date
            '变动人': ["John Doe", "Jane Smith"], # name
            '变动股份数量': [10000.0, -5000.0], # transaction_shares (positive for buy, negative for sell)
            '交易平均价': [10.50, 10.20], # transaction_price_per_share
            '变动后持股总数': [50000.0, 20000.0], # shares_owned_after_transaction
            '董监高职务': ["Director", "VP"], # title
            '上市公司简称': ["公司A", "公司A"] # issuer
        }
        mock_df = pd.DataFrame(sample_akshare_data)
        mock_akshare_ggcg.return_value = mock_df

        ticker = "000001.SZ"
        end_date = "2023-01-15T00:00:00"
        start_date = "2023-01-01T00:00:00"
        
        trades = get_insider_trades(ticker, end_date, start_date=start_date, limit=5)

        mock_cache_get.assert_called_once_with(ticker)
        mock_akshare_ggcg.assert_called_once_with(symbol="000001")
        
        self.assertEqual(len(trades), 2)
        
        trade1 = trades[0] # Sorted by filing_date (公告日期) descending
        self.assertEqual(trade1.ticker, ticker)
        self.assertEqual(trade1.name, "John Doe")
        self.assertEqual(trade1.title, "Director")
        self.assertEqual(trade1.transaction_date, "2023-01-08")
        self.assertEqual(trade1.filing_date, "2023-01-10")
        self.assertEqual(trade1.transaction_shares, 10000.0)
        self.assertEqual(trade1.transaction_price_per_share, 10.50)
        self.assertEqual(trade1.transaction_value, 10000.0 * 10.50)
        self.assertEqual(trade1.shares_owned_after_transaction, 50000.0)
        self.assertEqual(trade1.shares_owned_before_transaction, 40000.0) # 50000 - 10000
        self.assertEqual(trade1.issuer, "公司A")

        trade2 = trades[1]
        self.assertEqual(trade2.name, "Jane Smith")
        self.assertEqual(trade2.transaction_shares, -5000.0)
        self.assertEqual(trade2.shares_owned_after_transaction, 20000.0)
        self.assertEqual(trade2.shares_owned_before_transaction, 25000.0) # 20000 - (-5000)

        expected_cache_data = [t.model_dump() for t in trades]
        mock_cache_set.assert_called_once_with(ticker, expected_cache_data)

    @patch('src.tools.api.akshare.stock_ggcg_em')
    @patch('src.tools.api._cache.set_insider_trades')
    @patch('src.tools.api._cache.get_insider_trades')
    def test_get_insider_trades_chinese_ticker_akshare_error(self, mock_cache_get, mock_cache_set, mock_akshare_ggcg):
        mock_cache_get.return_value = None
        mock_akshare_ggcg.side_effect = Exception("AKShare API error")
        trades = get_insider_trades("000001.SZ", "2023-01-15T00:00:00")
        self.assertEqual(trades, [])
        mock_cache_set.assert_called_once_with("000001.SZ", []) # Caches empty list on error

    @patch('src.tools.api.akshare.stock_ggcg_em')
    @patch('src.tools.api._cache.set_insider_trades')
    @patch('src.tools.api._cache.get_insider_trades')
    def test_get_insider_trades_chinese_ticker_akshare_empty(self, mock_cache_get, mock_cache_set, mock_akshare_ggcg):
        mock_cache_get.return_value = None
        mock_akshare_ggcg.return_value = pd.DataFrame()
        trades = get_insider_trades("000001.SZ", "2023-01-15T00:00:00")
        self.assertEqual(trades, [])
        mock_cache_set.assert_called_once_with("000001.SZ", []) # Caches empty list

    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_insider_trades')
    @patch('src.tools.api._cache.get_insider_trades')
    @patch('src.tools.api.akshare.stock_ggcg_em')
    def test_get_insider_trades_non_chinese_ticker_fallback(self, mock_akshare_ggcg, mock_cache_get, mock_cache_set, mock_requests_get):
        mock_cache_get.return_value = None
        mock_response = MagicMock()
        sample_api_data = InsiderTradeResponse(insider_trades=[
            InsiderTrade(ticker="MSFT.US", name="Satya Nadella", title="CEO", transaction_date="2023-01-10T00:00:00Z", filing_date="2023-01-12T00:00:00Z", transaction_shares=-1000, transaction_price_per_share=250.0, shares_owned_after_transaction=100000, security_title="Common Stock")
        ]).model_dump()
        mock_response.json.return_value = sample_api_data
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        trades = get_insider_trades("MSFT.US", "2023-01-15T00:00:00", start_date="2023-01-01T00:00:00")
        
        mock_akshare_ggcg.assert_not_called()
        mock_requests_get.assert_called_once()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].name, "Satya Nadella")
        
        expected_cache_data = [InsiderTrade(**t).model_dump() for t in sample_api_data['insider_trades']]
        mock_cache_set.assert_called_once_with("MSFT.US", expected_cache_data)

    @patch('src.tools.api.akshare.stock_ggcg_em')
    @patch('src.tools.api.requests.get')
    @patch('src.tools.api._cache.set_insider_trades')
    @patch('src.tools.api._cache.get_insider_trades')
    def test_get_insider_trades_cache_hit(self, mock_cache_get, mock_cache_set, mock_requests_get, mock_akshare_ggcg):
        cached_trades_data = [
            InsiderTrade(ticker="GOOG.US", name="Sundar Pichai", title="CEO", transaction_date="2023-01-05T00:00:00Z", filing_date="2023-01-07T00:00:00Z", transaction_shares=100, transaction_price_per_share=100.0, shares_owned_after_transaction=5000, security_title="Class C Stock").model_dump(),
            InsiderTrade(ticker="GOOG.US", name="Ruth Porat", title="CFO", transaction_date="2023-01-03T00:00:00Z", filing_date="2023-01-04T00:00:00Z", transaction_shares=50, transaction_price_per_share=98.0, shares_owned_after_transaction=2000, security_title="Class C Stock").model_dump()
        ]
        mock_cache_get.return_value = cached_trades_data
        
        trades = get_insider_trades("GOOG.US", "2023-01-06T00:00:00", start_date="2023-01-01T00:00:00", limit=1) # end_date filters one, limit filters another
        
        mock_akshare_ggcg.assert_not_called()
        mock_requests_get.assert_not_called()
        mock_cache_set.assert_not_called()
        
        self.assertEqual(len(trades), 1) # Only one trade should match date and limit
        self.assertEqual(trades[0].name, "Ruth Porat")
        self.assertEqual(trades[0].filing_date, "2023-01-04T00:00:00Z")


class TestSearchLineItems(unittest.TestCase):

    @patch('src.tools.api.akshare.stock_financial_report_sina')
    # No caching is implemented for search_line_items in the current version of api.py
    def test_search_line_items_chinese_ticker_akshare_success(self, mock_akshare_report):
        # Prepare mock data for Balance Sheet, Income Statement, Cash Flow
        balance_sheet_data = {
            '资产负债表': ['货币资金', '应收账款', '存货', '总资产'], # Using a more realistic item name for index
            '2023-12-31': [100.0, 200.0, 150.0, 1000.0],
            '2023-09-30': [90.0, 190.0, 140.0, 950.0]
        }
        # Set the first column as index
        df_balance_sheet = pd.DataFrame(balance_sheet_data).set_index('资产负债表')

        income_statement_data = {
            '利润表': ['营业总收入', '营业成本', '净利润'],
            '2023-12-31': [1000.0, 600.0, 200.0],
            '2023-09-30': [950.0, 550.0, 180.0]
        }
        df_income_statement = pd.DataFrame(income_statement_data).set_index('利润表')
        
        cash_flow_data = { # Empty for this test to ensure it's handled
            '现金流量表': [], '2023-12-31': [], '2023-09-30': []
        }
        df_cash_flow = pd.DataFrame(cash_flow_data).set_index('现金流量表')

        # Mock akshare to return different DFs based on the 'symbol' (report name)
        def mock_report_side_effect(stock, symbol):
            if symbol == "资产负债表":
                return df_balance_sheet
            elif symbol == "利润表":
                return df_income_statement
            elif symbol == "现金流量表":
                return df_cash_flow
            return pd.DataFrame() # Should not happen with current mapping
        
        mock_akshare_report.side_effect = mock_report_side_effect

        ticker = "600519.SH"
        line_items_to_search = ["货币资金", "净利润", "NonExistentItem"]
        end_date = "2023-12-31"
        
        results = search_line_items(ticker, line_items_to_search, end_date, limit=10)

        self.assertEqual(mock_akshare_report.call_count, 3) # Called for each report type
        mock_akshare_report.assert_any_call(stock="600519", symbol="资产负债表")
        mock_akshare_report.assert_any_call(stock="600519", symbol="利润表")
        mock_akshare_report.assert_any_call(stock="600519", symbol="现金流量表")

        self.assertEqual(len(results), 4) # 2 for 货币资金, 2 for 净利润
        
        # Results are sorted by report_period descending across all found items
        # Check a few items
        # Example: 货币资金 for 2023-12-31
        item1 = next(r for r in results if r.report_period == "2023-12-31" and hasattr(r, "货币资金"))
        self.assertEqual(item1.ticker, ticker)
        self.assertEqual(item1.currency, "CNY")
        self.assertEqual(item1.period, "annual")
        self.assertEqual(getattr(item1, "货币资金"), 100.0)

        # Example: 净利润 for 2023-12-31
        item2 = next(r for r in results if r.report_period == "2023-12-31" and hasattr(r, "净利润"))
        self.assertEqual(getattr(item2, "净利润"), 200.0)

        # Example: 货币资金 for 2023-09-30
        item3 = next(r for r in results if r.report_period == "2023-09-30" and hasattr(r, "货币资金"))
        self.assertEqual(getattr(item3, "货币资金"), 90.0)
        self.assertEqual(item3.period, "quarterly")

    @patch('src.tools.api.akshare.stock_financial_report_sina')
    def test_search_line_items_chinese_ticker_akshare_error(self, mock_akshare_report):
        mock_akshare_report.side_effect = Exception("AKShare API error")
        results = search_line_items("600519.SH", ["营收"], "2023-12-31")
        self.assertEqual(results, [])

    @patch('src.tools.api.akshare.stock_financial_report_sina')
    def test_search_line_items_chinese_ticker_akshare_empty_reports(self, mock_akshare_report):
        mock_akshare_report.return_value = pd.DataFrame() # All reports are empty
        results = search_line_items("600519.SH", ["营收"], "2023-12-31")
        self.assertEqual(results, [])

    @patch('src.tools.api.requests.post')
    @patch('src.tools.api.akshare.stock_financial_report_sina')
    def test_search_line_items_non_chinese_ticker_fallback(self, mock_akshare_report, mock_requests_post):
        mock_response = MagicMock()
        # Sample data for Financial Datasets API (already a list of LineItem-like dicts)
        sample_api_data_items = [
            {"ticker": "AAPL.US", "report_period": "2023-12-31", "period": "annual", "currency": "USD", "revenue": 3.8e11, "netIncome": 9.0e10},
        ]
        # The API returns LineItemResponse which has a 'search_results' field
        mock_response.json.return_value = LineItemResponse(search_results=[LineItem(**item) for item in sample_api_data_items]).model_dump()
        mock_response.status_code = 200
        mock_requests_post.return_value = mock_response

        results = search_line_items("AAPL.US", ["revenue", "netIncome"], "2023-12-31", limit=1)
        
        mock_akshare_report.assert_not_called()
        mock_requests_post.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertTrue(hasattr(results[0], "revenue"))
        self.assertEqual(results[0].revenue, 3.8e11)

    # No cache tests for search_line_items as it's not cached in the current implementation


class TestGetMarketCap(unittest.TestCase):

    @patch('src.tools.api.akshare.stock_individual_info_em')
    @patch('src.tools.api.get_financial_metrics') # Mocking the already tested get_financial_metrics
    def test_get_market_cap_chinese_ticker_today_akshare_success(self, mock_get_financial_metrics, mock_akshare_info):
        # Test fetching market cap for a Chinese ticker on today's date, akshare success
        ticker = "600519.SH"
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Mock akshare.stock_individual_info_em for "总市值"
        # DataFrame structure: Column 0 is item name, Column 1 is value
        mock_info_df = pd.DataFrame({
            'item': ['总市值', '流通市值', '市盈率'],
            'value': ['2.15万亿', '2.10万亿', '30.5'] 
        })
        # Rename columns to match how they are accessed in get_market_cap
        mock_info_df.columns = ['指标', '数值'] # Example column names, adjust if known
        mock_akshare_info.return_value = mock_info_df
        
        market_cap = get_market_cap(ticker, today_str)

        mock_akshare_info.assert_called_once_with(stock="600519")
        mock_get_financial_metrics.assert_not_called() # Should not fall back
        self.assertAlmostEqual(market_cap, 2.15 * 10**12) # 2.15万亿

    @patch('src.tools.api.akshare.stock_individual_info_em')
    @patch('src.tools.api.get_financial_metrics')
    def test_get_market_cap_chinese_ticker_today_akshare_parse_error(self, mock_get_financial_metrics, mock_akshare_info):
        # Test Chinese ticker, today, akshare returns unparsable market cap, fallback to get_financial_metrics
        ticker = "600519.SH"
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        mock_info_df = pd.DataFrame({'指标': ['总市值'], '数值': ['InvalidData']})
        mock_akshare_info.return_value = mock_info_df
        
        # Mock fallback get_financial_metrics
        mock_get_financial_metrics.return_value = [FinancialMetrics(ticker=ticker, report_period=today_str, period="annual", currency="CNY", market_cap=2.0e12)]
        
        market_cap = get_market_cap(ticker, today_str)

        mock_akshare_info.assert_called_once_with(stock="600519")
        mock_get_financial_metrics.assert_called_once_with(ticker, today_str, period="ttm", limit=1)
        self.assertAlmostEqual(market_cap, 2.0e12)

    @patch('src.tools.api.akshare.stock_individual_info_em')
    @patch('src.tools.api.get_financial_metrics')
    def test_get_market_cap_chinese_ticker_today_akshare_api_error(self, mock_get_financial_metrics, mock_akshare_info):
        # Test Chinese ticker, today, akshare API error, fallback to get_financial_metrics
        ticker = "600519.SH"
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        mock_akshare_info.side_effect = Exception("AKShare API error")
        mock_get_financial_metrics.return_value = [FinancialMetrics(ticker=ticker, report_period=today_str, period="annual", currency="CNY", market_cap=1.9e12)]
        
        market_cap = get_market_cap(ticker, today_str)

        mock_akshare_info.assert_called_once_with(stock="600519")
        mock_get_financial_metrics.assert_called_once_with(ticker, today_str, period="ttm", limit=1)
        self.assertAlmostEqual(market_cap, 1.9e12)

    @patch('src.tools.api.akshare.stock_individual_info_em')
    @patch('src.tools.api.get_financial_metrics')
    def test_get_market_cap_chinese_ticker_historical_date(self, mock_get_financial_metrics, mock_akshare_info):
        # Test Chinese ticker, historical date, should use get_financial_metrics
        ticker = "600519.SH"
        historical_date = "2023-06-30" # Not today
        
        mock_get_financial_metrics.return_value = [FinancialMetrics(ticker=ticker, report_period=historical_date, period="annual", currency="CNY", market_cap=1.8e12)]
        
        market_cap = get_market_cap(ticker, historical_date)

        mock_akshare_info.assert_not_called() # Should not be called for historical
        mock_get_financial_metrics.assert_called_once_with(ticker, historical_date, period="ttm", limit=1)
        self.assertAlmostEqual(market_cap, 1.8e12)

    @patch('src.tools.api.akshare.stock_individual_info_em')
    @patch('src.tools.api.get_financial_metrics')
    def test_get_market_cap_chinese_ticker_fallback_returns_none(self, mock_get_financial_metrics, mock_akshare_info):
        # Test Chinese ticker, fallback to get_financial_metrics, which returns no data
        ticker = "600519.SH"
        historical_date = "2023-06-30"
        
        mock_get_financial_metrics.return_value = [] # No metrics found
        
        market_cap = get_market_cap(ticker, historical_date)
        
        mock_get_financial_metrics.assert_called_once_with(ticker, historical_date, period="ttm", limit=1)
        self.assertIsNone(market_cap)

    @patch('src.tools.api.requests.get')
    @patch('src.tools.api.get_financial_metrics') # Also mock this for the non-Chinese fallback's own fallback
    def test_get_market_cap_non_chinese_ticker_today_api_success(self, mock_get_financial_metrics_fallback, mock_requests_get):
        # Test non-Chinese ticker, today, Financial Datasets API success
        ticker = "AAPL.US"
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        mock_response = MagicMock()
        # CompanyFactsResponse model has company_facts: CompanyFacts. CompanyFacts has market_cap: float | None
        sample_api_data = {"company_facts": {"ticker": "AAPL.US", "name": "Apple Inc.", "market_cap": 3.1e12}} # Other fields omitted
        mock_response.json.return_value = sample_api_data
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        
        market_cap = get_market_cap(ticker, today_str)

        mock_requests_get.assert_called_once()
        self.assertTrue(f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}" in mock_requests_get.call_args[0][0])
        mock_get_financial_metrics_fallback.assert_not_called()
        self.assertAlmostEqual(market_cap, 3.1e12)

    @patch('src.tools.api.requests.get')
    @patch('src.tools.api.get_financial_metrics')
    def test_get_market_cap_non_chinese_ticker_today_api_error_fallback_success(self, mock_get_financial_metrics, mock_requests_get):
        # Test non-Chinese ticker, today, Financial Datasets API error, fallback to get_financial_metrics success
        ticker = "AAPL.US"
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        mock_requests_get.side_effect = Exception("API error") # Or status_code != 200
        mock_get_financial_metrics.return_value = [FinancialMetrics(ticker=ticker, report_period=today_str, period="annual", currency="USD", market_cap=3.0e12)]

        market_cap = get_market_cap(ticker, today_str)

        mock_requests_get.assert_called_once()
        mock_get_financial_metrics.assert_called_once_with(ticker, today_str, period="ttm", limit=1)
        self.assertAlmostEqual(market_cap, 3.0e12)

    @patch('src.tools.api.get_financial_metrics')
    def test_get_market_cap_non_chinese_ticker_historical_date(self, mock_get_financial_metrics):
        # Test non-Chinese ticker, historical date
        ticker = "AAPL.US"
        historical_date = "2023-06-30"
        
        mock_get_financial_metrics.return_value = [FinancialMetrics(ticker=ticker, report_period=historical_date, period="annual", currency="USD", market_cap=2.8e12)]
        
        market_cap = get_market_cap(ticker, historical_date)
        
        mock_get_financial_metrics.assert_called_once_with(ticker, historical_date, period="ttm", limit=1)
        self.assertAlmostEqual(market_cap, 2.8e12)

    # No specific cache tests for get_market_cap as it relies on get_financial_metrics for caching historical,
    # and current day data is usually not cached or has short TTL. The caching of get_financial_metrics is tested elsewhere.
