import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from src.tools.api import (
    get_prices_akshare,
    get_financial_metrics_akshare,
    get_company_news_akshare
)
from src.data.models import Price, FinancialMetrics, CompanyNews

class TestAkshareApiFunctions(unittest.TestCase):

    @patch('src.tools.api.ak.stock_zh_a_hist')
    def test_get_prices_akshare_success(self, mock_ak_stock_zh_a_hist):
        # Prepare mock DataFrame
        mock_data = {
            '日期': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')],
            '开盘': [10.0, 10.5],
            '收盘': [10.2, 10.7],
            '最高': [10.3, 10.8],
            '最低': [9.9, 10.4],
            '成交量': [1000, 1500]
        }
        mock_df = pd.DataFrame(mock_data)
        mock_ak_stock_zh_a_hist.return_value = mock_df

        ticker = "sh600000"
        start_date = "2023-01-01"
        end_date = "2023-01-02"
        
        prices = get_prices_akshare(ticker, start_date, end_date)

        self.assertEqual(len(prices), 2)
        self.assertIsInstance(prices[0], Price)
        self.assertEqual(prices[0].open, 10.0)
        self.assertEqual(prices[0].close, 10.2)
        self.assertEqual(prices[0].high, 10.3)
        self.assertEqual(prices[0].low, 9.9)
        self.assertEqual(prices[0].volume, 1000)
        self.assertEqual(prices[0].time, "2023-01-01") # Check date format

        self.assertEqual(prices[1].open, 10.5)
        self.assertEqual(prices[1].time, "2023-01-02")
        
        mock_ak_stock_zh_a_hist.assert_called_once_with(
            symbol=ticker, 
            period="daily", 
            start_date="20230101", 
            end_date="20230102", 
            adjust=""
        )

    @patch('src.tools.api.ak.stock_zh_a_hist')
    def test_get_prices_akshare_empty_df(self, mock_ak_stock_zh_a_hist):
        mock_ak_stock_zh_a_hist.return_value = pd.DataFrame() # Empty DataFrame

        prices = get_prices_akshare("sh600000", "2023-01-01", "2023-01-02")
        self.assertEqual(len(prices), 0)

    @patch('src.tools.api.ak.stock_zh_a_hist')
    def test_get_prices_akshare_api_error(self, mock_ak_stock_zh_a_hist):
        mock_ak_stock_zh_a_hist.side_effect = Exception("AKShare API Error")

        prices = get_prices_akshare("sh600000", "2023-01-01", "2023-01-02")
        self.assertEqual(len(prices), 0)

    @patch('src.tools.api.ak.stock_financial_analysis_indicator')
    def test_get_financial_metrics_akshare_success(self, mock_ak_financial_indicator):
        # Prepare mock DataFrame
        mock_data = {
            '总市值': [1000000000.0],  # Market Cap
            '市盈率TTM': [15.5],       # P/E TTM
            '市净率': [1.2],           # P/B
            '销售毛利率': [0.60],     # Gross Margin
            '净资产收益率ROE': [0.12], # ROE
            # Add other relevant metrics that FinancialMetrics model expects
            # Ensure index is a date-like string for report_period
        }
        # ak.stock_financial_analysis_indicator often returns a DataFrame indexed by date (report period)
        mock_df = pd.DataFrame(mock_data, index=[pd.Timestamp('2023-12-31')]) 
        mock_ak_financial_indicator.return_value = mock_df

        ticker = "sh600000"
        end_date = "2024-01-15" # end_date is not directly used by akshare func but passed
        
        metrics_list = get_financial_metrics_akshare(ticker, end_date)

        self.assertEqual(len(metrics_list), 1)
        metrics = metrics_list[0]
        self.assertIsInstance(metrics, FinancialMetrics)
        self.assertEqual(metrics.ticker, ticker)
        self.assertEqual(metrics.report_period, "2023-12-31") # Check date format
        self.assertEqual(metrics.currency, "CNY")
        self.assertEqual(metrics.market_cap, 1000000000.0)
        self.assertEqual(metrics.price_to_earnings_ratio, 15.5)
        self.assertEqual(metrics.price_to_book_ratio, 1.2)
        self.assertEqual(metrics.gross_margin, 0.60)
        self.assertEqual(metrics.return_on_equity, 0.12)
        # Test a metric that might be None if not in mock_data
        self.assertIsNone(metrics.peg_ratio) 

        mock_ak_financial_indicator.assert_called_once_with(symbol=ticker)

    @patch('src.tools.api.ak.stock_financial_analysis_indicator')
    def test_get_financial_metrics_akshare_missing_metrics(self, mock_ak_financial_indicator):
        # Prepare mock DataFrame with only a few metrics
        mock_data = {
            '市盈率TTM': [15.5] 
            # Other metrics like '总市值', '市净率' are missing
        }
        mock_df = pd.DataFrame(mock_data, index=[pd.Timestamp('2023-12-31')])
        mock_ak_financial_indicator.return_value = mock_df

        metrics_list = get_financial_metrics_akshare("sh600000", "2024-01-15")
        
        self.assertEqual(len(metrics_list), 1)
        metrics = metrics_list[0]
        self.assertEqual(metrics.price_to_earnings_ratio, 15.5)
        self.assertIsNone(metrics.market_cap) # Should be None as it's missing
        self.assertIsNone(metrics.price_to_book_ratio) # Should be None

    @patch('src.tools.api.ak.stock_financial_analysis_indicator')
    def test_get_financial_metrics_akshare_empty_df(self, mock_ak_financial_indicator):
        mock_ak_financial_indicator.return_value = pd.DataFrame() # Empty DataFrame

        metrics_list = get_financial_metrics_akshare("sh600000", "2024-01-15")
        self.assertEqual(len(metrics_list), 0)

    @patch('src.tools.api.ak.stock_financial_analysis_indicator')
    def test_get_financial_metrics_akshare_api_error(self, mock_ak_financial_indicator):
        mock_ak_financial_indicator.side_effect = Exception("AKShare API Error")

        metrics_list = get_financial_metrics_akshare("sh600000", "2024-01-15")
        self.assertEqual(len(metrics_list), 0)

    @patch('src.tools.api.ak.stock_news_em')
    def test_get_company_news_akshare_success(self, mock_ak_stock_news_em):
        # Prepare mock DataFrame
        mock_data = {
            'code': ['sh600000', 'sh600000'],
            'title': ['News Title 1', 'News Title 2'],
            'content': ['Content 1', 'Content 2'], # Not directly used in CompanyNews model but part of typical response
            'public_time': [pd.Timestamp('2023-01-01 10:00:00'), pd.Timestamp('2023-01-02 11:00:00')],
            'source': ['Source A', 'Source B'],
            'article_url': ['http://example.com/news1', 'http://example.com/news2'] # Example URL field
        }
        mock_df = pd.DataFrame(mock_data)
        mock_ak_stock_news_em.return_value = mock_df

        ticker = "sh600000"
        limit = 5 # Test with limit
        
        news_list = get_company_news_akshare(ticker, limit=limit)

        self.assertEqual(len(news_list), 2) # Mock returns 2, which is less than limit
        self.assertIsInstance(news_list[0], CompanyNews)
        self.assertEqual(news_list[0].ticker, ticker)
        self.assertEqual(news_list[0].title, "News Title 1")
        self.assertEqual(news_list[0].author, "Source A") # 'source' is mapped to 'author'
        self.assertEqual(news_list[0].source, "Source A") # and also to 'source'
        self.assertEqual(news_list[0].date, "2023-01-01") # Check date format
        self.assertEqual(news_list[0].url, "http://example.com/news1")
        self.assertIsNone(news_list[0].sentiment)

        self.assertEqual(news_list[1].title, "News Title 2")
        self.assertEqual(news_list[1].date, "2023-01-02")
        
        mock_ak_stock_news_em.assert_called_once_with(stock=ticker)

    @patch('src.tools.api.ak.stock_news_em')
    def test_get_company_news_akshare_limit(self, mock_ak_stock_news_em):
        # Prepare mock DataFrame with more items than limit
        mock_data_list = []
        for i in range(10):
            mock_data_list.append({
                'code': 'sh600000',
                'title': f'News Title {i+1}',
                'public_time': pd.Timestamp(f'2023-01-01 10:0{i}:00'),
                'source': f'Source {i+1}',
                'article_url': f'http://example.com/news{i+1}'
            })
        mock_df = pd.DataFrame(mock_data_list)
        mock_ak_stock_news_em.return_value = mock_df

        ticker = "sh600000"
        limit = 3
        news_list = get_company_news_akshare(ticker, limit=limit)
        self.assertEqual(len(news_list), limit) # Should be truncated by the function's .head(limit)

    @patch('src.tools.api.ak.stock_news_em')
    def test_get_company_news_akshare_empty_df(self, mock_ak_stock_news_em):
        mock_ak_stock_news_em.return_value = pd.DataFrame() # Empty DataFrame

        news_list = get_company_news_akshare("sh600000", limit=5)
        self.assertEqual(len(news_list), 0)

    @patch('src.tools.api.ak.stock_news_em')
    def test_get_company_news_akshare_api_error(self, mock_ak_stock_news_em):
        mock_ak_stock_news_em.side_effect = Exception("AKShare API Error")

        news_list = get_company_news_akshare("sh600000", limit=5)
        self.assertEqual(len(news_list), 0)

if __name__ == '__main__':
    unittest.main()
