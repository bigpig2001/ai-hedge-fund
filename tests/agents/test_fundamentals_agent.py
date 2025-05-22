import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.fundamentals import fundamentals_agent
from src.graph.state import AgentState # AgentState might need to be simplified or mocked if too complex
from src.data.models import FinancialMetrics

# A helper to create a simplified AgentState for testing
def create_test_agent_state(tickers, end_date="2024-01-01", start_date="2023-10-01"):
    return {
        "messages": [],
        "data": {
            "tickers": tickers,
            "portfolio": {}, # Assuming not directly used by fundamentals_agent
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": False, # Keep false to avoid printing during tests
            "model_name": "test_model",
            "model_provider": "test_provider",
        },
    }

class TestFundamentalsAgent(unittest.TestCase):

    def _create_mock_financial_metrics(self, ticker="TEST.US", roe=0.16, net_margin=0.21, op_margin=0.16, rev_growth=0.11, earn_growth=0.11, bv_growth=0.11, current_ratio=2.0, de_ratio=0.4, fcf_ps=5.0, eps=4.0, pe_ratio=15.0, pb_ratio=2.0, ps_ratio=1.0, market_cap=1e9, currency="USD", period="ttm", report_period="2023-12-31"):
        return FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period=period,
            currency=currency,
            market_cap=market_cap,
            enterprise_value=None,
            price_to_earnings_ratio=pe_ratio,
            price_to_book_ratio=pb_ratio,
            price_to_sales_ratio=ps_ratio,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=None, # Assuming these are not directly used by current agent logic score, or need more detailed mock
            operating_margin=op_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=None,
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=current_ratio,
            quick_ratio=None,
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=de_ratio,
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=rev_growth,
            earnings_growth=earn_growth,
            book_value_growth=bv_growth,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=None,
            earnings_per_share=eps,
            book_value_per_share=None,
            free_cash_flow_per_share=fcf_ps,
        )

    @patch('src.agents.fundamentals.get_financial_metrics_akshare')
    @patch('src.agents.fundamentals.get_financial_metrics')
    def test_fundamentals_agent_chinese_stock_success(self, mock_get_fm, mock_get_fm_akshare):
        chinese_ticker = "600519.SS"
        initial_state = create_test_agent_state(tickers=[chinese_ticker])
        
        # Mock akshare to return data
        mock_metrics_data = self._create_mock_financial_metrics(ticker=chinese_ticker, currency="CNY", roe=0.20, net_margin=0.25, rev_growth=0.15) # Strong metrics
        mock_get_fm_akshare.return_value = [mock_metrics_data]

        # Call the agent
        result_state = fundamentals_agent(initial_state)

        # Assertions
        mock_get_fm_akshare.assert_called_once_with(ticker=chinese_ticker, end_date=initial_state["data"]["end_date"])
        mock_get_fm.assert_not_called()

        self.assertIn("fundamentals_agent", result_state["data"]["analyst_signals"])
        analysis = result_state["data"]["analyst_signals"]["fundamentals_agent"].get(chinese_ticker)
        self.assertIsNotNone(analysis)
        
        # Based on _create_mock_financial_metrics default strong values + overrides:
        # Profitability: roe(0.20)>0.15 (T), net_margin(0.25)>0.20 (T), op_margin(0.16)>0.15 (T) -> Score 3 -> bullish
        # Growth: rev_growth(0.15)>0.10 (T), earn_growth(0.11)>0.10 (T), bv_growth(0.11)>0.10 (T) -> Score 3 -> bullish
        # Health: current_ratio(2.0)>1.5 (T), de_ratio(0.4)<0.5 (T), fcf_ps(5)>eps(4)*0.8 (T) -> Score 3 -> bullish
        # Price Ratios: pe(15)<25 (T), pb(2)<3 (T), ps(1)<5 (T) -> Score 0 -> bullish (lower is better for price ratios)
        # Overall: 4 bullish -> bullish
        self.assertEqual(analysis["signal"], "bullish")
        self.assertEqual(analysis["confidence"], 100.0) # 4 bullish / 4 total signals
        self.assertIn("profitability_signal", analysis["reasoning"])
        self.assertEqual(analysis["reasoning"]["profitability_signal"]["signal"], "bullish")

    @patch('src.agents.fundamentals.get_financial_metrics_akshare')
    @patch('src.agents.fundamentals.get_financial_metrics')
    def test_fundamentals_agent_non_chinese_stock_success(self, mock_get_fm, mock_get_fm_akshare):
        us_ticker = "AAPL.US"
        initial_state = create_test_agent_state(tickers=[us_ticker])
        
        mock_metrics_data = self._create_mock_financial_metrics(ticker=us_ticker, currency="USD", roe=0.10) # Less strong metrics
        mock_get_fm.return_value = [mock_metrics_data]

        result_state = fundamentals_agent(initial_state)

        mock_get_fm.assert_called_once_with(ticker=us_ticker, end_date=initial_state["data"]["end_date"], period="ttm", limit=10)
        mock_get_fm_akshare.assert_not_called()

        self.assertIn("fundamentals_agent", result_state["data"]["analyst_signals"])
        analysis = result_state["data"]["analyst_signals"]["fundamentals_agent"].get(us_ticker)
        self.assertIsNotNone(analysis)
        # Profitability: roe(0.10)<=0.15 (F), net_margin(0.21)>0.20 (T), op_margin(0.16)>0.15 (T) -> Score 2 -> bullish
        # Growth: rev_growth(0.11)>0.10 (T), earn_growth(0.11)>0.10 (T), bv_growth(0.11)>0.10 (T) -> Score 3 -> bullish
        # Health: current_ratio(2.0)>1.5 (T), de_ratio(0.4)<0.5 (T), fcf_ps(5)>eps(4)*0.8 (T) -> Score 3 -> bullish
        # Price Ratios: pe(15)<25 (T), pb(2)<3 (T), ps(1)<5 (T) -> Score 0 -> bullish
        # Overall: 4 bullish -> bullish
        self.assertEqual(analysis["signal"], "bullish")
        self.assertEqual(analysis["confidence"], 100.0)


    @patch('src.agents.fundamentals.get_financial_metrics_akshare')
    @patch('src.agents.fundamentals.get_financial_metrics')
    def test_fundamentals_agent_chinese_stock_data_not_found(self, mock_get_fm, mock_get_fm_akshare):
        chinese_ticker_no_data = "000002.SZ"
        initial_state = create_test_agent_state(tickers=[chinese_ticker_no_data])
        
        mock_get_fm_akshare.return_value = [] # Simulate no data found

        result_state = fundamentals_agent(initial_state)

        mock_get_fm_akshare.assert_called_once_with(ticker=chinese_ticker_no_data, end_date=initial_state["data"]["end_date"])
        mock_get_fm.assert_not_called()
        
        self.assertIn("fundamentals_agent", result_state["data"]["analyst_signals"])
        analysis = result_state["data"]["analyst_signals"]["fundamentals_agent"].get(chinese_ticker_no_data)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["signal"], "neutral")
        self.assertEqual(analysis["confidence"], 0)
        self.assertEqual(analysis["reasoning"], {"error": "No financial metrics found"})

    @patch('src.agents.fundamentals.get_financial_metrics_akshare')
    @patch('src.agents.fundamentals.get_financial_metrics')
    def test_fundamentals_agent_mixed_stocks(self, mock_get_fm, mock_get_fm_akshare):
        tickers = ["AAPL.US", "600519.SS", "MSFT.US", "000001.SZ"]
        initial_state = create_test_agent_state(tickers=tickers)

        # Mock returns for US stocks
        mock_aapl_metrics = self._create_mock_financial_metrics(ticker="AAPL.US", roe=0.30) # Strong AAPL
        mock_msft_metrics = self._create_mock_financial_metrics(ticker="MSFT.US", roe=0.05) # Weak MSFT (for profitability)
        
        # Mock returns for Chinese stocks
        mock_600519_metrics = self._create_mock_financial_metrics(ticker="600519.SS", currency="CNY", net_margin=0.05) # Weak Kweichow Moutai (for profitability)
        mock_000001_metrics = self._create_mock_financial_metrics(ticker="000001.SZ", currency="CNY", net_margin=0.25) # Strong Ping An Bank
        
        # Setup side_effect for multiple calls based on ticker
        def get_fm_side_effect(ticker, end_date, period, limit):
            if ticker == "AAPL.US": return [mock_aapl_metrics]
            if ticker == "MSFT.US": return [mock_msft_metrics]
            return []
        mock_get_fm.side_effect = get_fm_side_effect
        
        def get_fm_akshare_side_effect(ticker, end_date):
            if ticker == "600519.SS": return [mock_600519_metrics]
            if ticker == "000001.SZ": return [mock_000001_metrics]
            return []
        mock_get_fm_akshare.side_effect = get_fm_akshare_side_effect

        result_state = fundamentals_agent(initial_state)
        
        self.assertEqual(mock_get_fm.call_count, 2)
        self.assertEqual(mock_get_fm_akshare.call_count, 2)
        
        signals = result_state["data"]["analyst_signals"]["fundamentals_agent"]
        self.assertIn("AAPL.US", signals)
        self.assertIn("600519.SS", signals)
        self.assertIn("MSFT.US", signals)
        self.assertIn("000001.SZ", signals)

        # Basic check for signal existence, detailed checks can be done if specific logic is targeted
        self.assertIn(signals["AAPL.US"]["signal"], ["bullish", "bearish", "neutral"])
        self.assertIn(signals["600519.SS"]["signal"], ["bullish", "bearish", "neutral"])
        
        # Example: AAPL (strong roe) -> bullish profitability -> likely bullish overall
        self.assertEqual(signals["AAPL.US"]["reasoning"]["profitability_signal"]["signal"], "bullish")
        
        # Example: MSFT (weak roe=0.05, but net_margin=0.21, op_margin=0.16 -> 2/3 -> bullish profitability)
        self.assertEqual(signals["MSFT.US"]["reasoning"]["profitability_signal"]["signal"], "bullish")

        # Example: 600519.SS (weak net_margin=0.05, but roe=0.16, op_margin=0.16 -> 2/3 -> bullish profitability)
        self.assertEqual(signals["600519.SS"]["reasoning"]["profitability_signal"]["signal"], "bullish")

        # Example: 000001.SZ (strong net_margin=0.25, roe=0.16, op_margin=0.16 -> 3/3 -> bullish profitability)
        self.assertEqual(signals["000001.SZ"]["reasoning"]["profitability_signal"]["signal"], "bullish")


if __name__ == '__main__':
    unittest.main()
