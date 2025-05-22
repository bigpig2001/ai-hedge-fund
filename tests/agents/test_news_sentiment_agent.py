import unittest
from unittest.mock import patch, MagicMock, call
import json
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.news_sentiment_agent import news_sentiment_agent, MAX_NEWS_ITEMS_TO_ANALYZE
from src.graph.state import AgentState # As before, might need simplification or mocking
from src.data.models import CompanyNews

# A helper to create a simplified AgentState for testing
def create_test_agent_state(tickers, end_date="2024-01-01", start_date="2023-10-01"):
    return {
        "messages": [],
        "data": {
            "tickers": tickers,
            "portfolio": {}, # Assuming not directly used by this agent
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

class TestNewsSentimentAgent(unittest.TestCase):

    def _create_mock_company_news(self, ticker, count=1, source_type="US"):
        news_list = []
        for i in range(count):
            news_list.append(
                CompanyNews(
                    ticker=ticker,
                    title=f"News {i+1} for {ticker} ({source_type})",
                    author=f"Author {i+1}",
                    source=f"Source {source_type}",
                    date=f"2024-01-0{i+1}",
                    url=f"http://example.com/{ticker}/news{i+1}",
                    sentiment=None # Sentiment is determined by LLM
                )
            )
        return news_list

    @patch('src.agents.news_sentiment_agent.generate_text_gemini')
    @patch('src.agents.news_sentiment_agent.get_company_news_akshare')
    @patch('src.agents.news_sentiment_agent.get_company_news')
    def test_news_sentiment_agent_chinese_stock_success(self, mock_get_news, mock_get_news_akshare, mock_gemini):
        chinese_ticker = "600519.SS"
        initial_state = create_test_agent_state(tickers=[chinese_ticker])
        
        mock_news = self._create_mock_company_news(chinese_ticker, count=3, source_type="AKShare")
        mock_get_news_akshare.return_value = mock_news
        
        # Mock Gemini responses: Positive, Negative, Neutral
        mock_gemini.side_effect = [
            "Sentiment: Positive. Reasoning: Strong earnings reported.",
            "Sentiment: Negative. Reasoning: Market share declining.",
            "Sentiment: Neutral. Reasoning: Standard operational update."
        ]

        result_state = news_sentiment_agent(initial_state)

        mock_get_news_akshare.assert_called_once_with(ticker=chinese_ticker, limit=MAX_NEWS_ITEMS_TO_ANALYZE)
        mock_get_news.assert_not_called()
        
        self.assertEqual(mock_gemini.call_count, 3)
        # Check a sample prompt
        expected_prompt_1 = f"Analyze the sentiment of the following news headline for company {chinese_ticker}: '{mock_news[0].title}'. Is the sentiment positive, negative, or neutral? Provide a one-sentence reasoning. Format your response as: Sentiment: [Positive/Negative/Neutral]. Reasoning: [Your one-sentence reason]."
        self.assertEqual(mock_gemini.call_args_list[0][0][0], expected_prompt_1)

        analysis = result_state["data"]["analyst_signals"]["news_sentiment_agent"].get(chinese_ticker)
        self.assertIsNotNone(analysis)
        # P:1, N:1, Neu:1 -> Neutral
        self.assertEqual(analysis["signal"], "neutral") 
        # Confidence for neutral in this tie-breaking case would be 1/3 (approx 33%)
        self.assertEqual(analysis["confidence"], round((1/3)*100)) 
        self.assertEqual(len(analysis["articles_analyzed"]), 3)
        self.assertEqual(analysis["articles_analyzed"][0]["sentiment"], "positive")
        self.assertEqual(analysis["articles_analyzed"][1]["sentiment"], "negative")
        self.assertEqual(analysis["articles_analyzed"][2]["sentiment"], "neutral")

    @patch('src.agents.news_sentiment_agent.generate_text_gemini')
    @patch('src.agents.news_sentiment_agent.get_company_news_akshare')
    @patch('src.agents.news_sentiment_agent.get_company_news')
    def test_news_sentiment_agent_non_chinese_stock_success(self, mock_get_news, mock_get_news_akshare, mock_gemini):
        us_ticker = "AAPL.US"
        initial_state = create_test_agent_state(tickers=[us_ticker])
        
        mock_news = self._create_mock_company_news(us_ticker, count=2, source_type="DefaultAPI")
        mock_get_news.return_value = mock_news
        
        mock_gemini.side_effect = [
            "Sentiment: Positive. Reasoning: New product launch.",
            "Sentiment: Positive. Reasoning: Analyst upgrade."
        ]

        result_state = news_sentiment_agent(initial_state)

        mock_get_news.assert_called_once_with(ticker=us_ticker, end_date=initial_state["data"]["end_date"], start_date=initial_state["data"]["start_date"], limit=MAX_NEWS_ITEMS_TO_ANALYZE)
        mock_get_news_akshare.assert_not_called()
        
        self.assertEqual(mock_gemini.call_count, 2)
        
        analysis = result_state["data"]["analyst_signals"]["news_sentiment_agent"].get(us_ticker)
        self.assertIsNotNone(analysis)
        # P:2, N:0, Neu:0 -> Positive
        self.assertEqual(analysis["signal"], "positive")
        self.assertEqual(analysis["confidence"], 100)
        self.assertEqual(len(analysis["articles_analyzed"]), 2)
        self.assertEqual(analysis["articles_analyzed"][0]["sentiment"], "positive")

    @patch('src.agents.news_sentiment_agent.generate_text_gemini')
    @patch('src.agents.news_sentiment_agent.get_company_news_akshare')
    @patch('src.agents.news_sentiment_agent.get_company_news')
    def test_news_sentiment_agent_gemini_failure(self, mock_get_news, mock_get_news_akshare, mock_gemini):
        ticker = "MSFT.US"
        initial_state = create_test_agent_state(tickers=[ticker])
        
        mock_news = self._create_mock_company_news(ticker, count=1)
        mock_get_news.return_value = mock_news
        
        mock_gemini.return_value = None # Simulate Gemini failure

        result_state = news_sentiment_agent(initial_state)
        
        analysis = result_state["data"]["analyst_signals"]["news_sentiment_agent"].get(ticker)
        self.assertIsNotNone(analysis)
        # If Gemini fails, sentiment for that article should be neutral (default)
        self.assertEqual(analysis["articles_analyzed"][0]["sentiment"], "neutral")
        self.assertEqual(analysis["articles_analyzed"][0]["reason_from_llm"], "Could not determine sentiment from LLM response.")
        # Overall sentiment with one neutral article
        self.assertEqual(analysis["signal"], "neutral")
        self.assertEqual(analysis["confidence"], 100) # 1 neutral / 1 total

    @patch('src.agents.news_sentiment_agent.generate_text_gemini')
    @patch('src.agents.news_sentiment_agent.get_company_news_akshare')
    @patch('src.agents.news_sentiment_agent.get_company_news')
    def test_news_sentiment_agent_gemini_malformed_response(self, mock_get_news, mock_get_news_akshare, mock_gemini):
        ticker = "AMZN.US"
        initial_state = create_test_agent_state(tickers=[ticker])
        
        mock_news = self._create_mock_company_news(ticker, count=1)
        mock_get_news.return_value = mock_news
        
        malformed_text = "This is not the expected format."
        mock_gemini.return_value = malformed_text

        result_state = news_sentiment_agent(initial_state)
        
        analysis = result_state["data"]["analyst_signals"]["news_sentiment_agent"].get(ticker)
        article_analysis = analysis["articles_analyzed"][0]
        
        self.assertEqual(article_analysis["sentiment"], "neutral") # Default due to parsing error
        self.assertIn(malformed_text, article_analysis["reason_from_llm"])
        self.assertEqual(analysis["signal"], "neutral")

    @patch('src.agents.news_sentiment_agent.get_company_news')
    def test_news_sentiment_agent_no_news_found(self, mock_get_news):
        ticker = "GOOG.US"
        initial_state = create_test_agent_state(tickers=[ticker])
        
        mock_get_news.return_value = [] # No news

        result_state = news_sentiment_agent(initial_state)
        
        analysis = result_state["data"]["analyst_signals"]["news_sentiment_agent"].get(ticker)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["signal"], "neutral")
        self.assertEqual(analysis["confidence"], 0)
        self.assertEqual(analysis["reasoning"], "No news items found for analysis.")
        self.assertEqual(len(analysis["articles_analyzed"]), 0)

    @patch('src.agents.news_sentiment_agent.generate_text_gemini')
    @patch('src.agents.news_sentiment_agent.get_company_news_akshare')
    def test_news_sentiment_agent_news_limit(self, mock_get_news_akshare, mock_gemini):
        ticker = "BABA.SS" # Chinese stock, use akshare
        initial_state = create_test_agent_state(tickers=[ticker])

        # Mock more news items than MAX_NEWS_ITEMS_TO_ANALYZE
        mock_news_items = self._create_mock_company_news(ticker, count=MAX_NEWS_ITEMS_TO_ANALYZE + 2)
        mock_get_news_akshare.return_value = mock_news_items
        
        # Gemini will be called MAX_NEWS_ITEMS_TO_ANALYZE times
        mock_gemini.return_value = "Sentiment: Neutral. Reasoning: Generic news."

        result_state = news_sentiment_agent(initial_state)

        mock_get_news_akshare.assert_called_once_with(ticker=ticker, limit=MAX_NEWS_ITEMS_TO_ANALYZE)
        self.assertEqual(mock_gemini.call_count, MAX_NEWS_ITEMS_TO_ANALYZE)
        
        analysis = result_state["data"]["analyst_signals"]["news_sentiment_agent"].get(ticker)
        self.assertEqual(len(analysis["articles_analyzed"]), MAX_NEWS_ITEMS_TO_ANALYZE)


if __name__ == '__main__':
    unittest.main()
