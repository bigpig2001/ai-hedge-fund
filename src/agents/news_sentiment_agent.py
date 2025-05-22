import json
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.tools.api import get_company_news, get_company_news_akshare
from src.llm.gemini_api import generate_text_gemini

# Constants
MAX_NEWS_ITEMS_TO_ANALYZE = 5 # Limit the number of news items to analyze

def news_sentiment_agent(state: AgentState):
    """
    Analyzes the sentiment of company news using the Gemini API.
    """
    progress.update_status("news_sentiment_agent", None, "Starting news sentiment analysis")
    
    data = state["data"]
    tickers = data.get("tickers", [])
    end_date = data.get("end_date")
    start_date = data.get("start_date") # May not always be present, handle gracefully

    news_sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("news_sentiment_agent", ticker, "Fetching news")
        
        news_items = []
        if ticker.upper().endswith((".SS", ".SZ")):
            # For Chinese stocks, akshare's get_company_news_akshare typically gets recent news.
            # The 'limit' param was added to the function in a previous step.
            news_items_raw = get_company_news_akshare(ticker=ticker, limit=MAX_NEWS_ITEMS_TO_ANALYZE)
        else:
            # For other stocks, use the existing get_company_news
            # Ensure start_date is provided if required by get_company_news, or handle its absence.
            # The existing get_company_news seems to handle optional start_date.
            news_items_raw = get_company_news(
                ticker=ticker, 
                end_date=end_date, 
                start_date=start_date, 
                limit=MAX_NEWS_ITEMS_TO_ANALYZE # Use limit here as well
            )
        
        # Ensure we only take up to MAX_NEWS_ITEMS_TO_ANALYZE, even if the API returned more
        news_items = news_items_raw[:MAX_NEWS_ITEMS_TO_ANALYZE]

        if not news_items:
            progress.update_status("news_sentiment_agent", ticker, "No news found, assigning neutral sentiment")
            news_sentiment_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": "No news items found for analysis.",
                "articles_analyzed": []
            }
            continue

        progress.update_status("news_sentiment_agent", ticker, f"Analyzing sentiment for {len(news_items)} news items")
        
        individual_sentiments = []
        detailed_reasoning = []

        for i, news_item in enumerate(news_items):
            progress.update_status("news_sentiment_agent", ticker, f"Analyzing news item {i+1}/{len(news_items)}")
            
            # Use title for brevity, or a summary if available and title is too short/generic
            # Assuming news_item.title is the most relevant part for quick sentiment analysis
            prompt = f"Analyze the sentiment of the following news headline for company {ticker}: '{news_item.title}'. Is the sentiment positive, negative, or neutral? Provide a one-sentence reasoning. Format your response as: Sentiment: [Positive/Negative/Neutral]. Reasoning: [Your one-sentence reason]."

            gemini_response_text = generate_text_gemini(prompt)

            sentiment = "neutral" # Default if parsing fails
            reason = "Could not determine sentiment from LLM response."

            if gemini_response_text:
                try:
                    # Basic parsing:
                    # Example expected: "Sentiment: Positive. Reasoning: The news indicates strong earnings."
                    sentiment_part = gemini_response_text.split("Sentiment:")[1].split(". Reasoning:")[0].strip().lower()
                    reason_part = gemini_response_text.split("Reasoning:")[1].strip()
                    
                    if "positive" in sentiment_part:
                        sentiment = "positive"
                    elif "negative" in sentiment_part:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral" # Default to neutral if specific terms not found
                    reason = reason_part
                except IndexError:
                    # Fallback if parsing fails, use the whole response as reason
                    reason = f"LLM response: {gemini_response_text}"
                    # Attempt to infer sentiment from keywords if specific format failed
                    if "positive" in gemini_response_text.lower(): sentiment = "positive"
                    elif "negative" in gemini_response_text.lower(): sentiment = "negative"

            individual_sentiments.append(sentiment)
            detailed_reasoning.append({
                "title": news_item.title,
                "url": news_item.url,
                "date": news_item.date,
                "sentiment": sentiment,
                "reason_from_llm": reason
            })

        # Aggregate sentiments
        if not individual_sentiments:
            overall_signal = "neutral"
            confidence = 0
        else:
            positive_count = individual_sentiments.count("positive")
            negative_count = individual_sentiments.count("negative")
            neutral_count = individual_sentiments.count("neutral")
            
            if positive_count > negative_count and positive_count >= neutral_count:
                overall_signal = "positive"
                confidence = round((positive_count / len(individual_sentiments)) * 100)
            elif negative_count > positive_count and negative_count >= neutral_count:
                overall_signal = "negative"
                confidence = round((negative_count / len(individual_sentiments)) * 100)
            else: # Includes cases where neutral is dominant or counts are equal
                overall_signal = "neutral"
                confidence = round((neutral_count / len(individual_sentiments)) * 100)
                # If positive and negative are equal and higher than neutral, could be mixed.
                # For simplicity, this defaults to neutral if not clearly positive or negative.

        news_sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": f"Overall sentiment based on {len(individual_sentiments)} news items. Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}.",
            "articles_analyzed": detailed_reasoning
        }
        progress.update_status("news_sentiment_agent", ticker, f"Done. Overall sentiment: {overall_signal}")

    # Create the news sentiment analysis message
    message_content = json.dumps(news_sentiment_analysis, indent=2)
    message = HumanMessage(
        content=message_content,
        name="news_sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"].get("show_reasoning", False): # Check if key exists
        show_agent_reasoning(news_sentiment_analysis, "News Sentiment Analysis Agent")

    # Add the signal to the analyst_signals
    if "analyst_signals" not in data:
        data["analyst_signals"] = {}
    data["analyst_signals"]["news_sentiment_agent"] = news_sentiment_analysis
    
    progress.update_status("news_sentiment_agent", None, "Finished news sentiment analysis for all tickers")

    return {
        "messages": [message],
        "data": data, # Ensure updated data is passed back
    }
