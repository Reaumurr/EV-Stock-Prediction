import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Load the banner image
banner = Image.open("StocKnock.png")
banner1 = Image.open("StocKnock2.png")
Tesla = Image.open('Tesla.png')
NVDA = Image.open('Nvidia.png')
Nio = Image.open('Nio.png')

# Load the model pipeline
model_pipeline = joblib.load('model_LinReg.pkl')

# Load SARIMA models for each company
sarima_models = {
    'TSLA': joblib.load('SARIMAX_model_TSLA.pkl'),
    'NVDA': joblib.load('SARIMAX_model_NVDA.pkl'),
    'NIO': joblib.load('SARIMAX_model_NIO.pkl')
}

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return sia.polarity_scores(text)

def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y', interval='1d')  # Get 1 year of data for better SARIMA forecasting
    if stock_data.empty:
        return None
    return stock_data

def create_input_df(company, headlines):
    company_ticker = {'Tesla': 'TSLA', 'Nvidia': 'NVDA', 'NIO': 'NIO'}
    ticker = company_ticker.get(company)
    if not ticker:
        return None

    stock_data = get_stock_data(ticker)
    if stock_data is None:
        return None

    # Filter stock data to include only entries from 2024
    stock_data_2024 = stock_data[stock_data.index.year == 2024]
    if stock_data_2024.empty:
        return None

    latest_stock = stock_data_2024.iloc[-1]

    data = {
        'Company_ID': [ticker],
        'Open': [latest_stock['Open']],
        'High': [latest_stock['High']],
        'Low': [latest_stock['Low']],
        'Close': [latest_stock['Close']],
        'Volume': [latest_stock['Volume']],
        'news_count': [len(headlines)]
    }

    # Initialize sentiment scores
    pos_score = neg_score = neu_score = compound_score = 0

    # Calculate sentiment scores for each headline
    for headline in headlines:
        sentiment = analyze_sentiment(headline)
        pos_score += sentiment['pos']
        neg_score += sentiment['neg']
        neu_score += sentiment['neu']
        compound_score += sentiment['compound']

    # Calculate average sentiment scores
    num_headlines = len(headlines)
    avg_pos_score = pos_score / num_headlines
    avg_neg_score = neg_score / num_headlines
    avg_neu_score = neu_score / num_headlines
    avg_compound_score = compound_score / num_headlines

    # Categorize sentiment based on the average compound score
    sentiment_category = categorize_sentiment(avg_compound_score)

    # Add sentiment scores and category to the data dictionary
    data.update({
        'positive': [avg_pos_score],
        'negative': [avg_neg_score],
        'neutral': [avg_neu_score],
        'compound': [avg_compound_score],
        'sentiment_category': [sentiment_category]
    })

    return pd.DataFrame(data), stock_data_2024

def predict_stock_price(company, headlines):
    if len(headlines) > 10:
        return "Please provide up to 10 headlines."

    input_df, stock_data_2024 = create_input_df(company, headlines)
    if input_df is None:
        return "Invalid company selected or no data available for 2024."

    st.write("Input DataFrame:")
    st.write(input_df)  # Display the input DataFrame for debugging

    # Predict the next closing price
    predicted_next_close = model_pipeline.predict(input_df)[0]

    # Perform SARIMA forecast
    ticker = input_df['Company_ID'][0]
    sarima_model = sarima_models.get(ticker)
    if sarima_model is None:
        return "SARIMA model not available for the selected company."

    # Prepare data for SARIMA forecast with predicted value
    history_with_predicted = stock_data_2024['Adj Close']
    future_with_predicted = np.append(history_with_predicted, predicted_next_close)

    # Prepare data for SARIMA forecast without predicted value
    history_without_predicted = stock_data_2024['Adj Close']

    # Forecast future prices with predicted value
    forecast_steps = 30
    forecast_with_predicted = sarima_model.forecast(steps=forecast_steps, exog=[predicted_next_close])

    # Plot the results
    fig = make_subplots(rows=1, cols=1)

    # Historical data
    fig.add_trace(go.Scatter(x=history_without_predicted.index, y=history_without_predicted, mode='lines', name='Historical Data'))

    # Predicted next close price
    predicted_date = history_with_predicted.index[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_next_close], mode='markers', name='Predicted Next Close'))

    # Forecast data with predicted value
    forecast_index_with_predicted = [predicted_date + pd.Timedelta(days=i) for i in range(1, forecast_steps + 1)]
    forecast_with_predicted_line = go.Scatter(x=forecast_index_with_predicted, y=forecast_with_predicted, mode='lines', name='Forecast')
    fig.add_trace(forecast_with_predicted_line)

    fig.update_layout(title=f"SARIMA Forecast for {company}", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig)

    return f"Predicted Next Close Price: {predicted_next_close}"

def main():
    st.sidebar.image(banner1, use_column_width=True)
    st.sidebar.title("**StocKnock**")
    st.sidebar.write("Welcome to **StocKnock**, where we use sentiment analysis on social media to predict stock prices. Join us for smarter investing!")
    st.sidebar.title("What model do we use?")
    st.sidebar.write("We utilize **Linear Regression** to predict the stock for the next day and **Sarimax** to forecast future stock prices, including the predicted results.")
    st.sidebar.title("Stocks you can predict")
    st.sidebar.write("For the time being, these are the stock that you can predict!")
    st.sidebar.image(Tesla, use_column_width=True)
    st.sidebar.image(NVDA, use_column_width=True)
    st.sidebar.image(Nio, use_column_width=True)
    st.image(banner, use_column_width=True)
    st.title("Stock Price Prediction App")
    st.write("Select a company and provide up to 10 headlines to predict the next stock price based on tweets.")

    company_options = ['Tesla', 'Nvidia', 'NIO']
    company = st.selectbox("Select Company", company_options, key="company_select")

    headlines = st.text_area("Enter Headlines (up to 10 headlines)", key="headlines_input")

    if st.button("Predict", key="predict_button"):
        if headlines:
            headlines = headlines.split("\n")
        else:
            st.error("Please enter headlines.")

        prediction = predict_stock_price(company, headlines)
        st.success(prediction)

if __name__ == "__main__":
    main()
