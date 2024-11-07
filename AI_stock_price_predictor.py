#Stock Price Predictor AI
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Title for the app
st.title("Stock Price Predictor App")

# User input for Stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define start and end dates for fetching data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Fetch stock data
google_data = yf.download(stock, start=start, end=end)

# Load pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Define the test data split
splitting_len = int(len(google_data) * 0.7)
x_test = google_data.Close[splitting_len:]
x_test = pd.DataFrame(x_test)

# Function to plot graphs with optional extra data
def plot_graph(figsize, values, full_data, extra_data_flag=0, extra_data=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, color='orange')
    plt.plot(full_data.Close, color='blue')
    if extra_data_flag == 1:
        plt.plot(extra_data)
    return fig

# Plot Moving Averages for different intervals
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(window=250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(window=100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare data for prediction
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

# Convert lists to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Make predictions using the trained model
predictions = model.predict(x_data)

# Inverse transform predictions and actual values to get original scale
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame for plotting original vs predicted values
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

# Display the predicted vs actual data
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot the original close price vs predicted close price
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data - not used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)
