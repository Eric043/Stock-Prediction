import os
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import io
import base64

def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None, f"No data found for ticker {ticker}"
        return data, None
    except Exception as e:
        return None, str(e)

def train_and_predict(ticker, start_date, end_date):
    data, error = get_stock_data(ticker, start_date, end_date)
    
    if error:
        return None, None, error
    
    if data.shape[0] <= 5:
        return None, None, "Not enough data to proceed."
    
    data = data[['Close']]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    look_back = 5
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    actual_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return actual_stock_price, predicted_stock_price, None

def plot_results(actual_stock_price, predicted_stock_price, ticker):
    if actual_stock_price is None or predicted_stock_price is None:
        return None

    plt.figure(figsize=(14, 5))
    plt.plot(actual_stock_price, color='red', label='Actual Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_str

def index(request):
    return render(request, 'predictor/index.html')

def predict(request):
    if request.method == 'POST':
        try:
            ticker = request.POST.get('ticker')
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')

            actual_stock_price, predicted_stock_price, error = train_and_predict(ticker, start_date, end_date)

            if error:
                return JsonResponse({'error': error}, status=500)

            img_str = plot_results(actual_stock_price, predicted_stock_price, ticker)

            actual_recent_price = f"{actual_stock_price[-1][0]:.2f}" if actual_stock_price is not None and len(actual_stock_price) > 0 else "N/A"
            predicted_recent_price = f"{predicted_stock_price[-1][0]:.2f}" if predicted_stock_price is not None and len(predicted_stock_price) > 0 else "N/A"

            return JsonResponse({
                'image': img_str,
                'actual_recent_price': actual_recent_price,
                'predicted_recent_price': predicted_recent_price
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method.'}, status=400)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' will suppress INFO and WARNING logs, showing only ERROR logs

# Alternatively, you can use the TensorFlow logger
tf.get_logger().setLevel('ERROR')
