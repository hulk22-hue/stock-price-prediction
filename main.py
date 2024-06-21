import numpy as np
from src.fetch_data import fetch_and_save_data
from src.data_preprocessing import load_and_preprocess_data, create_sequences
from src.lstm_model import build_lstm_model, train_lstm_model
from src.evaluate_model import evaluate_model, plot_results

fetch_and_save_data('AAPL', '2010-01-01', '2024-01-01', 'data/stock_data.csv')

train_data, test_data, scaler = load_and_preprocess_data('data/stock_data.csv')

seq_length = 60
train_sequences, train_labels = create_sequences(train_data, seq_length)
test_sequences, test_labels = create_sequences(test_data, seq_length)

model = build_lstm_model((train_sequences.shape[1], 1))
model = train_lstm_model(model, train_sequences, train_labels, epochs=10, batch_size=32)

mae, rmse, predictions, actual = evaluate_model(model, test_sequences, test_labels, scaler)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

plot_results(actual, predictions, "LSTM")