import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_model(model, test_sequences, test_labels, scaler):
    predictions = model.predict(test_sequences)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(test_labels.reshape(-1, 1))

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    return mae, rmse, predictions, actual

def plot_results(actual, predictions, model_name):
    ensure_dir('images')
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {model_name}')
    plt.savefig(f'images/{model_name}_predictions.png')
    plt.show()