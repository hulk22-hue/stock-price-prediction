import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    data = data['Close'].values.reshape(-1, 1) 

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    return train_data, test_data, scaler

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)