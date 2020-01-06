import random

import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler


def load_smtp_dataset(seed=42):
    dataset = fetch_kddcup99(subset="smtp", percent10=False, random_state=42)
    data = dataset.data
    # target = np.array([1 if y == b'normal.' else -1 for y in dataset.target])

    # Seed
    random.seed(seed)
    np.random.seed(seed)

    # Data Set
    split_index = int(data.shape[0] * 0.1)
    X_train = data[:split_index]
    X_test = data[split_index:]

    # Validation
    valid_index = int(data.shape[0] * 0.1)
    X_valid = X_test[:valid_index]
    X_test = X_test[valid_index:]

    # Standardization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test


def load_dataset(key):
    loader = {
        "SMTP": load_smtp_dataset
    }
    return loader[key]()
