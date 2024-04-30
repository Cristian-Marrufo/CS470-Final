import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=float)
    X = data[:, :-1]  # features
    y = data[:, -1]   # labels
    return X, y

def add_intercept(X):
    return np.c_[np.ones((len(X), 1)), X]

def loss(y, pred_y):
    return -np.sum(y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y)) / len(y)

def gradient(X, y, pred_y):
    return X.T.dot(pred_y - y) * 2 / len(y)

def logistic_regression(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100):
    X_train = add_intercept(X_train)
    X_val = add_intercept(X_val)
    
    M = np.random.randn(X_train.shape[1], 1)
    best_model = M
    best_performance = 0

    for epoch in range(epochs):
        pred_y = sigmoid(X_train.dot(M))
        grad = gradient(X_train, y_train, pred_y)
        M -= learning_rate * grad

        # Evaluate model on validation data
        pred_y_val = sigmoid(X_val.dot(M))
        pred_labels_val = (pred_y_val > 0.5).astype(int)
        accuracy_val = np.mean(pred_labels_val == y_val)

        if accuracy_val > best_performance:
            best_model = np.copy(M)
            best_performance = accuracy_val

    return best_model

def predict(X, M):
    X = add_intercept(X)
    pred_y = sigmoid(X.dot(M))
    pred_labels = (pred_y > 0.5).astype(int)
    return pred_labels

# Example usage
if __name__ == "__main__":
    # Load training data
    X_train, y_train = load_data('train_data.csv')

    # Load validation data
    X_val, y_val = load_data('val_data.csv')

    # Train logistic regression model
    best_model = logistic_regression(X_train, y_train, X_val, y_val)

    # Evaluate model on validation data
    y_pred_val = predict(X_val, best_model)
    accuracy_val = np.mean(y_pred_val == y_val)
    print("Accuracy on validation data:", accuracy_val)
