import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sum((self.X_train - x) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
        return np.array(predictions)

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=int)
    X = data[:, :-1]  # features
    y = data[:, -1]   # labels
    return X, y

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Example usage
if __name__ == "__main__":
    # Load training data
    X_train, y_train = load_data('train_data.csv')

    # Load validation data
    X_val, y_val = load_data('val_data.csv')

    # Train KNN model
    model = KNN(k=3)
    model.fit(X_train, y_train)

    # Evaluate model on validation data
    y_pred = model.predict(X_val)
    acc = accuracy(y_val, y_pred)
    print("Accuracy on validation data:", acc)
