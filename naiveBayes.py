import numpy as np

class NaiveBayes:
    def __init__(self):
        self.prior = {}
        self.word_probs = {}
        self.classes = []

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.classes = np.unique(y_train)

        # Calculate class priors
        for c in self.classes:
            self.prior[c] = (y_train == c).sum() / n_samples

        # Calculate word probabilities
        word_counts = {}
        class_counts = {}
        for c in self.classes:
            word_counts[c] = np.zeros(n_features)
            class_counts[c] = 0

        for i in range(n_samples):
            for j in range(n_features):
                if X_train[i, j] > 0:
                    word_counts[y_train[i]][j] += 1
                    class_counts[y_train[i]] += 1

        for c in self.classes:
            self.word_probs[c] = (word_counts[c] + 1) / (class_counts[c] + n_features)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            probs = {}
            for c in self.classes:
                probs[c] = np.log(self.prior[c])
                for i, word in enumerate(x):
                    if word > 0:
                        probs[c] += np.log(self.word_probs[c][i])
            predictions.append(max(probs, key=probs.get))
        return predictions

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=int)
    X = data[:, :-4]  # features
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

    # Train Naive Bayes model
    model = NaiveBayes()
    model.train(X_train, y_train)

    # Evaluate model on validation data
    y_pred = model.predict(X_val)
    acc = accuracy(y_val, y_pred)
    print("Accuracy on validation data:", acc)