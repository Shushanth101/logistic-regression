import numpy as np

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    #intialization
    def __init__(self, *, lr=0.001, epochs=1000):#default values of learning rate and no of iterations
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    #curve-fitting(finding gradients),gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)
            w_gradient = (1/n_samples) * np.dot( X.T , (predictions - y))
            b_gradient = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - w_gradient * self.lr
            self.bias = self.bias - b_gradient * self.lr
            if i%50==0:
                print("epoch: ",i)

    #prediction
    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = sigmoid(linear_predictions)
        class_predictions = []
        for y in y_predictions:
            if y <= 0.5:
                class_predictions.append(0)
            else:
                class_predictions.append(1)

        return class_predictions

    #Accuracy->checks the actual dataset and predicted dataset
    def accuracy(self,y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)
