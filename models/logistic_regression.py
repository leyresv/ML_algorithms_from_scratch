import numpy as np


class LogisticRegressionClassifier:
    """
    An implementation of a logistic regression classifier
    """

    def __init__(self):
        self.theta = None   # weights vector. Numpy array of shape (1, number of features)
        self.J = None       # cost (float)

    def __sigmoid(self, X, theta):
        """
        Sigmoid function for logistic regression

        :param X: input features. Numpy array of shape (m, n+1) (m=number of input sequences,
                                                                n=number of features + 1 for bias)
        :param theta: weights vector. Numpy array of shape (n+1, 1)
        :return: logistic regression over X. Numpy array of shape (m, 1)
        """
        return 1.0 / (1.0 + np.exp(-np.dot(X, theta)))

    def __cost(self, Y_prob, Y):
        """
        Compute the log loss between predicted labels and gold labels

        :param Y_prob: predicted probabilities. Numpy array of shape (m, 1) (m= number of labels)
        :param Y: true labels. Numpy array of shape (m, 1)
        :return: cost value (float)
        """
        #Y = Y + np.expm1(1e-10)
        return float(-(np.dot(Y.T, np.log(Y_prob)) + np.dot((1-Y).T, np.log(1 - Y_prob)))/Y.shape[0])

    def __gradient_descent(self, X, Y, Y_prob, theta, alpha):
        """
        Update the weights vector

        :param X: training features. Numpy array of shape (m, n+1) (m=number of training sequences,
                                                                     n=number of features + 1 for bias)
        :param Y: training labels. Numpy array of shape (m, 1)
        :param Y_prob: predicted probabilities. Numpy array of shape (m, 1)
        :param theta: weights vector. Numpy array of shape (n+1, 1)
        :param alpha: learning rate. Float
        :return: updated weight vector
        """
        return theta - alpha / X.shape[0] * np.dot(X.T, Y_prob-Y)

    def train(self, X, Y, alpha, num_iters, theta=None, verbose=False):
        """
        Train the logistic regression classifier on the provided X sequences

        :param X: training features. Numpy array of shape (m, n+1) (m=number of training sequences,
                                                                     n=number of features + 1 for bias)
        :param Y: training labels. Numpy array of shape (m, 1)
        :param alpha: learning rate (float)
        :param num_iters: number of training iterations (int)
        :param theta: weights vector. Numpy array of shape (n+1, 1)
        :param verbose: print beginning and end of the process
        :return: reference to the instance object
        """
        if not theta:
            theta = np.zeros((X.shape[1], 1))

        if verbose:
            print("Training LR classifier...")

        for i in range(num_iters):
            Y_prob = self.__sigmoid(X, theta) #Prob for list of sequences
            J = self.__cost(Y_prob, Y)
            theta = self.__gradient_descent(X, Y, Y_prob, theta, alpha)

        self.theta = theta
        self.J = J

        if verbose:
            print(f"Training finished")
            print(f"The cost after training is {self.get_cost()}")
            print(f"The resulting weights vector is {self.get_weights()}")
            print()
        return self

    def predict(self, X):
        """
        Predict polarity labels for the input sequences X (0=Negative, 1=Positive)

        :param X: input sequences. Numpy array of features with shape (m, n+1) (m=number of input sequences,
                                                                             n=number of features + 1 for bias)
        :return: predicted labels. Numpy array of size (m, 1)
        """
        return np.array([1 if self.__sigmoid(x, self.theta) >= 0.5 else 0 for x in X])

    def get_cost(self):
        """
        Get the computed cost

        :return: cost value (float)
        """
        return self.J

    def get_weights(self):
        """
        Get the weights vector

        :return: weights vector
        """
        return [round(t, 8) for t in np.squeeze(self.theta)]

