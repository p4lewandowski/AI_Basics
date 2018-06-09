import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs, make_classification
import tkinter as tk


class Adaline(object):

    def __init__(self, X, Y, X_test, Y_test, eta = 0.01, epochs = 50):
        self.X = X
        self.Y = Y

        self.eta = eta
        self.epochs = epochs
        self.cost = []

        self.W = np.zeros(X.shape[0])
        self.b = 0
        self.X_t = X_test
        self.Y_t = Y_test

        self.adaline_training(self.X, self.Y, self.epochs, self.eta)

    def adaline_training(self, X, Y, epochs, eta):
       for i in range(epochs):
           #prediction = np.dot(X.T, self.W) + self.b
           prediction = np.dot(self.W, X) + self.b
           self.error = (Y - prediction) #true - estimated
           self.W += eta * (np.dot(self.error, np.transpose(X))) #self.eta * X.T.dot(errors)
           erno = self.error.sum()
           self.b += eta * self.error.sum()
           cost = ((self.error**2).sum() / 2.0)
           self.cost.append(cost)
           if(self.cost[i].sum() <= 0.5):
               break

       return self

    def show_results(self):
       self.visualize_plot(self.X, self.Y)
       self.adaline_test()

    def adaline_test(self):
        y_pred = []
        for i in range(0, len(self.X_t[0, :])):
            f = np.dot(self.W, self.X_t[:, i]) + self.b
            if f > 0: # 0 - our treshold
                yhat = 1
            else:
                yhat = -1
            y_pred.append(yhat)

        print(np.array(y_pred))
        print (self.Y_t)
        self.visualize_plot(self.X, self.Y, self.X_t, self.Y_t, division =1, test =1, error_line = 1)

    def visualize_plot(self, X, Y,  X_t = [], Y_t = [], division = 0, test = 0, error_line = 0):
        plt.ion()
        plt.subplot(211)
        plt.scatter(X[0, :], X[1, :], s=50, linewidths=1, c=Y)

        if (division == 1):
            x1 = -self.b/self.W[1]
            x2 = -self.b/self.W[0]
            plt.plot([0, x2], [x1, 0])

        if (test == 1):
            plt.scatter(X_t[0, :], X_t[1, :], s=50, marker = "*", linewidths=1, c=Y_t)
            plt.show()

        if error_line == 1:
            plt.subplot(212)
            plt.plot(range(1, len(self.cost) + 1), np.log10(self.cost), marker='o')
            plt.xlabel('Iterations')
            plt.ylabel('log(Sum-squared-error)')
            plt.title('Adaline - Learning rate for {} '.format(self.eta))

        try:
            plt.pause(0.2);
        except tk.TclError:
            pass
        plt.show(block=True)

if __name__ == '__main__':
    ### Create data
    size_of_elem = 40
    X, Y = make_classification(n_features=2, n_samples=size_of_elem, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, class_sep = 3.0)
    #X, Y = make_blobs(n_samples=size_of_elem, centers=2, n_features=2)


    ### Data Manipulation
    X = np.transpose(X)
    Y[Y == 0] = -1

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    X_train = X_std[:,0:int(0.7 * size_of_elem)]  # 70 percent to training set
    X_test = X_std[:,int(0.7 * size_of_elem): size_of_elem]  # 30 percent to testing
    Y_train = Y[0:int(0.7 * size_of_elem)]  # 70 percent to training set
    Y_test = Y[int(0.7 * size_of_elem): size_of_elem]  # 30 percent to testing

    ### create a perceptron
    adaline = Adaline(X_train, Y_train, X_test, Y_test)
    adaline.show_results()
