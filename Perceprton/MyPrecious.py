import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs, make_classification
import random
import tkinter as tk

class Perceptron(object):

    def __init__(self, X, Y, X_test, Y_test):
        #self.epochs = epochs
        self.X = X
        self.Y = Y
        self.W = np.random.rand(1, len(X[:, 0]))
        self.b = random.uniform(0.0, 1.0)
        self.X_t = X_test
        self.Y_t = Y_test

        self.perceptron_training(self.X, self.Y)

    def activation_fn(self, a):
        return 1 if a > 0 else 0

    def predict(self, Xi):
        a = np.dot(self.W,Xi) + self.b
        y = self.activation_fn(a)
        return y

    def perceptron_training(self, X, Y):
        seen = []; i = 0; err = 1; check_req = 0
        while(len(seen) != len(X[0, :])):
           y = self.predict(X[:,i])
           err = Y[i] - y
           if(err != 0):
               self.W += + np.dot(err, np.transpose(X[:, i]))
               self.b += err
               seen.clear()
               check_req = 1
           else:
               if i not in seen:
                    seen.append(i)
               i += 1
               if(check_req == 1):
                   i = 0
                   check_req = 0

        self.visualize_plot(self.X, self.Y, division = 1)
        self.perceptron_test()

    def perceptron_test(self):
        y_pred = []
        print(len(self.X_t))
        for i in range(0, len(self.X_t[0, :])):
            f = np.dot(self.W, self.X_t[:, i]) + self.b
            # activation function
            if f > 0: # 0 - our treshold
                yhat = 1
            else:
                yhat = 0
            y_pred.append(yhat)

        print(y_pred)
        print (self.Y_t)
        sum_t = sum(y_pred)
        sum_perceptron = sum(self.Y_t)
        if (sum_t == sum_perceptron): print ("Proper classification")
        else: print ("Improper classification")
        self.visualize_plot(self.X, self.Y, self.X_t, self.Y_t, division =1, test =1, )

    def visualize_plot(self, X, Y,  X_t = [], Y_t = [], division = 0, test = 0):
        plt.ion()
        plt.scatter(X[0, :], X[1, :], s=50, linewidths=1, c=Y)

        if (division == 1):
            x1 = -self.b/self.W[:,1]
            x2 = -self.b/self.W[:,0]
            plt.plot([0, x2], [x1, 0])

        if (test == 1):
            plt.scatter(X_t[0, :], X_t[1, :], s=50, marker = "*", linewidths=1, c=Y_t)

        try:
            plt.pause(0.2);
        except tk.TclError:
            pass
        plt.show(block=True)

if __name__ == '__main__':
    ### Create data
    size_of_elem = 40
    X, Y = make_classification(n_features=2, n_samples=size_of_elem, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, class_sep = 2.0)
    #X, Y = make_blobs(n_samples=size_of_elem, centers=2, n_features=2)

    ### Data Manipulation
    X = np.transpose(X)
    Y = np.transpose(Y)
    X_train = X[:,0:int(0.7 * size_of_elem)]  # 70 percent to training set
    X_test = X[:,int(0.7 * size_of_elem): size_of_elem]  # 30 percent to testing
    Y_train = Y[0:int(0.7 * size_of_elem)]  # 70 percent to training set
    Y_test = Y[int(0.7 * size_of_elem): size_of_elem]  # 30 percent to testing

    ### create a perceptron
    perceptron = Perceptron(X_train, Y_train, X_test, Y_test)