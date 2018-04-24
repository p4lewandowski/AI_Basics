import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs, make_classification

a = [2, 3, 4]
print(len(a))
size_of_elem = 10
#
# X = np.array([
#     [-2,4,-1],
#     [4,1,-1],
#     [1, 6, -1],
#     [2, 4, -1],
#     [6, 2, -1],])
# Y = np.array([-1,-1,1,1,1])
# print(np.shape(X), np.shape(Y))

# generate 2d classification dataset
#X, Y = make_blob(n_samples=size_of_elem, centers=2, n_features=2)
X, Y = make_classification(n_features=2, n_samples= size_of_elem, n_redundant=0, n_informative=1, n_clusters_per_class=1)
### Plot the data
plt.scatter(X[:,0], X[:,1], s=50, linewidths=2, c = Y)
plt.show()
bias = np.ones((size_of_elem, 1))
### Data Manipulation ###
X = np.hstack((X, bias)) ## add next column for bias
X_train = X[0:int(0.7 *size_of_elem)] # 70 percent to training set
X_test = X[int(0.7 * size_of_elem): size_of_elem] # 30 percent to testing
Y_train = Y[0:int(0.7 *size_of_elem)] # 70 percent to training set
Y_test = Y[int(0.7 * size_of_elem): size_of_elem] # 30 percent to testing
#print(np.shape(X_train), np.shape(X_test), np.shape(Y_train), np.shape(Y_test))

def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 20
    errors = []
    # for t in range(epochs):
    #     total_error = 0
    #     for i, x in enumerate(X):
    #         if (np.dot(X[i], w)*Y[i]) <= 0:
    #             total_error += (np.dot(X[i], w) * Y[i]) #total loss per epoch
    #             w = w + eta*X[i]*Y[i]
    #     errors.append(total_error * -1)
    #     if total_error == 0:
    #         break

    # for t in range(epochs):
    #     total_error = 0
    #     for i, x in enumerate(X):
    #         a = 1 if (np.dot(X[i,0:2], w[0:2])+ w[2]) >= 0 else 0 #weights only?
    #         error_value = a - Y[i];
    #         w[2] = w[2] + error_value;
    #         w[0] = w[0] + error_value * np.transpose(X[i] )
    #         w[1] = w[1] + error_value * np.transpose(X[i])
    #     errors.append(error_value)

    b = 0
    error_value = 0
    w = np.zeros(2) #mogloby tez byc random z zakresu 0 - 1
    for t in range(epochs): #dla liczby iteracji
        total_error = 0
        for i, x in enumerate(X): # dla elementow
            if ((np.dot(np.transpose(X[i, 0:2]), w) + b) * Y[i]) < 0:
                error_value += (np.dot(np.transpose(X[i, 0:2]), w) + b) * Y[i]
                w = w + np.dot(error_value, X[i, 0:2])
                b = b + error_value

        errors.append(error_value)




    print("Number of epochs until proper result = {}".format(t))
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.show()

    return w, b
def plot_result(X, Y):
    X = X[:,:2]
    x2 = [w[0], w[1], -w[1], w[0]]
    x3 = [w[0], w[1], w[1], -w[0]]
    x2x3 = np.array([x2, x3])
    Z, C, U, V = zip(*x2x3)

    plt.quiver(Z,C,U,V,scale=1, color='blue')
    plt.scatter(X[:, 0], X[:, 1], s=50, linewidths=2, c=Y)
    plt.show()

w, b = perceptron_sgd(X_train,Y_train)
plot_result(X_train,Y_train)
print(w, b)