import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(x, y, z, eta, t):
    # Initializing parameters for the Perceptron
    w = np.zeros(len(x[0]))  # initial weights
    #print(x[0]) [1. 1.60682619 0.99783134]
    #print(w) [0. 0. 0.]
    n = 0

    # Initializing additional parameters to compute sum-of-squared errors
    yhat_vec = np.ones(len(y))  # vector for predictions
    errors = np.ones(len(y))  # vector for errors (actual - predictions)
    J = []  # vector for the SSE cost function

    while n < t:
        for i in range(0, len(x)):
            # summation step
            f = np.dot(x[i], w)

            # activation function
            if f >= z:
                yhat = 1.
            else:
                yhat = 0.
            yhat_vec[i] = yhat

            # updating the weights
            for j in range(0, len(w)):
                w[j] = w[j] + eta * (y[i] - yhat) * x[i][j]

        n += 1

        # computing the sum-of-squared errors
        for i in range(0, len(y)):
            errors[i] = (y[i] - yhat_vec[i]) ** 2
        J.append(0.5 * np.sum(errors))

    return w, J

def perceptron_test(x, w, z, eta, t):
    y_pred = []
    for i in range(0, len(x-1)):
        f = np.dot(x[i], w)

            # activation function
        if f > z:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
    return y_pred

def decision_boundary_plot():
    # plot the decision boundary
    # 0 = w0x0 + w1x1 + w2x2
    # x2 = (-w0x0-w1x1)/w2

    min = np.min(x_test[:, 1])
    max = np.max(x_test[:, 1])
    x1 = np.linspace(min, max, 100)

    def x2(x1, w):
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        x2 = []
        for i in range(0, len(x1 - 1)):
            x2_temp = (-w0 - w1 * x1[i]) / w2
            x2.append(x2_temp)
        return x2

    x_2 = np.asarray(x2(x1, w))

    plt.scatter(features[:, 1], features[:, 2], c=colour)
    plt.plot(x1, x_2)
    plt.show()

##### MAIN CODE #####

# setting the random seed to reproduce results
np.random.seed(5)

# number of observations
obs = 1000

# generating synthetic data from multivariate normal distribution
class_zeros = np.random.multivariate_normal([1, 1], [[1., .95], [.95, 1.]], obs)
class_ones = np.random.multivariate_normal([1, 5], [[1., .85], [.85, 1.]], obs)

# generating a column of ones as a dummy feature to create an intercept
intercept = np.ones((2 * obs, 1))

# vertically stacking the two classes (still 2x1000)
features = np.vstack((class_zeros, class_ones)).astype(np.float32)

# putting in the dummy feature column - bias? 3x1000
features = np.hstack((intercept, features))

# creating the labels for the two classes
label_zeros = np.zeros((obs,1))
label_ones = np.ones((obs, 1))

# stacking the labels, and then adding them to the dataset
# 4 x 1000    [intercept, feature1, feature 2, label]
labels = np.vstack((label_zeros, label_ones))
dataset = np.hstack((features, labels))
colour = labels.reshape(2000,)

# scatter plot to visualize the two classes (red=1, blue=0)
plt.scatter(features[:, 1], features[:, 2], c = colour)
plt.show()

# shuffling the data to make the sampling random
np.random.shuffle(dataset)

# splitting the data into train/test sets
train = dataset[0:int(0.7 * (obs * 2))] # 70 percent to training set
test = dataset[int(0.7 * (obs * 2)):(obs * 2)] # 30 percent to testing

########## Training the Perceptron ##########
# Inputs
# x:   feature data
# y:   outputs
# z:   threshold
# eta: learning rate
# t:   number of iterations

# reshaping the data for the function
x_train = train[:, 0:3] #incercept, feature1 and 2
y_train = train[:, 3] # label

x_test = test[:, 0:3]
y_test = test[:, 3]


z = 0.0  # threshold
eta = 0.1  # learning rate
t = 5  # number of iterations

w, J = perceptron_train(x_train, y_train, z, eta, t)
epoch = np.linspace(1, len(J), len(J))

print ("The weights are: " + str(w) + " The sum-of-squared errors are: " + str(J))

# plt.figure(1)
# plt.plot(epoch, J)
# plt.xlabel('Epoch')
# plt.ylabel('Sum-of-Squared Error')
# plt.title('Perceptron Convergence')
# plt.show()

decision_boundary_plot()

y_pred = perceptron_test(x_test, w, z, eta, t)
plt.scatter(x_test[:, 1], x_test[:, 2], c = y_pred)
plt.show()