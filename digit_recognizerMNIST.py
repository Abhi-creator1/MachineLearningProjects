import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_train.csv')
data.head()

data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

#

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255.
#_,m_train = X_train.shape

#NEURAL NETWORK
#Initializa parameters
def init_params():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2

def Relu(z):
    return np.maximum(z,0)

def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A

#forard propogation
def forward_prop(W1,b1,W2,b2,X):
    z1 = W1.dot(X)+b1
    A1 = Relu(z1)
    z2 = W2.dot(A1)+b2
    A2 = softmax(z2)
    return z1,A1,z2,A2

#Back propogation
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_Relu(z):
    return z>0

def back_prop(z1,A1,z2,A2,W1,W2,X,Y):
    one_hot_Y = one_hot(Y)
    dz2 = A2 - one_hot_Y
    dW2 = 1/m * dz2.dot(A1.T)
    db2 = 1/m * np.sum(dz2)
    dz1 = W2.T.dot(dz2) * deriv_Relu(z1)
    dW1 = (1/m) * dz1.dot(X.T)
    db1 = (1/m) * np.sum(dz1)
    return dW1,db1,dW2,db2

#update parameters
def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#gradiant desent
def get_prediction(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        z1, A1, z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(z1, A1, z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

#running gradiant decent
W1,b1,W2,b2 = gradient_descent(X_train,Y_train,0.21,501)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

