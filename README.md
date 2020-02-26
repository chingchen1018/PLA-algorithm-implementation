# PLA-algorithm-implementation
Perceptron Learning Algorithm (PLA) implementation using python 
dataset : iris dataset



import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
## input data
iris = datasets.load_iris()

## check data information
print(iris.data)
print(iris.target)
print("size of iris data: ", iris.data.shape)

## form a Data Frame
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
Y = pd.DataFrame(iris['target'], columns=['target'])
iris = pd.concat([X, Y], axis=1)
print(iris.head())

del X
del Y

## Observe the relation of sepal/petal width between three iris flowers
class_0 = iris[iris['target']==0]
class_1 = iris[iris['target']==1]
class_2 = iris[iris['target']==2]
# plt.scatter(class_0['sepal width (cm)'], class_0['petal width (cm)'], cmap="red")
# plt.scatter(class_1['sepal width (cm)'], class_1['petal width (cm)'], cmap="blue")
# plt.scatter(class_2['sepal width (cm)'], class_2['petal width (cm)'], cmap="green")
# plt.plot()
# plt.show()


## Choose class_0 / class_1 to implement PLA 
## Because of linear seperable
data = pd.concat([class_0, class_1], axis=0)
change_name={0 : -1,
             1 : 1}
data['target'] = data['target'].map(change_name)
data = pd.concat([data['sepal width (cm)'], data["petal width (cm)"], data['target']], axis=1)
print(data)


def sign_activative(x):
    if x > 0:
        return 1
    else:
        return -1


## initialized weigght
w = np.array([10, -3, -100])

data = pd.concat([pd.DataFrame(np.ones(100)), data], axis = 1)
X = np.array(data)[:,0:3]
Y = np.array(data)[:,3]


def PLA(X, Y, w):
    error = 1
    while error != 0:
        error = 0
        for i in range(len(X)):
            if sign_activative(np.dot(w, X[i])) != Y[i]:
                error = error + 1
                w = w + Y[i] * X[i]
        print("error rate: ", error/150)
        
        a,b = -w[1]/w[2], -w[0]/w[2]
        l = np.linspace(2,4.5)
        plt.plot(l, a*l + b, 'k-')
        plt.scatter(class_0['sepal width (cm)'], class_0['petal width (cm)'], cmap="red")
        plt.scatter(class_1['sepal width (cm)'], class_1['petal width (cm)'], cmap="blue")
        plt.plot()
        plt.show()
    return w

w = PLA(X, Y, w)

print(w)


a,b = -w[1]/w[2], -w[0]/w[2]
l = np.linspace(2,4.5)
plt.plot(l, a*l + b, 'k-')
plt.scatter(class_0['sepal width (cm)'], class_0['petal width (cm)'], cmap="red")
plt.scatter(class_1['sepal width (cm)'], class_1['petal width (cm)'], cmap="blue")
plt.plot()
plt.show()
