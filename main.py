import math
import random
import util
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import dataprocessing as dp
import matplotlib.pyplot as plt
iris = datasets.load_iris()
#split it in features and labels
X = iris.data
y = iris.target
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

#hours of studying vs good/bad grades
#10 different stidents
#train a model with 8 students
#predict with the remaining 2
#level of accuruarcy of our model


## Deal with data being heavily inbalanced (a lot more negative)
def sparsify(input, size):
    num_to_select = size
    selected_elements = np.random.choice(input, num_to_select, replace=False)
    return selected_elements



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
#print(acc)
#for i in range(len(predictions)):
#print(classes[predictions[i]])

inputVector = dp.create_input('b1/')
##Training
X = []
#yNp = np.array(inputVector).flatten()
y = []
print(inputVector)

for j in range(len(inputVector)):
    y = y + inputVector[j][17::18]

for i in range(len(inputVector)):
    dataPoint = []
    for j in range(len(inputVector[i])):
        if (j + 1) % 18 != 0:
            dataPoint.append(inputVector[i][j])
        if ((j + 1) % 18 == 0):
            X.append(dataPoint)
            dataPoint = []

yNp = np.array(y)
XNp = np.array(X)
yPositive = yNp[y]
xPositive = XNp[y]
print(yNp)
print("-----" * 40)
print(XNp)
print("-----" * 40)
print(yPositive)
print(len(yPositive))
print("-----" * 40)
print(xPositive)
print(len(xPositive))
print("-----" * 40)


yNpNegative = yNp[yNp != True]
xNpNegative = XNp[yNp != True]

yNegative = sparsify(yNpNegative, len(xPositive))
xNegative = xNpNegative[np.random.choice(xNpNegative.shape[0], len(xPositive), replace=False)]
print(yNegative)
print(len(yNegative))
print("-----" * 40)
print(xNegative)
print(len(xNegative))
print("-----" * 40)

X = np.vstack((xNegative, xPositive))
y = np.hstack((yNegative, yPositive))
print("-----" * 40)
print("PLOTS")
x = np.arange(len(X[:,0]))
colors = ['red' if truth else 'blue' for truth in y]
print("-----" * 40)
print("Edgelength")
plt.scatter(x, X[:,0], c=colors)
plt.show()
print("-----" * 40)
print("Neighbourhood")
plt.scatter(x, X[:,1], c=colors)
plt.show()
print("-----" * 40)
print("Distance to starting point")
plt.scatter(x, X[:,2], c=colors)
plt.show()
plt.scatter(x, X[:,3], c=colors)
plt.show()
plt.scatter(x, X[:,4], c=colors)
plt.show()
plt.scatter(x, X[:,5], c=colors)
plt.show()
plt.scatter(x, X[:,6], c=colors)
plt.show()
print("-----" * 40)
print("Distance to destination point")
plt.scatter(x, X[:,7], c=colors)
plt.show()
plt.scatter(x, X[:,8], c=colors)
plt.show()
plt.scatter(x, X[:,9], c=colors)
plt.show()
plt.scatter(x, X[:,10], c=colors)
plt.show()
plt.scatter(x, X[:,11], c=colors)
plt.show()
print("-----" * 40)
print("Clusterness")
plt.scatter(x, X[:,12], c=colors)
plt.show()
plt.scatter(x, X[:,13], c=colors)
plt.show()
print("-----" * 40)
print("Charging station edge")
plt.scatter(x, X[:,14], c=colors)
plt.show()
plt.scatter(x, X[:,15], c=colors)
plt.show()
plt.scatter(x, X[:,16], c=colors)
plt.show()
print("-----" * 40)
#print("Alpha-nearness")
#plt.scatter(x, X[:,17], c=colors)
#plt.show()
#plt.scatter(x, X[:,18], c=colors)
#plt.show()
#plt.scatter(x, X[:,19], c=colors)
#plt.show()
print("-----" * 40)
print(X)
print(len(X))
print("-----" * 40)
print(y)
print(len(y))
print("-----" * 40)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(sum(predictions))
acc = accuracy_score(y_test, predictions)
print(acc)

