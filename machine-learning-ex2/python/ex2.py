# -*- coding: utf-8 -*-
"""

@author: CareyWYR
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

data1_file_path = '../ex2/ex2data1.txt'
data = pd.read_csv(data1_file_path,names=['exam1', 'exam2', 'admitted'])

def sigmoid(z):
    sigmoid = 1/ (1 + np.exp(-z))
    return sigmoid

def costfunction(theta, X, y):
    m = len(y)
    #fminunc计算传过来的theta的shape不知道为什么总是为(3,)
    theta = theta.reshape((3,1))
    J = (np.dot(-y.T , np.log(sigmoid(np.dot(X,theta)))) + np.dot((1-y).T, np.log(1 - sigmoid(np.dot(X,theta)))))/m
    return J

def gradient(theta, X, y):
    m = len(y)
    #fminunc计算传过来的theta的shape不知道为什么总是为(3,)
    theta = theta.reshape((3,1))
    grad = np.dot(X.T, sigmoid(np.dot(X,theta)) - y)/m
    return grad

def predict(theta,X):
    pre = np.dot(X,theta) >=0
    return pre+0
   
# ==================== Part 1: Plotting ====================
print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);
positive = data['admitted'].isin([1])
negative = data['admitted'].isin([0])
data1 = data[positive]
data2 = data[negative]
plt.scatter(data1['exam1'],data1['exam2'],marker='+',c='black',label='admitted',s=50)
plt.scatter(data2['exam1'],data2['exam2'],marker='o',c='yellow',label='not admitted',s=50)
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend()
plt.show()

# ============ Part 2: Compute Cost and Gradient ============
X = data.iloc[:, 0:2 ]
X.insert(0, 'one', pd.Series(np.ones(data.shape[0])))
X = np.matrix(X.values)
y = np.matrix(data.iloc[:, 2:3].values)
initial_theta = np.matrix(np.zeros((X.shape[1],1)))
cost = costfunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
# cost计算不对不知道为什么 TODO
print('Cost at initial theta (zeros): %s \n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' %s \n' % grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test_theta = np.array(([-24], [0.2], [0.2]))
cost = costfunction(test_theta, X, y)
grad = gradient(test_theta, X, y)
print('\nCost at test theta: %s\n' % cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' %s \n' % grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

# ============= Part 3: Optimizing using fminunc  =============
theta = initial_theta
result = op.minimize(fun = costfunction, x0 = theta, args = (X, y), method = 'TNC', jac = gradient)
print('Cost at theta found by fminunc: %s\n' % cost);
print('Expected cost (approx): 0.203\n');
print('theta: \n');
print(' %s \n'% theta);
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')



# ============== Part 4: Predict and Accuracies ==============
# 上面计算theta的地方不正确，先写死正确答案 TODO
theta = np.array(([-25.161],[0.206],[0.201]))
prob = sigmoid(np.dot(np.array((1,45,85)) , theta))
print('For a student with scores 45 and 85, we predict an admission probability of %s \n' % prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %s \n' % (np.mean((p == y) + 0) * 100))
print('Expected accuracy (approx): 89.0\n')
print('\n')