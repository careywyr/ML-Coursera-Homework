# -*- coding: utf-8 -*-
"""

@author: CareyWYR
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data1_file_path = '../ex1/ex1data1.txt'
data2_file_path = '../ex1/ex1data2.txt'
theta = np.zeros((2,1))

print("Running warmUpExercise ... \n")

def warmup():
    A = np.eye(5)
    print(A)
    
def computeCost(X,y,theta):
    m = len(y)  
    return sum((np.dot(X,theta)-y)**2)/(2*m)

def gradientDescent(X, y, theta, alpha, iterations):
    theta0 = theta[0][0]
    theta1 = theta[1][0]
    thetas = theta
    for i in range(iterations):
        theta0 = theta0 - alpha / m * sum(np.dot(X , thetas) - y)
        temp_x_i = X[:,1].reshape((m,1))
        theta1 = theta1 - alpha / m * sum(np.multiply((np.dot(X , thetas) - y) , temp_x_i))
        theta = np.vstack((theta0,theta1))
        thetas = theta
    return thetas


def filloutCost(theta0_vals,theta1_vals,J_vals):
    for i,value in enumerate(list(theta0_vals)):
        for j,value in enumerate(list(theta1_vals)):
            t = np.array(([theta0_vals[i]],[theta1_vals[j]]))
            J_vals[i][j] = computeCost(X, y, t);
    return J_vals
                
print('Plot data... \n')
data = np.loadtxt(data1_file_path,delimiter = ',')
X = data[:,0]
y = data[:,1]
m = len(y)  
X = X.reshape((m,1))
y = y.reshape((m,1))
plt.scatter(X,y,c='r',marker='x')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')

print('\nTesting the cost function ...\n')

X=np.concatenate((np.ones((m,1)),X),axis=1)
J = computeCost(X,y,theta)
print('With theta = [0 ; 0]\nCost computed = %s \n' % J)
print('Expected cost value (approx) 32.07\n')

theta = np.array(([-1],[2]))    
J = computeCost(X,y,theta)
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ... \n')
iterations = 1500
alpha = 0.01

theta = gradientDescent(X,y,theta,alpha,iterations)
print('Theta found by gradient descent:\n')
print('%s \n'% theta);
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n');

plt.plot(X[:,1],np.dot(X,theta),'b')
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array((1,3.5)), theta)[0]
print('For population = 35,000, we predict a profit of %s \n' % (predict1*10000))
predict2 = np.dot(np.array((1, 7)) , theta)[0]
print('For population = 70,000, we predict a profit of %s \n' %  (predict2*10000))

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
J_vals = filloutCost(theta0_vals,theta1_vals,J_vals)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
plt.show()

fig2 = plt.figure()


plt.contourf(theta0_vals, theta1_vals, J_vals, 20, alpha = 0.6, cmap = plt.cm.hot)
plt.contour(theta0_vals, theta1_vals, J_vals, colors = 'black')
plt.plot(theta[0], theta[1], 'r',marker='x', markerSize=10, LineWidth=2)


plt.show()

