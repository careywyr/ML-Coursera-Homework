# -*- coding: utf-8 -*-
"""

@author: CareyWYR
"""

import numpy as np
import matplotlib.pyplot as plt

data1_file_path = '../ex1/ex1data1.txt'
data2_file_path = '../ex1/ex1data2.txt'
theta = np.zeros((2,1))

print("Running warmUpExercise ... \n")

def warmup():
    A = np.eye(5)
    print(A)


print('Plot data... \n')
data = np.loadtxt(data1_file_path,delimiter = ',')
X = data[:,0]
y = data[:,1]
m = len(y)  
X = X.reshape((m,1))
y = y.reshape((m,1))
plt.scatter(X,y)


print('\nTesting the cost function ...\n')

X=np.concatenate((np.ones((m,1)),X),axis=1)
J = sum((np.dot(X,theta)-y)**2)/(2*m)
print('With theta = [0 ; 0]\nCost computed = %s \n' % J)

    
