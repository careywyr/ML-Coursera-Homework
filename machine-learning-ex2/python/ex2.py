# -*- coding: utf-8 -*-
"""

@author: CareyWYR
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

data1_file_path = '../ex2/ex2data1.txt'

data = pd.read_csv(data1_file_path,names=['exam1', 'exam2', 'admitted'])


def sigmoid(z):
    sigmoid = 1/ (1 + math.exp(-z))
    return sigmoid
    
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

X = data.iloc[:, 0:2 ]
X.insert(0, 'one', pd.Series(np.ones(data.shape[0])))
X = np.matrix(X.values)
y = np.matrix(data.iloc[:, 2:3].values)
initial_theta = np.matrix(np.zeros(X.shape[1]))

