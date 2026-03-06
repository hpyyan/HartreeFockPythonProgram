import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#settings for plts
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#-------------------------------------------------------------------------------
#Method1: use of normal equation
X = 2 * np.random.rand(100,1) #shape (100,1), matrix
y = 4 + 3 * X + np.random.randn(100,1) #shape (100,1), matrix
#(defined function y --> m:3, c:4 --> our true ans)


X_b = np.c_[np.ones((100,1)), X]  #shape (100,2)
#np.c_ is combining all ones on lhs, and random data X on rhs, Combined column-wise
#adding x0 = 1 to each instance, keep theta_0 as the const. bias term

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #shape (2,1)
#it is predicting the best fit value of m and c with scatter x dataset
#1st trial value: [[3.95006155][2.95845195]] row1: y-int; row2: slope
#as the noise in random dataset X --> value off from ideal cases
#theta_best: parameter vector of the best fit value


#Making prediction using the parameter vector: theta_best
X_new = np.array([[0],[2]])  #create matrix of 2 row 1 col (row1: 0, row2: 2), shape (2,1)
X_new_b = np.c_[np.ones((2,1)), X_new] # add x0 = 1 to each instance, shape (2,2)
y_predict = X_new_b.dot(theta_best) # predict y value using linear model: y = theta x
print(f'Method 1: normal equation: {y_predict}')

#Ploting results
# plt.scatter(X[:, 0], y, c='b', marker='o', label='data')
# plt.plot(X_new, y_predict, 'r-', label='prediction')
# plt.xlim(0, 2.1)
# plt.ylim(0, 15)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

#-------------------------------------------------------------------------------
#Method 2: Using Scikit-Learn
# it based on scipy.linalg.lstsq() function (least sq. fit)
# It uses the method SVD (singular value decomposition)
lin_reg = LinearRegression()
lin_reg.fit(X, y) #fit data X into function y
print(f'Method2: Scikit-learn: '
      f'linear regression intercept: {lin_reg.intercept_}, coefficient: {lin_reg.coef_}') #the regression parameters found
lin_reg.predict(X_new)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(f'Method2: linear sq. fit function of scipy/numpy: {theta_best_svd}')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
##Batch Gradient Descent

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
eta = 0.1  # learning rate
n_iterations = 1000
m =100

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(f'Batch Gradient Method: {theta}')


##Stochastic Gradient Descent
