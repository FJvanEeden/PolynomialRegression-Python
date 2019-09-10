# Importing of libraries and packages
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Loads the dataSet and assigns values
dataSet = pd.read_csv('salariesPerPosition.csv')    # Calls pandas csv file reader
xAxisData = dataSet.iloc[:, 1:2].values     # Assigns x axis values from called dataset
yAxisData = dataSet.iloc[:, 2].values       # Assigns y axis values from called dataset

# Splitting the dataSet into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(xAxisData, yAxisData, test_size = 0.2, random_state = 0)

# Scales the features from split dataset
scaleX = StandardScaler()   
x_train = scaleX.fit_transform(x_train)     # Fits and transfroms scaled data from training set
x_test = scaleX.transform(x_test)       # Transfroms scaled data from testing set

# Fits Linear Regression to the dataSet
regLine = LinearRegression()
regLine.fit(xAxisData, yAxisData)

# Fitting Polynomial Regression to the dataSet
regPoly = PolynomialFeatures(degree = 4)
xQuadValue = regPoly.fit_transform(xAxisData)   # Fits and transfroms data from x axis data
regPoly.fit(xQuadValue, yAxisData)      # Fits polynomial data
polyLine = LinearRegression()
polyLine.fit(xQuadValue, yAxisData)

# Plots the Linear Regression and Polynomial Regression results
plt.scatter(xAxisData, yAxisData, color = 'red', label= 'All positions')
plt.plot(xAxisData, regLine.predict(xAxisData), color = 'black', label='Best fitting line')

x_grid = np.arange(min(xAxisData), max(xAxisData), 0.1)     # Arranges and reshapes x axis data to grid
x_grid = x_grid.reshape((len(x_grid), 1))
plt.plot(x_grid, polyLine.predict(regPoly.fit_transform(x_grid)), color = 'blue', label='Quadratic curve', linestyle='--')

# Set figure details
plt.title('Polynomial Regression of salay growth measured against position level')
plt.xlabel('Position level')
plt.ylabel('Salary Estimate')
plt.grid(True)
plt.legend(loc='best')
plt.show()

