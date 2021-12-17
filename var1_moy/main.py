#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data_v1-01.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

def plot_xy(px, py):
    to_plot = sorted([(x, y) for x, y in zip(px, py)], key=lambda a: a[0][0])
    x_plot = [i[0] for i in to_plot]
    y_plot = [i[1] for i in to_plot]
    return (x_plot, y_plot)



# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    x_plt, y_plt = plot_xy(X, lin_reg.predict(X))
    plt.plot(x_plt, y_plt, color='blue')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return
#viz_linear()


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)


y_pred = pol_reg.predict(poly_reg.fit_transform(X))
mse = np.mean((y - y_pred)**2)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    x_plt, y_plt = plot_xy(X, y_pred)
    plt.plot(x_plt, y_plt, color='blue')
    x_plt, y_plt = plot_xy(X, mse)
    plt.plot(x_plt, y_plt, color='green')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return
viz_polymonial()

# Predicting a new result with Linear Regression
lin_reg.predict([[5.5]])
#output should be 249500

# Predicting a new result with Polymonial Regression
pol_reg.predict(poly_reg.fit_transform([[5.5]]))
#output should be 132148.43750003

