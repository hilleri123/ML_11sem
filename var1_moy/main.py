#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 

from clint.textui import progress

# Importing the dataset
dataset = pd.read_csv('data_v1-01.csv')
X_base = dataset.iloc[:, 0:1].values
y_base = dataset.iloc[:, 1].values

def plot_xy(px, py):
    to_plot = sorted([(x, y) for x, y in zip(px, py)], key=lambda a: a[0][0])
    x_plot = [i[0] for i in to_plot]
    y_plot = [i[1] for i in to_plot]
    return (x_plot, y_plot)

fix_x = 10
mse_size = []
test_size_base = [0.2, 0.3, 0.4, 0.5]
for test_size in test_size_base:
    X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=test_size, random_state=0)
    mse = [[],[],[]]
    bias = [[],[],[]]
    var = [[],[],[]]
    print(f'test_size={test_size}')
    for m in progress.bar(range(2, 9)):
        poly_reg = PolynomialFeatures(degree=m)
        X = X_train
        y = y_train
        X_poly = poly_reg.fit_transform(X)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y)
    
        y_pred_test = pol_reg.predict(poly_reg.fit_transform(X_test))
        mse_test = np.mean((y_test - y_pred_test)**2)
        mse[0].append(mse_test)

        bias_test = (y_test - y_pred_test)**2
        bias[0].append(bias_test[fix_x])
        var_test = np.var(y_pred_test) # несколько измерений
        var[0].append(var_test[fix_x])

        y_pred_train = pol_reg.predict(poly_reg.fit_transform(X_train))
        mse_train = np.mean((y_train - y_pred_train)**2)
        mse[1].append(mse_train)
        
        bias_train = (y_train - y_pred_train)**2
        bias[1].append(bias_train[fix_x])
        var_train = np.var(y_pred_train)
        var[1].append(var_train[fix_x])

        y_pred_base = pol_reg.predict(poly_reg.fit_transform(X_base))
        mse_base = np.mean((y_base - y_pred_base)**2)
        mse[2].append(mse_base)
    
        bias_base = (y_base - y_pred_base)**2
        bias[2].append(bias_base[fix_x])
        var_base = np.var(y_pred_base)
        var[2].append(var_base[fix_x])

        f1 = plt.figure()
        plt.scatter(X_base, y_base, color='red')
        x_plt, y_plt = plot_xy(X_train, y_pred_train)
        plt.plot(x_plt, y_plt, color='blue')
        plt.title(f'Linear Regression train m={m} test_size={test_size}')
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.legend()
        plt.savefig(f'train_{m}_{test_size*10}.png')
        plt.close(f1)

        f2 = plt.figure()
        plt.scatter(X_base, y_base, color='red')
        x_plt, y_plt = plot_xy(X_test, y_pred_test)
        plt.plot(x_plt, y_plt, color='blue')
        plt.title(f'Linear Regression test m={m} test_size={test_size}')
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.legend()
        plt.savefig(f'test_{m}_{test_size*10}.png')
        plt.close(f2)

        f2 = plt.figure()
        x_plt, y_plt = plot_xy(X_train, bias_train)
        plt.plot(x_plt, y_plt, color='blue', label='train')
        x_plt, y_plt = plot_xy(X_test, bias_test)
        plt.plot(x_plt, y_plt, color='green', label='test')
        plt.title(f'Bias m={m} test_size={test_size}')
        plt.xlabel('X')
        plt.ylabel('bias')
        plt.legend()
        plt.savefig(f'bias_{m}_{test_size*10}.png')
        plt.close(f2)

        f2 = plt.figure()
        x_plt, y_plt = plot_xy(X_train, var_train)
        plt.plot(x_plt, y_plt, color='blue', label='train')
        x_plt, y_plt = plot_xy(X_test, var_test)
        plt.plot(x_plt, y_plt, color='green', label='test')
        plt.title(f'Variance m={m} test_size={test_size}')
        plt.xlabel('X')
        plt.ylabel('var')
        plt.legend()
        plt.savefig(f'var_{m}_{test_size*10}.png')
        plt.close(f2)

    mse_size.append(mse)
    f3 = plt.figure()
    plt.plot(range(2, 9), mse[0], label='mse_test', color='blue')
    plt.plot(range(2, 9), mse[1], label='mse_train', color='green')
    #plt.plot(range(2, 9), mse[2], label='mse')
    plt.title(f'MSE (number of regressors) test_size={test_size}')
    plt.xlabel('m')
    plt.ylabel('Mse')
    plt.legend()
    plt.savefig(f'MSE_{test_size*10}.png')
    plt.close(f3)

for idx, m in enumerate(range(2, 9)):
    f4 = plt.figure()
    plt.plot(test_size_base, [mse[0][idx] for mse in mse_size], label='mse_test', color='blue')
    plt.plot(test_size_base, [mse[1][idx] for mse in mse_size], label='mse_train', color='green')
    #plt.plot(range(2, 9), mse[2], label='mse')
    plt.title(f'MSE (test size) m={m}')
    plt.xlabel('test_size')
    plt.ylabel('Mse')
    plt.legend()
    plt.savefig(f'MSE_{m}.png')
    plt.close(f4)
