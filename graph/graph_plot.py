import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(x1, model):

    x_min, x_max = x1.min() - 1, x1.max() + 1
    weights = -model.weights[0] / model.weights[1]
    line_points = np.linspace(x_min, x_max)
    plt.plot(line_points, weights * line_points -
             (model.bias / model.weights[1]))


def plot_iris_data(dataframe):

    plt.scatter(
        dataframe.loc[:50, 'sepallength'], dataframe.loc[:50, 'petallength'],
        color='red',
        marker='x',
        label='Iris setosa')
    plt.scatter(
        dataframe.loc[50:100,
                      'sepallength'], dataframe.loc[50:100, 'petallength'],
        color='blue',
        marker='o',
        label='Iris versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
