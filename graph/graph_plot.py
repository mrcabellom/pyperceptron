import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(X1, X2, model):

    h = .02
    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1, X2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    V = model.weights
    weights = -V[0] / V[1]
    line_points = np.linspace(x_min, x_max)
    plt.plot(line_points, weights*line_points)


def plot_iris_data(dataframe):

    plt.scatter(
        dataframe.iloc[:50, 0].values, dataframe.iloc[:50, 2].values,
        color='red',
        marker='x',
        label='Iris setosa')
    plt.scatter(
        dataframe.iloc[50:100, 0].values, dataframe.iloc[50:100, 2].values,
        color='blue',
        marker='o',
        label='Iris versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
