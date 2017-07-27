from sklearn.model_selection import train_test_split
from algoritms.perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
from graph.graph_plot import plot_decision_boundary, plot_iris_data

def main():

    dataframe = raw_data = pd.read_csv('./datasets/iris-last.csv', sep=',')
    dataframe.replace(['Iris-setosa', 'Iris-versicolor'], [0, 1], inplace=True)
        
    train, test = train_test_split(dataframe, test_size=0.5, random_state=42)
    perceptron = Perceptron()
    x_train = train.iloc[:, [0,2]].values
    y_train = train.iloc[:, -1].values
    perceptron.fit(x_train, y_train)
    classification = perceptron.predict(test.iloc[:, [0,2]].values)
    test = test.assign(irisclassification=classification)
    plot_iris_data(dataframe)
    plot_decision_boundary(dataframe.iloc[:,0].values,dataframe.iloc[:,2].values, perceptron)
    plt.show()

if __name__ == "__main__":
    main()
