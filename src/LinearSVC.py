import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


class LinearSVC():
    def __init__(self, learningRate=0.01, n_iter=50, random_state=1, C=1):
        self.learningRate = learningRate
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C

    def fit(self, X, y):
        """Fit training data.
               Parameters
               ---------
               X : {array-like}, shape = [n_examples, n_features]
                 Training vectors, where n_examples is the number of
                 examples and n_features is the number of features.
               y : array-like, shape = [n_examples]
                 Target values.
               Returns
               ------
               self : object
               """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)

        for iteration in range(self.n_iter):

            errors = 0
            for xi, target in zip(X, y):
                # print("This is the loss " , self.net_input(xi))
                # print("This is the weights " , self.w_)

                if self.hingeLoss(xi, target) <= 0:
                    self.w_ += self.learningRate * self.w_

                else:
                    errors += 1
                    self.w_ -= self.learningRate * (self.w_ - self.C * target * xi)  # gradient with respect to the weights
                    self.b_ -= self.learningRate * (self.C * -target)  # gradient with respect to the bias
            print("This is the amount of bad predictions after ", iteration, "iterations ", errors)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def hingeLoss(self, X, yi):
        return max(0, 1 - yi * self.net_input(X))

    def predict(self, X):
        """Return class label after unit step"""
        return np.sign(self.net_input(X))


def plot_decision_regions(X, y, classifier, resolution=0.02, show_after=False):
    # setup marker generator and color map
    markers = ('o', 's',
               '^'
               , 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}'
                    ,
                    edgecolor='black')

if __name__ == '__main__':
    df = pd.DataFrame(datasets.load_iris(as_frame=True).data)

    # select setosa and versicolor
    df = pd.read_csv(
        '../iris.data',
        header=None, encoding='utf-8')

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values

    y = np.where(y == 'Iris-setosa', -1, 1)
    # extract sepal length and petal length

    X = df.iloc[0:100, [0, 2]].values

    LinearSVC = LinearSVC()
    LinearSVC.fit(X, y)
    plot_decision_regions(X, y, LinearSVC)

    plt.title('SVM Decision Boundary')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.show()
