import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

d = 2
n = 100
u = 1.0


def make_classification(d, n, u, seed=1):
    # Returns: X_train, X_test, y_train, y_test
    np.random.seed(seed)
    a = np.random.randn(d)
    X = np.random.uniform(-u, u, size=(n, d))
    y = np.array([-1 if np.dot(a.T, x) < 0 else 1 for x in X])
    return train_test_split(X, y, test_size=0.3, random_state=seed)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_classification(d, n, u)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    if d == 2:
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        plt.title('Training dataset')
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.savefig('training-dataset.eps')
        plt.show()

        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.title('Test dataset')
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.savefig('test-dataset.eps')
        plt.show()