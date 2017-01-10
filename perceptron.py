import numpy as np


class Perceptron(object):
    def __init__(self, l_rate=0.01, n_iter=100):
        self._l_rate = l_rate
        self._n_iter = n_iter

    def fit(self, X, y):
        self._weights = np.zeros(1 + X.shape[1])
        self._errors = []
        for i in range(self._n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                prediction = self.predict(x_i)
                update = self._l_rate * (target - prediction)
                self._weights[1:] += update * x_i
                self._weights[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
            if errors == 0:
                print self._weights
                break
        print self._errors

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


if __name__ == '__main__':
    p = Perceptron()
    X = np.array([[1., 2.], [-1., -3.]])
    y = np.array([[-1.], [-1.]])
    p.fit(X, y)
