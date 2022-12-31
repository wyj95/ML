import numpy as np

class Perceptron:
    def __init__(self):
        self.w = None  # [1, d]
        self.b = 0  # float
        self.d = None  # int
        self.n = None  # int

    def train(self, x: np.array, y: np.array, lr=0.001, runtime=1000):
        """
        train
        :param lr: learning rate
        :param runtime: max running time
        :param x: [n, d]
        :param y: [1, n]
        :return:
        """
        self.d, self.n = np.shape(x)
        alpha = np.zeros((self.n, 1))
        t = 0
        xy = x*y
        while t < runtime:
            flag = True
            for i in range(self.n):
                xi, yi, = x[:, i], y[0, i]
                self.w = np.dot(xy, alpha).T
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    alpha[i, 0] += lr
                    self.b += lr * yi
                    flag = False
            if flag:
                break
        else:
            return "more than max runtime"
        return "SUCCESS"

    def test(self, x, y=None):
        """
        test
        :param x: [d, n]
        :param y: [1, n]
        :return: (pre_y, acc) if y is given else pre_y
        """
        pre_y = np.array([[1 if yi > 0 else -1 for yi in np.dot(self.w, x)[0] + self.b]])
        if y is None:
            return pre_y
        else:
            return pre_y, np.round(np.sum(pre_y == y) / np.shape(y)[1]*100, 2)


if __name__ == "__main__":
    np.random.seed(321)
    x = np.random.random((2, 20))
    w = np.array([[-1, -1]])
    b = 1
    y = np.array([[1 if yi > 0 else -1 for yi in np.dot(w, x)[0] + b]])
    perceptron = Perceptron()
    print(perceptron.train(x, y))
    x = np.random.random((2, 100))
    w = np.array([[-1, -1]])
    b = 1
    y = np.array([[1 if yi > 0 else -1 for yi in np.dot(w, x)[0] + b]])
    print(y)
    print(perceptron.test(x, y))
    print(perceptron.w, perceptron.b)
    import matplotlib.pyplot as plt
    plt.plot([0, 1], [-b/w[0, 0], -(b+w[0, 1])/w[0, 0]], c="red")
    plt.plot([0, 1], [-perceptron.b/perceptron.w[0, 0], -(perceptron.b+perceptron.w[0, 1])/perceptron.w[0, 0]],
             c='black')
    plt.legend(["True", "Train"])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

