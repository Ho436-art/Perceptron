import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, max_epochs=1000):
        self.lr = lr
        self.max_epochs = max_epochs

    def weighted_sum(self, X):
        return np.asarray(X) @ self.w + self.b

    @staticmethod
    def step(z):
        return (z >= 0).astype(int)

    def predict(self, X):
        return self.step(self.weighted_sum(X))

    def fit(self, X, y, track=False):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []
        self.pred_history = []           # filled only when track=True
        self.converged = False
        self.epochs_ = 0
        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for xi, ti in zip(X, y):
                yhat = int(xi @ self.w + self.b >= 0)
                update = self.lr * (ti - yhat)
                if update != 0:
                    self.w += update * xi
                    self.b += update
                    errors += 1
            self.errors_per_epoch.append(errors)
            if track:
                self.pred_history.append(self.predict(X).copy())
            self.epochs_ = epoch
            if errors == 0:                  # a full pass with no mistakes -> converged
                self.converged = True
                break
        return self

def accuracy(model, X, y):
    return float(np.mean(model.predict(X) == y))
