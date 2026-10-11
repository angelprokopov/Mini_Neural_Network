import numpy as np

rng = np.random.default_rng(0)

class MLP:
    def __init__(self, sizes):
        self.W = [rng.normal(0,np.sqrt(1 / a),(a,b)) for a,b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1,b)) for b in sizes[1:]]

    def forward(self,X):
        self.acts = [X]
        for i, (W,b) in enumerate(zip(self.W, self.b)):
            z = self.acts[-1] @ W + b
            if i < len(self.W) - 1:
                self.acts.append(np.tanh(z))
            else:  # Softmax output
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                self.acts.append(e / e.sum(axis=1, keepdims=True))  # Numerical stability
        return self.acts[-1]

    def loss(self, probs, y):
        return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

    def backward(self, y, lr):
        n = len(y)
        delta = self.acts[-1].copy()
        delta[np.arange(n), y] -= 1  # Gradient of Softmax + cross-entropy
        delta /= n
        for i in reversed(range(len(self.W))):
            dW = self.acts[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:  # Propagate the error to the previous layer (tanh derivative)
                delta = (delta @ self.W[i].T) * (1 - self.acts[i] ** 2)
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def fit(self, X, y, epochs=2000, lr=1.5, log_every=250):
        history = []
        for epoch in range(epochs + 1):
            probs = self.forward(X)
            self.backward(y, lr)
            if epoch % log_every == 0:
                l = self.loss(probs, y)
                acc = (probs.argmax(axis=1) == y).mean()
                history.append((epoch, l, acc))
                print(f"epoch {epoch:5d} | loss {l:.4f} | accuracy {acc:.4f}")
        return history

def make_spirals(n_per_class=100,classes=3):
    X = np.zeros((n_per_class * classes, 2))
    y = np.zeros(n_per_class * classes, dtype=int)
    for c in range(classes):
        ix = range(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, 0.2, n_per_class)
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = c
    return X, y


if __name__ == "__main__":
    print("== XOR (the classic problem a single neuron cannot solve) ==")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    net = MLP([2, 8, 2])
    net.fit(X, y, epochs=2000, lr=0.5, log_every=500)
    print(f"Predictions: {net.predict(X)}, expected: {y}")

    print("\n == Three-class spiral ==")
    X, y = make_spirals()
    net = MLP([2, 45, 45, 3])  # 2 inputs, two hidden layers of 45 neurons, 3 classes
    net.fit(X, y, epochs=2000, lr=0.5, log_every=500)
    print(f"Final accuracy: {(net.predict(X) == y).mean()}")