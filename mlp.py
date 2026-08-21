import numpy as np

class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.n_in, self.n_out = n_in, n_out

        self.b = np.random.randn(n_out)
        self.w = np.random.randn(n_out, n_in)

        self.last_activation = None
        self.last_entries = None

    def activation(self, x: np.ndarray) -> np.ndarray: #here sigmoide
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, entries: np.ndarray) -> np.ndarray:
        self.last_entries = entries
        self.last_activation = self.activation(np.dot(entries,  self.w.T) + self.b)
        return self.last_activation

    def back_prop(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: #return (dw, dw, delta)
        delta = delta * self.last_activation * (1 - self.last_activation)
        dw = delta.T @ self.last_entries
        db = np.sum(delta, axis=0)
        delta = delta @ self.w
        return (dw, db, delta)

class Mlp:
    def __init__(self, nb_layers: int, nb_inputs: int, layer_sizes: list[int]):
        self.nb_layers = nb_layers
        self.layers = []
        for i in range(nb_layers):
            if i == 0:
               self.layers.append(Layer(nb_inputs, layer_sizes[i]))
            else:
               self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i]))

    def mse(self, predicted: np.ndarray, expected: np.ndarray):
        return np.mean((expected - predicted)**2)

    def predict(self, entries: np.ndarray) -> np.ndarray: #launch forward prop and return result
        for i in range(self.nb_layers):
            entries = self.layers[i].forward_prop(entries)
        return entries 

    def gradient_descent(self, predicted: np.ndarray, expected: np.ndarray, learning_rate: float): #launch back prop and apply correction
        N = expected.shape[0] #batch size
        delta = 2 * (predicted - expected) / N #mse derivative
        for i in reversed(range(self.nb_layers)):
            (dw, db, delta) = self.layers[i].back_prop(delta)
            self.layers[i].w -= learning_rate * dw
            self.layers[i].b -= learning_rate * db

    def train(self, entries: np.ndarray, expected: np.ndarray, nb_iteration: int= 1000, learning_rate: float= 1.) -> list[float]: #return losses
        losses = []
        for i in range(nb_iteration):
            pred = self.predict(entries)
            losses.append(self.mse(pred, expected))
            self.gradient_descent(pred, expected, learning_rate)
        return losses