import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlp import *

DATA_FILE_PATH = "data/Iris.csv"

data = pd.read_csv(DATA_FILE_PATH)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
species = { 
    "Iris-setosa" : 0,
    "Iris-versicolor" : 1,
    "Iris-virginica" : 2
    }

X = data[features].to_numpy()
y = data["Species"].map(species).to_numpy(dtype=int)

rng = np.random.default_rng(3)
perm = rng.permutation(len(X))
X = X[perm]
y = y[perm]

n_train = int(0.8 * y.shape[0])
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

nb_inputs = X.shape[1]
layer_sizes = [nb_inputs, 6, 6, 3]
mlp = MLP(layer_sizes)
mlp.set_output_layers_activation(softmax, softmax_prime)
mlp.set_cost_func(log_likelihood, log_likelihood_prime)

epochs = 10000
learning_rate = 0.0005
mlp.train(X_train, y_train, epochs= epochs, learning_rate= learning_rate)

losses = mlp.get_last_training_data()["train_losses"]

global_pred = mlp.predict(X)
global_pred = np.argmax(global_pred, axis=1)
global_accuracy = np.mean(global_pred == y)
print("global accuracy : ", global_accuracy)

test_pred = mlp.predict(X_test)
test_pred = np.argmax(test_pred, axis=1)
test_accuracy = np.mean(test_pred == y_test)
print("test accuracy : ", test_accuracy)

x = np.arange(epochs)
y = losses

plt.plot(x, y)
plt.show()