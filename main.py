import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.mlp import *

DATA_FILE_PATH = "data/Iris.csv"

data = pd.read_csv(DATA_FILE_PATH)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
species = { 
    "Iris-setosa" : np.array([1, 0, 0]), 
    "Iris-versicolor" : np.array([0, 1, 0]), 
    "Iris-virginica" : np.array([0, 0, 1]) 
    }

X = data[features].to_numpy()
y = np.zeros([150, 3])

y[:50], y[50:100], y[100:] = species["Iris-setosa"], species["Iris-versicolor"], species["Iris-virginica"]

rng = np.random.default_rng(2)
perm = rng.permutation(len(X))
X = X[perm]
y = y[perm]

n_train = int(0.8 * y.shape[0])
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

nb_inputs = X.shape[1]
layer_sizes = [6, 4, 3]
mlp = Mlp(3, nb_inputs, layer_sizes)

nb_iterations = 10000
learning_rate = 0.1
losses = mlp.train(X_train, y_train, X_test= X_test, y_test= y_test, nb_iteration= nb_iterations, learning_rate= learning_rate)

x = np.arange(nb_iterations)
y1 = losses[0]
y2 = losses[1]

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()