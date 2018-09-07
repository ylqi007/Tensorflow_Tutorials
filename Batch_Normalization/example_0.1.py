# Code adapted from Andrej Karpathy's CS231n slides.
# 2018-09-07

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

inputs = np.random.randn(500, 200)
hidden_layers = [200] * 10
nonlinearities = ['tanh']*len(hidden_layers)

activations = {
        'relu': lambda x:np.maximum(0, x),
        'tanh': lambda x:np.tanh(x)}

hidden_activations = {}

for i in range(len(hidden_layers)):
    X = inputs if i == 0 else hidden_activations[i-1]
    fan_in = X.shape[1]
    fan_out = hidden_layers[i]

    W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) # Xavier Init
    # W = np.random.randn(fan_in, fan_out) * 0.01 # Normal init

    H = np.dot(X, W)
    H = activations[nonlinearities[i]](H)

    hidden_activations[i] = H


layer_means = [np.mean(H) for i, H in hidden_activations.items()]
layer_stds = [np.std(H) for i, H in hidden_activations.items()]

plt.figure()
plt.subplot(121)
plt.plot(list(hidden_activations.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(list(hidden_activations.keys()), layer_stds, 'or-')
plt.title('layer std')

fig = plt.figure()
for i, H in hidden_activations.items():
    ax = plt.subplot(1, len(hidden_activations), i+1)
    ax.hist(H.ravel(), 30, range=(-1,1))
    ax.grid(True)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plt.title('Layer: %s \nMean: %1.4f\nStd: %1.6f' % (
            i+1, layer_means[i], layer_stds[i]), fontsize=10)
plt.show()

