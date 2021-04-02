import copy

import matplotlib.pylab as plt
import numpy as np


class HopfieldNet:
    def __init__(self, num_neurons, threshold=0):
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.W = np.zeros([self.num_neurons, self.num_neurons])
        self.y = []

    def train(self, X):
        self.W = np.dot(X.T, X)
        np.fill_diagonal(self.W, 0)

    def activate(self, x):
        self.y = [x]
        t = 0
        neurons = list(range(self.num_neurons))

        while True:
            np.random.shuffle(neurons)
            converged = [False for _ in range(self.num_neurons)]

            for i in neurons:
                t += 1

                self.y.append(copy.copy(self.y[-1]))

                # Hopfield activation function
                y_in = self.y[t - 1]@self.W[:, i]

                if y_in > self.threshold:
                    self.y[t][i] = 1
                elif y_in < self.threshold:
                    self.y[t][i] = -1

                # Convergence test
                converged[i] = self.y[t][i] == self.y[t-1][i]

            if all(converged):
                break

        return self.y[-1]

    def states(self):
        return self.y


def main():
    # Fix seed for test purposes
    # Remove in production environments
    np.random.seed(0)

    # Training data
    data = np.array([
        [1, 1, -1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1],
        [-1, -1, 1, 1, 1, -1],
        [-1, 1, 1, -1, 1, 1],
    ], dtype=np.float64)

    # PART 1
    # Define a Hopfield Network
    hopfield = HopfieldNet(num_neurons=len(data[0]))

    # Train the Hopfield Network
    hopfield.train(data)

    # Finding equilibrium states for the training data
    def find_equilibrium_state(y, data):
        for i, d in enumerate(data):
            if all(y == d):
                return i
        return None

    print('PART 1')
    print('Weights:')
    print(hopfield.W)
    print('\nInput\t\t\t\t Output\t\t\t\t Equilibrium State')
    for x in data:
        y = hopfield.activate(x)
        print(x, '\t', y, '\t', find_equilibrium_state(y, data))

    # PART 2
    # Finding all equilibrium states
    def generate_states(size):
        if size == 1:
            return [1], [-1]
        else:
            data = generate_states(size-1)
            return [d + [1] for d in data] + [d + [-1] for d in data]

    X = [np.array(x, dtype=np.float64) for x in generate_states(len(data[0]))]
    Y = []
    equilibria = []

    for x in X:
        y = hopfield.activate(x)
        Y.append(y)
        equilibria.append(find_equilibrium_state(y, data))
    spurious = list(filter(lambda e: e == None, equilibria))

    print('\nPART 2')
    print('Spurious states:', len(spurious))
    for x, y, e in zip(X, Y, equilibria):
        if e is None:
            print(x)

    print('\nX x X^T')
    print(data@data.T)

    # PART 3
    # Finding all bases of attraction
    bases_of_attraction = {}
    for x, e in zip(X, equilibria):
        bases_of_attraction[e] = bases_of_attraction.get(e, []) + [x]

    print('\nPART 3')
    for e in set(equilibria):
        print('Equilibrium:', data[e])
        print('States:\t\t\t\t Hamming Distance:')
        for s in bases_of_attraction[e]:
            print(s, '\t', sum(s != data[e]))
        print()

    # PART 4
    print('\nPART 4')

if __name__ == '__main__':
    main()
