import itertools

import matplotlib.pylab as plt
import numpy as np


class ADALINE:
    def __init__(self, input_size, output_size=1):
        self.W = np.zeros([input_size+1, output_size]) # weights and bias matrix
        self.training = False

    def _activate_neuron(self, Y):
        '''Heaviside unit step function.'''
        if self.training:
            return Y

        Y[Y >= 0] = 1
        Y[Y < 0] = -1
        return Y

    def feedforward(self, X):
        '''Feed-forward input X through the ADALINE.'''
        # Add 1 column to X so the bias is incorporated in the last row of W
        X_with_bias = np.hstack((X, np.ones([X.shape[0], 1])))
        Y = X_with_bias @ self.W
        return np.array(list(map(self._activate_neuron, Y)))

    def train(self, X, T, learning_rate=1, learning_rate_scheduling='constant', max_training_iterations=100):
        '''Trains ADALINE with input data X and labels T using
        the delta rule.'''
        self.training = True
        self.training_errors = []
        self.learning_rate = learning_rate
        self.learning_rates = []
        self.learning_rate_scheduling = learning_rate_scheduling

        X_with_bias = np.hstack((X, np.ones([X.shape[0], 1])))
        Y = self.feedforward(X)

        # Train for a definite number of steps
        for i in range(max_training_iterations):
            # Update weights
            for i, (x, y, t) in enumerate(zip(X_with_bias, Y, T)):
                # if i == 7:
                #     import pdb
                #     pdb.set_trace()
                #     print(self.W)

                self.W = self.W - 2*self.learning_rate*np.outer(x, (y - t))

            # Calculate epoch error
            error = self.calculate_mean_square_error(Y, T)
            self.training_errors.append(error)

            # Adaptive learning rate scheduling
            self.learning_rates.append(self.learning_rate)

            if learning_rate_scheduling == 'time-based':
                self.learning_rate *= 1/(1 + 0.0005*(i+1))
            elif learning_rate_scheduling == 'drop-based':
                self.learning_rate *= 0.95
            elif learning_rate_scheduling == 'momentum':
                if len(self.training_errors) >= 2:
                    error_ratio = self.training_errors[-1]/self.training_errors[-2]
                    if error_ratio > 1.04:
                        self.learning_rate = 0.7*self.learning_rate
                    else:
                        self.learning_rate = 1.05*self.learning_rate

            # Output relevant training info
            print(f'iteration #{i+1}')
            print(f'\tlearning_rate: {self.learning_rate}')
            print(f'\terror: {error}')

            # Prepare next iteration
            Y = self.feedforward(X)

        print(f'Final training error: {error}')
        self.training = False
        self.learning_rate = self.learning_rates[-1] # undo last learning_rate update

    def calculate_mean_square_error(self, Y, T):
        return np.sum(((Y - T) ** 2).mean(axis=1))/len(Y)

    def plot_training_errors(self):
        plt.plot(self.training_errors)
        plt.title('Training error')
        plt.xlabel('Training iteration')
        plt.ylabel('Error')
        plt.show()

    def plot_learning_rates(self):
        plt.plot(self.learning_rates)
        plt.title(f'Learning rate with {self.learning_rate_scheduling} scheduling')
        plt.xlabel('Training iteration')
        plt.ylabel('Learning rate')
        plt.show()


def main():
    # Training data
    data = np.array([
        [1, 1, -1],
        [1, 2, -1],
        [2, -1, 1],
        [2, 0, 1],
        [1, -2, 1],
        [0, 0, 1],
        [-1, 2, 1],
        [-2, 1, 1],
        [-1, -1, -1],
        [-2, -2, -1],
        [-2, -1, -1],
    ], dtype=np.float64)
    labels = np.array([
        [-1, -1],
        [-1, -1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [1, -1],
        [1, -1],
        [1, 1],
        [1, 1],
        [1, 1],
    ], dtype=np.float64)

    # PART 1
    # Define a new ADALINE
    adaline = ADALINE(input_size=3, output_size=2)

    # Train ADALINE using training data
    adaline.train(data, labels, learning_rate=0.05, learning_rate_scheduling='drop-based', max_training_iterations=100)

    print('Number of training cycles:')
    print(len(adaline.training_errors))
    print('Learning rate:')
    print(adaline.learning_rate)
    print('Learned weights and bias:')
    print(adaline.W)

    # Test ADALINE with training data
    output = adaline.feedforward(data)
    print('Test MSE:', adaline.calculate_mean_square_error(output, labels))

    # Plot graphs
    adaline.plot_training_errors()
    adaline.plot_learning_rates()


    # PART 2
    # Explore training and test errors when changing data[7][0]
    training_errors = []
    test_errors = []
    weights = []

    # Iterates for -2 <= x < 10, in steps of 0.1
    # for x in np.arange(-2, 10, 0.1):
    for x in np.arange(-2, 10, 0.1):
        # Change value
        data[7][0] = x

        adaline = ADALINE(input_size=3, output_size=2)

        # Train ADALINE using altered training data
        adaline.train(data, labels, learning_rate=0.05, learning_rate_scheduling='drop-based', max_training_iterations=100)

        # Test ADALINE with training data
        output = adaline.feedforward(data)
        error = adaline.calculate_mean_square_error(output, labels)

        # Store training and test errors
        training_errors.append((x, adaline.training_errors[-1]))
        test_errors.append((x, error))
        weights.append((x, adaline.W))

    # Plot graphs
    plt.figure()
    plt.plot([x[0] for x in training_errors], [x[1] for x in training_errors])
    plt.title('Training error for various x')
    plt.xlabel('x')
    plt.ylabel('Training error')
    plt.figure()
    plt.plot([x[0] for x in test_errors], [x[1] for x in test_errors])
    plt.title('Test error for various x')
    plt.xlabel('x')
    plt.ylabel('Test error')
    plt.show()
    plt.plot([x[0] for x in weights], [np.mean(x[1]**2) for x in weights])
    plt.title('Average squared weights for various x')
    plt.xlabel('x')
    plt.ylabel('Avg W')
    plt.show()

if __name__ == '__main__':
    main()
