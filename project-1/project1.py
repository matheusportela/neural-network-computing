import itertools

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size):
        self.W = np.zeros(input_size) # weights matrix
        self.b = np.zeros(1) # bias

    def _activate_neuron(self, Y):
        '''Heaviside unit step function with half-maximum convention.'''
        if Y > 0:
            return 1
        elif Y == 0:
            return 0
        else:
            return -1

    def feedforward(self, X):
        '''Feed-forward input X through the neural network.'''
        Y = X @ self.W + self.b
        return np.array(list(map(self._activate_neuron, Y)))

    def train(self, X, T):
        '''Trains neural network with input data X and labels T using
        Hebb's rule.'''
        Y = self.feedforward(X)

        # Train until fitting perfectly the training data
        while not np.array_equal(Y, T):
            for x, t in zip(X, T):
                self.W = self.W + x*t
                self.b = self.b + t

            Y = self.feedforward(X)


def generate_modified_data(data, mistake_type=None, num_modified_pixels=0):
    '''Generates modifications of `data` by introducing mistakes.

    Generate one modified data at a time by introducing a
    `mistake_type` such as "corrupted", to flip pixels, or "missing",
    to replace pixels by 0. This function will return data with all
    possible mistakes by combining values [0, 1, ..., 14] in groups of
    size `num_modified_pixels`.
    '''
    # Generate all possible combinations of pixels in the set
    # [0, 1, ..., 14] in groups of size num_modified_pixels
    modified_pixels_combinations = itertools.combinations(
        range(15),
        num_modified_pixels
    )

    for pixels in modified_pixels_combinations:
        # Avoid corrupting original data
        corrupted_data = np.copy(data)
        for pixel in pixels:
            if mistake_type == 'corrupted':
                # Flip pixel value
                corrupted_data[pixel] *= -1
            elif mistake_type == 'missing':
                # Replace pixel by 0
                corrupted_data[pixel] = 0
            else:
                raise ValueError(
                    'mistake_type must be "corrupted" or "missing"'
                )

        # Generate one corrupted data after modifying pixels
        yield corrupted_data


def main():
    # Training data
    data = np.array([
        [
            ['.', '#', '#'],
            ['#', '.', '.'],
            ['#', '.', '.'],
            ['#', '.', '.'],
            ['.', '#', '#']
        ], [
            ['#', '.', '#'],
            ['#', '.', '#'],
            ['.', '#', '.'],
            ['#', '.', '#'],
            ['#', '.', '#']
        ]
    ])
    labels = np.array([1, -1])

    # Preprocessing data
    ## Concatenate column-wise
    data = np.array([d.ravel(order='F') for d in data])
    ## Bipolar transformation
    data = np.array(
        [list(map(lambda x: -1 if x == '.' else 1, d)) for d in data]
    )

    # Defining neural network with 15 input neurons
    nn = NeuralNetwork(input_size=15)

    # Train neural network using training data
    nn.train(data, labels)

    # Test with corrupted pixels
    errors_with_corrupted_pixels = []
    for num_modified_pixels in range(16):
        num_errors = 0
        for modified_data in generate_modified_data(
                data[0],
                mistake_type='corrupted',
                num_modified_pixels=num_modified_pixels
            ):
            output = nn.feedforward(modified_data)

            if not np.array_equal(output, [1]):
                num_errors += 1
        errors_with_corrupted_pixels.append(num_errors)

    # Test with missing pixels
    errors_with_missing_pixels = []
    input_data_for_first_error = None
    for num_modified_pixels in range(16):
        num_errors = 0
        for modified_data in generate_modified_data(
                data[0],
                mistake_type='missing',
                num_modified_pixels=num_modified_pixels
            ):
            output = nn.feedforward(modified_data)

            if not np.array_equal(output, [1]):
                num_errors += 1

                if input_data_for_first_error is None:
                    input_data_for_first_error = modified_data

        errors_with_missing_pixels.append(num_errors)

    # Print results
    print('Errors with corrupted pixels:')
    print(errors_with_corrupted_pixels)
    print('Errors with missing pixels:')
    print(errors_with_missing_pixels)
    print('Input on first error with missing pixels:')
    print(input_data_for_first_error)

if __name__ == '__main__':
    main()