import itertools

import numpy as np


class Reader:
    def __init__(self):
        self.sources = {
            '': self._read_empty,
            'example': self._read_example,
        }

    def read(self, source=''):
        return self.sources[source]()

    def _read_empty(self):
        data = np.array([])
        labels = np.array([])
        return data, labels

    def _read_example(self):
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
        return data, labels


class Preprocessor:
    def vectorize(self, matrix):
        return matrix.ravel()

    def transform(self, array, fn):
        return np.array(list(map(fn, array)))

    def preprocess(self, data, labels):
        # Matrix vectorization
        data = np.array([self.vectorize(d) for d in data])

        # Bipolar transformation
        data = np.array([self.transform(d, lambda x: -1 if x == '.' else 1) for d in data])

        return data, labels


class NeuralNetwork:
    def __init__(self, input_size):
        self.W = np.zeros(input_size)
        self.b = np.zeros(1)

    def activate_neuron(self, Y):
        if Y > 0:
            return 1
        elif Y == 0:
            return 0
        else:
            return -1
        # return 1 if Y >= 0 else -1

    def feedforward(self, X):
        Y = X @ self.W + self.b
        # print('Y out:', Y)
        return np.array(list(map(self.activate_neuron, Y)))

    def train(self, X, T):
        Y = self.feedforward(X)

        while not np.array_equal(Y, T):
            # print('X:', X)
            # print('T:', T)
            # print('W:', self.W)
            # print('b:', self.b)
            # print('Y:', Y)

            for x, t in zip(X, T):
                self.W = self.W + x*t
                self.b = self.b + t

            Y = self.feedforward(X)


def main():
    data, labels = Reader().read('example')
    data, labels = Preprocessor().preprocess(data, labels)
    nn = NeuralNetwork(input_size=15)
    nn.train(data, labels)

    # Flipped pixels
    errors = []
    for i in range(16):
        corrupted_pixels = itertools.combinations(range(15), i)
        num_errors = 0
        for pixels in corrupted_pixels:
            # print('Pixels:', pixels)
            corrupted_data = np.copy(data)
            for pixel in pixels:
                corrupted_data[0][pixel] *= -1
            output = nn.feedforward(corrupted_data[0])

            if not np.array_equal(output, [1]):
                num_errors += 1

            # print('X:', data[0])
            # print('Corrupted X:', corrupted_data[0])
            # print('T:', labels[0])
            # # print('W:', nn.W)
            # # print('b:', nn.b)
            # print('Y:', output)
            # print()
        errors.append(num_errors)
    print('Errors:', errors)

    # Undetermined pixels
    errors = []
    first_error = None
    for i in range(16):
        corrupted_pixels = itertools.combinations(range(15), i)
        num_errors = 0
        for pixels in corrupted_pixels:
            # print('Pixels:', pixels)
            corrupted_data = np.copy(data)
            for pixel in pixels:
                corrupted_data[0][pixel] = 0
            output = nn.feedforward(corrupted_data[0])

            if not np.array_equal(output, [1]):
                num_errors += 1

                if first_error is None:
                    first_error = corrupted_data[0]

            # print('X:', data[0])
            # print('Corrupted X:', corrupted_data[0])
            # print('W:', nn.W)
            # print('T:', labels[0])
            # print('W:', nn.W)
            # print('b:', nn.b)
            # print('Y:', output)
            # print()
        errors.append(num_errors)
    print('Errors:', errors)
    print('First error:', first_error)

    # _ _ #
    # # . _
    # _ _ .
    # # . _
    # _ _ #

if __name__ == '__main__':
    main()