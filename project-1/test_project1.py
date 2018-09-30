import unittest

import numpy as np

from project1 import Reader, Preprocessor


class TestReader(unittest.TestCase):
    def setUp(self):
        self.reader = Reader()

    def test_read_returns_data_and_labels(self):
        data, labels = self.reader.read()
        self.assertEqual(type(data), np.ndarray)
        self.assertEqual(type(labels), np.ndarray)

    def test_read_returns_example_data(self):
        data, labels = self.reader.read('example')
        self.assertEqual(data.shape, (2, 5, 3))
        self.assertEqual(labels.shape, (2, ))


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.input = np.array([[1, 2, 3], [4, 5, 6]])
        self.preprocessor = Preprocessor()

    def test_vectorization_returns_numpy_array(self):
        result = self.preprocessor.vectorize(self.input)
        self.assertEqual(type(result), np.ndarray)

    def test_vectorization_returns_column_vector(self):
        elements = self.input.shape[0] * self.input.shape[1]
        expected_size = (elements,)
        result = self.preprocessor.vectorize(self.input)
        self.assertEqual(result.shape, expected_size)

    def test_vectorization_returns_elements_in_order(self):
        result = self.preprocessor.vectorize(self.input)
        self.assertTrue(np.array_equal(result, [1, 2, 3, 4, 5, 6]))

    def test_transform(self):
        result = self.preprocessor.transform(np.array([-3, -2, -1, 0, 1, 2, 3]), lambda x: 1 if x > 0 else 0)
        self.assertTrue(np.array_equal(result, [0, 0, 0, 0, 1, 1, 1]))



if __name__ == '__main__':
    unittest.main()