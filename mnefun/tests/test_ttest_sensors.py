import unittest
import numpy as np

class ttest_sensors_tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_is_data_ndarray(self):
        x = [1, 2, 3]
        self.assertTrue(isinstance(x, np.ndarray))

def main():
    unittest.main()

if __name__ == '__main__':
    main()
