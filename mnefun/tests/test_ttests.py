import unittest
import numpy as np

class ttests(unittest.TestCase):

    # test if user inputs int then raise TypeError
    def test_raise_exception(self):
        from mnefun.stats.ttests import ttest_time
        self.assertRaisesRegexp(TypeError, "Input not array", ttest_time, 0)

    # test whether t-values is approximately 1 for 2 x 2 (obs x time points)
    def test_returns_tvalue(self):
        from mnefun.stats.ttests import ttest_time
        outputs = ttest_time(np.eye(2,2))
        np.testing.assert_array_almost_equal(outputs[0], np.array([[1],[1]]),
                                             decimal=2)

if __name__ == '__main__':
    unittest.main()
