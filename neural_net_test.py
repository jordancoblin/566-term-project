from neural_net import *
import numpy.testing as npt


def test_relu():
    z = np.array([10, 20, 1, .5, -5])
    expected = [1, 1, 0.7311, 0.6225, 0.0067]
    npt.assert_array_equal(np.round_(relu(z), 4), expected)

test_relu()
print("Everything passed")