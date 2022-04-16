from neural_net import *
import numpy.testing as npt


def test_relu():
    z = np.array([10, -.5, 1, .5, -5, 0])
    expected = [10, 0, 1, 0.5, 0, 0]
    npt.assert_array_equal(np.round_(relu(z), 4), expected)

def test_step():
    z = np.array([10, -.5, 1, .5, -5, 0])
    print(step(z))
    expected = [1, 0, 1, 1, 0, 0]
    npt.assert_array_equal(np.round_(step(z), 4), expected)

test_relu()
test_step()
print("Everything passed")