import numpy as np
from .wright import wright


def test_wright_scalar():
    assert np.allclose(wright(18.0, 35.0, 200000.0), 1025.359957453976)


def test_wright_3D():
    np.random.seed(123)
    thetao = np.random.normal(15.0, 5.0, (5, 5))
    so = np.random.normal(35.0, 1.5, (5, 5))
    pressure = np.random.normal(2000.0, 500.0, (5, 5))
    result = wright(thetao, so, pressure)

    reference = np.array(
        [
            [1026.27714931, 1025.80466965, 1024.00867435, 1027.18649828, 1025.56678053],
            [1023.57166816, 1024.54489489, 1024.36623837, 1023.60821608, 1027.92001976],
            [1026.46215987, 1026.07983447, 1024.87526491, 1025.60248704, 1026.76781679],
            [1025.49313182, 1021.07436173, 1022.61556652, 1025.4144156, 1025.92198395],
            [1025.08902445, 1026.81827678, 1027.38010227, 1025.6464104, 1029.7902152],
        ]
    )

    assert np.allclose(result, reference)
