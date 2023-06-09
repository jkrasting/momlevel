import numpy as np
from momlevel.eos.linear import density, drho_dtemp, drho_dsal, alpha, beta

rng = np.random.default_rng(123)

thetao = rng.normal(15.0, 5.0, (5, 5))
so = rng.normal(35.0, 1.5, (5, 5))
pressure = rng.normal(2000.0, 500.0, (5, 5))


def test_linear_density_scalar():
    assert np.allclose(density(18.0, 35.0, 200000.0), 1024.4)


def test_linear_density_3D():
    result = density(thetao, so, pressure)

    reference = np.array(
        [
            [1025.81394788, 1026.90606932, 1025.00091149, 1025.27717059, 1024.08590628],
            [1023.98877596, 1024.16018501, 1025.92952293, 1022.71014279, 1024.8782123],
            [1025.10008876, 1027.55778783, 1025.92182738, 1026.86307821, 1023.64990487],
            [1025.73743195, 1021.95404654, 1027.37589565, 1025.12402447, 1023.85371989],
            [1026.44059898, 1024.09844497, 1022.9870277, 1026.62108514, 1023.38341298],
        ]
    )

    assert np.allclose(result, reference)


def test_linear_drho_dtemp():
    assert np.allclose(drho_dtemp(18.0, 35.0, 200000.0), -0.2)


def test_linear_drho_dtemp_3D():
    result = drho_dtemp(thetao, so, pressure)

    assert np.allclose(result, -0.2)


def test_linear_drho_dsal():
    assert np.allclose(drho_dsal(18.0, 35.0, 200000.0), 0.8)


def test_linear_drho_dsal_3D():
    result = drho_dsal(thetao, so, pressure)

    assert np.allclose(result, 0.8)


def test_linear_alpha():
    assert np.allclose(alpha(18.0, 35.0, 200000.0), 0.0001952362358453729)


def test_linear_alpha_3D():
    result = alpha(thetao, so, pressure)

    reference = np.array(
        [
            [0.00019497, 0.00019476, 0.00019512, 0.00019507, 0.0001953],
            [0.00019531, 0.00019528, 0.00019495, 0.00019556, 0.00019515],
            [0.0001951, 0.00019464, 0.00019495, 0.00019477, 0.00019538],
            [0.00019498, 0.0001957, 0.00019467, 0.0001951, 0.00019534],
            [0.00019485, 0.00019529, 0.00019551, 0.00019481, 0.00019543],
        ]
    )

    assert np.allclose(result, reference)


def test_linear_beta():
    assert np.allclose(beta(18.0, 35.0, 200000.0), 0.0007809449433814916)


def test_linear_beta_3D():
    result = beta(thetao, so, pressure)

    reference = np.array(
        [
            [0.00077987, 0.00077904, 0.00078049, 0.00078028, 0.00078118],
            [0.00078126, 0.00078113, 0.00077978, 0.00078224, 0.00078058],
            [0.00078041, 0.00077855, 0.00077979, 0.00077907, 0.00078152],
            [0.00077993, 0.00078281, 0.00077868, 0.00078039, 0.00078136],
            [0.00077939, 0.00078117, 0.00078202, 0.00077926, 0.00078172],
        ]
    )

    assert np.allclose(result, reference)
