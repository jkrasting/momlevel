from momlevel import reference
from momlevel.test_data import generate_test_data

dset = generate_test_data()


def test_setup_reference_state():
    result = reference.setup_reference_state(dset, eos="Wright")
    expected_vars = [
        "thetao",
        "so",
        "volcello",
        "rho",
        "volo",
        "masso",
        "rhoga",
        "areacello",
    ]
    assert len(set(expected_vars) - set(result.variables)) == 0
