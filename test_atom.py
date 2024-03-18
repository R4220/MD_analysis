# File: test_atom.py
import pytest
import numpy as np
from Codes.class_atom import atom

@pytest.fixture
def sample_atom():
    """
    Fixture for creating a sample atom object.

    This fixture defines an atom object representing a hydrogen atom belonging to group 1.

    Returns
    -------
    atom
        An instance of the 'atom' class with predefined attributes for a hydrogen atom.
    """
    return atom(name="H", mass=1.008, id_group=1)


# initialization test ---------------------------------------------------------------------------------------------

def test_initialization1(sample_atom):
    """
    Test case for the initialization of the 'atom' class.

    This test checks if the 'atom' class is initialized correctly with the provided attributes.
    Hydrogen atom is chosen.

    Parameters
    ----------
    sample_atom : atom
        An instance of the 'atom' class with predefined attributes.

    Raises
    ------
    AssertionError
        If any of the attributes of the 'atom' instance do not match the expected values.
    """
    assert sample_atom.name == "H"
    assert sample_atom.mass == 1.008
    assert sample_atom.id_group == 1
    assert sample_atom.N == 0
    assert sample_atom.id == []
    assert np.array_equal(sample_atom.position_past_past, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_atom.position_past, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_atom.position, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_atom.velocity, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_atom.force, np.array([], dtype=float).reshape(0, 3))


# generate_velocity test -----------------------------------------------------------------------------------------
    
def test_generate_velocity1(sample_atom):
    """
    Test case for the 'generate_velocity' method of the 'atom' class.

    This test checks if the 'generate_velocity' method correctly calculates the velocity based on the provided positions and time step.

    Parameters
    ----------
    sample_atom : atom
        An instance of the 'atom' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated velocity does not match the expected velocity.
    """
    # Assuming dt is 1 for simplicity
    sample_atom.position_past_past = np.array([[0, 0, 0]])
    sample_atom.position = np.array([[2, 0, 0]])
    sample_atom.generate_velocity(dt=1)

    expected_velocity = np.array([[1, 0, 0]])
    assert np.array_equal(sample_atom.velocity, expected_velocity)

def test_generate_velocity2(sample_atom):
    """
    Test case for the 'generate_velocity' method of the 'atom' class.

    This test checks if the 'generate_velocity' method correctly calculates the velocity based on the provided positions and time step, even when the atom is frozen.

    Parameters
    ----------
    sample_atom : atom
        An instance of the 'atom' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated velocity does not match the expected velocity.

    Notes
    -----
    This test considers the case where both atoms are frozen.
    """
    # Assuming dt is 1 for simplicity
    sample_atom.position_past_past = np.array([[0, 0, 0]])
    sample_atom.position = np.array([[0, 0, 0]])
    sample_atom.generate_velocity(dt=1)

    expected_velocity = np.array([[0, 0, 0]])
    assert np.array_equal(sample_atom.velocity, expected_velocity)

def test_generate_velocity3(sample_atom):
    """
    Test case for the 'generate_velocity' method of the 'atom' class.

    This test checks if the 'generate_velocity' method correctly calculates the velocity based on the provided positions and time step, with a the time step parameter 'dt' different from 1.

    Parameters
    ----------
    sample_atom : atom
        An instance of the 'atom' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated velocity does not match the expected velocity.
    """
    # Assuming dt is 1 for simplicity
    sample_atom.position_past_past = np.array([[0, 0, 0]])
    sample_atom.position = np.array([[4, 0, 0]])
    sample_atom.generate_velocity(dt=2)

    expected_velocity = np.array([[1, 0, 0]])
    assert np.array_equal(sample_atom.velocity, expected_velocity)
