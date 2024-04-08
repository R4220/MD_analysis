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
    return atom(name="H", mass=1.008, id_group='1')


# initialization test ---------------------------------------------------------------------------------------------

def test_initialization(sample_atom):
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

    Notes
    -----
    This test focuses on verifying the initialization process of the 'atom' class, ensuring that all attributes 
    are set correctly. Specifically, we confirm that the 'name' attribute corresponds to "H", the 'mass' attribute
    is set to the expected value for a hydrogen atom, 'id_group' is assigned as '1', and other attributes such as 
    'N', 'id', 'position_past_past', 'position_past', 'position', 'velocity', and 'force' are initialized to 
    appropriate empty values or arrays.
    """

    # Ensure that the name attribute is set to "H" (Hydrogen)
    assert sample_atom.name == "H"
    
    # Ensure that the mass attribute is set to the expected value for a hydrogen atom
    assert sample_atom.mass == 1.008
    
    # Ensure that the id_group attribute is set to '1'
    assert sample_atom.id_group == '1'
    
    # Ensure that the N attribute (number of atoms) is set to 0
    assert sample_atom.N == 0
    
    # Ensure that the id attribute is set to an empty list
    assert sample_atom.id == []
    
    # Ensure that the position_past_past attribute is an empty array with shape (0, 3)
    assert np.array_equal(sample_atom.position_past_past, np.array([], dtype=float).reshape(0, 3))
    
    # Ensure that the position_past attribute is an empty array with shape (0, 3)
    assert np.array_equal(sample_atom.position_past, np.array([], dtype=float).reshape(0, 3))
    
    # Ensure that the position attribute is an empty array with shape (0, 3)
    assert np.array_equal(sample_atom.position, np.array([], dtype=float).reshape(0, 3))
    
    # Ensure that the velocity attribute is an empty array with shape (0, 3)
    assert np.array_equal(sample_atom.velocity, np.array([], dtype=float).reshape(0, 3))
    
    # Ensure that the force attribute is an empty array with shape (0, 3)
    assert np.array_equal(sample_atom.force, np.array([], dtype=float).reshape(0, 3))


# generate_velocity test -----------------------------------------------------------------------------------------
    
def test_generate_velocity(sample_atom):
    """
    Test case for the 'generate_velocity' method of the 'atom' class.
    This test aims to ensure that the 'generate_velocity' method within the 'atom' class accurately calculates the 
    velocity of an atom based on its positions and the time step provided.

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
    To set the stage, we establish the initial conditions by placing the atom at the origin (0, 0, 0) in its 
    past-past position and moving it to (2, 0, 0) in the current position. Next, we invoke the 'generate_velocity' 
    method of the 'atom' class, passing a time step (dt) of 1 unit.
    We then anticipate the velocity by calculating the expected change in position over the given time step. 
    In this scenario, considering a movement solely in the x-direction, we expect a resultant velocity of 
    (1, 0, 0). Subsequently, we compare the computed velocity with our expected value. If they do not match, an 
    AssertionError is raised, signaling a discrepancy in the 'generate_velocity' method's functionality.
    """
    # Set the past-past position of the sample atom to the origin (0, 0, 0)
    sample_atom.position_past_past = np.array([[0, 0, 0]])
    
    # Move the sample atom to the current position (2, 0, 0)
    sample_atom.position = np.array([[2, 0, 0]])
    
    # Invoke the 'generate_velocity' method with a time step (dt) of 1 unit
    sample_atom.generate_velocity(dt=1)

    # For this test scenario, we expect a velocity of (1, 0, 0) since the movement is solely in the x-direction
    expected_velocity = np.array([[1, 0, 0]])
    
    # Compare the computed velocity with the expected velocity
    assert np.array_equal(sample_atom.velocity, expected_velocity)

def test_generate_velocity_frozen(sample_atom):
    """
    Test case for the 'generate_velocity' method of the 'atom' class.
    This test checks if the 'generate_velocity' method correctly calculates the velocity based on the provided 
    positions and time step, even when the atom is frozen.

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
    While testing the 'generate_velocity' method under a specific condition where the atom is frozen, we ensure 
    that the method behaves as expected in this scenario. By setting both the past-past and current positions 
    of the atom to the origin (0, 0, 0), we simulate the condition of the atom being motionless. Following this 
    setup, we invoke the 'generate_velocity' method with a time step (dt) of 1 unit, expecting the resulting 
    velocity to be zero. This serves as a validation step to confirm that the method correctly handles the case 
    when the atom is frozen.
    """

    # Set the past-past position of the sample atom to the origin (0, 0, 0)
    sample_atom.position_past_past = np.array([[0, 0, 0]])
    
    # Set the current position of the sample atom to the origin (0, 0, 0)
    # This simulates the scenario where the atom is frozen
    sample_atom.position = np.array([[0, 0, 0]])
    
    # Invoke the 'generate_velocity' method with a time step (dt) of 1 unit
    sample_atom.generate_velocity(dt=1)

    # Since the atom is frozen, the velocity should be zero
    expected_velocity = np.array([[0, 0, 0]])
    
    # Compare the computed velocity with the expected velocity
    assert np.array_equal(sample_atom.velocity, expected_velocity)

def test_generate_velocity_diff_from_one(sample_atom):
    """
    Test case for the 'generate_velocity' method of the 'atom' class.
    This test checks if the 'generate_velocity' method correctly calculates the velocity based on the provided 
    positions and time step, with a the time step parameter 'dt' different from 1.

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
    This test focuses on evaluating the behavior of the 'generate_velocity' method when the time step parameter 
    'dt' is set to a value different from 1. By setting 'dt' to 2 (ps), we examine how the method handles larger 
    time intervals. The test case simulates a scenario where an atom moves from its initial position to a new 
    position twice as far away over a period of 2 units of time. The expected velocity is calculated based on the 
    change in position over the given time step, considering only the change in the x-direction.
    """

    # Set the past-past position of the sample atom to the origin (0, 0, 0)
    sample_atom.position_past_past = np.array([[0, 0, 0]])
    
    # Set the current position of the sample atom to (4, 0, 0)
    sample_atom.position = np.array([[4, 0, 0]])
    
    # Invoke the 'generate_velocity' method with a time step (dt) of 2 units
    sample_atom.generate_velocity(dt=2)

    # Calculate the expected velocity based on the change in position
    # Since the change in position over 2 units of time is 4 units, the velocity should be (4 / 2) = 2 units per unit time
    expected_velocity = np.array([[1, 0, 0]])
    
    # Compare the computed velocity with the expected velocity
    assert np.array_equal(sample_atom.velocity, expected_velocity)
