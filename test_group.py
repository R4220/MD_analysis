# File: test_group.py
import pytest
import numpy as np
from Codes.class_atom import atom
from Codes.class_group import group 

@pytest.fixture
def sample_group():
    '''
    Fixture for creating a group object.

    Inside this group are positioned hydrogen, carbon, phosphorus, and oxygen atoms. The ID of the group is 1.

    Returns
    -------
    group
        An instance of the 'group' class with predefined attributes.
    '''
    return group(type=['H', 'C', 'P', 'O'], id_group=1)


# initialization test ---------------------------------------------------------------------------------------------

def test_initialization(sample_group):
    '''
    Test case for the initialization of the 'group' class.

    This test checks if the 'group' class is initialized correctly with the provided attributes.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If any of the attributes of the 'group' instance do not match the expected values.

    '''
    assert sample_group.type == ['H', 'C', 'P', 'O']
    assert sample_group.id_group == 1
    assert sample_group.id_tot.size == 0
    assert sample_group.DOF == 0.0
    assert sample_group.Ek == 0.0
    assert sample_group.T == []
    assert np.array_equal(sample_group.Ftot, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_group.force, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_group.Ftot_store, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_group.Vtot_store, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_group.Vtot, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_group.velocity, np.array([], dtype=float).reshape(0, 3))
    assert sample_group.velocity_switch is False


# Kinetic_energy tests --------------------------------------------------------------------------------------------
    
def test_velocity_storage(sample_group):
    '''
    Test case for the 'velocity_storage' method of the 'group' class.

    This test checks if the 'velocity_storage' method correctly stores the velocities of the group's atoms.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the stored velocities of the group's atoms do not match the expected velocities.

    Notes
    -----
    This test initializes two atoms, assigns their positions, and then stores them into the group.
    Finally, it calculates the velocity of the group's atoms.

    '''
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[4, 0, 0]])
    atom1.position_past_past = np.array([[0, 0, 0]])
    atom2.position = np.array([[0, 6, 1]])
    atom2.position_past_past = np.array([[0, 0, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the velocities are stored correctly
    expected_velocity = np.array([[2, 0, 0], [0, 3, 0]], dtype=float)
    assert np.array_equal(expected_velocity, sample_group.velocity)

def test_group_velocity(sample_group):
    '''
    Test case for the 'group_velocity' method of the 'group' class.

    This test checks if the 'group_velocity' method correctly calculates the mean group's velocity.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated mean group velocity does not match the expected group velocity.

    Notes
    -----
    This test initializes two atoms, assigns their positions, and then stores them into the group.
    Finally, it calculates the mean group's velocity.

    '''
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[4, 0, 0]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 6, 1]])
    atom2.position_past_past = np.array([[0, 0, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the group velocity is calculated correctly
    expected_Vtot = np.array([1, 1.5, 0], dtype=float)
    assert np.array_equal(sample_group.Vtot, expected_Vtot)

def test_group_velocity_store(sample_group):
    '''
    Test case for the 'group_velocity_store' method of the 'group' class.

    This test checks if the 'group_velocity_store' method correctly stores the mean group's velocity.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the stored mean group velocity does not match the expected group velocity.

    Notes
    -----
    This test initializes two atoms, assigns their positions, and then stores them into the group.
    Finally, it calculates the mean group's velocity and checks if it is correctly stored.

    '''
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[4, 0, 0]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 6, 1]])
    atom2.position_past_past = np.array([[0, 0, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the group velocity is calculated correctly and stored
    expected_Vtot_store = np.array([[1, 1.5, 0]], dtype=float)
    assert np.array_equal(sample_group.Vtot_store, expected_Vtot_store)

def test_kinetic_energy1(sample_group):
    '''
    Test case for the 'Kinetic_energy' method of the 'group' class.

    This test checks if the 'Kinetic_energy' method correctly calculates kinetic energy for a single atom.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated kinetic energy does not match the expected kinetic energy.

    Notes
    -----
    This test initializes a single atom, assigns its position, and then calculates the kinetic energy.

    '''
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    sample_group.atoms = [atom1]
    sample_group.id_tot = ['1']

    # Set up some dummy positions for the atom
    atom1.position = np.array([[10, 0, 0]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the kinetic energy is calculated correctly
    expected_ek = 0
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

def test_kinetic_energy2(sample_group):
    '''
    Test case for the 'Kinetic_energy' method of the 'group' class.

    This test checks if the 'Kinetic_energy' method correctly calculates kinetic energy for two atoms with 
    different masses.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated kinetic energy does not match the expected kinetic energy.

    Notes
    -----
    This test initializes two atoms with different masses, assigns their positions, and then calculates the 
    kinetic energy.

    '''
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 0]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[0, 0, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the kinetic energy is calculated correctly
    expected_ek = 0.5 * (1.008 * np.linalg.norm([0.5, 0, 0]) ** 2 + 15.999 * np.linalg.norm([-0.5, 0, 0]) ** 2) * 0.0001036426948415943
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

def test_kinetic_energy3(sample_group):
    '''
    Test case for the 'Kinetic_energy' method of the 'group' class.

    This test checks if the 'Kinetic_energy' method correctly calculates kinetic energy for multiple atoms (3) 
    with different masses.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated kinetic energy does not match the expected kinetic energy.

    Notes
    -----
    This test initializes multiple atoms with different masses, assigns their positions, and then calculates 
    the kinetic energy.

    '''
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    atom3 = atom(name="C", mass=12.011, id_group=1)
    sample_group.atoms = [atom1, atom2, atom3]
    sample_group.id_tot = ['1', '2', '3']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the kinetic energy is calculated correctly
    expected_ek = 0.5 * (1.008 * np.linalg.norm([-0.5, 1.5, 2]) ** 2 + 15.999 * np.linalg.norm([-2.5, 1, -1]) ** 2 + 12.011 * np.linalg.norm([3, -2, -1]) ** 2) * 0.0001036426948415943
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)


# Extract_z tests -------------------------------------------------------------------------------------------------
    
def test_extract_from_just_one(sample_group):
    '''
    Test case for the 'Extract_z' method of the 'group' class.

    This test checks if the 'Extract_z' method correctly calculates the mean z coordinates of the group considering just one particle of one single type of atom in the group.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class containing only one atom for testing purposes.

    Raises
    ------
    AssertionError
        If the mean z coordinate calculated by the 'Extract_z' method does not match the expected value.

    Notes
    -----
    This test adds a single atom of a specific type to the group and assigns its position. Then, it calls the 'Extract_z' method to calculate the mean z coordinate of the group and asserts that it matches the expected value.
    '''
    # Add an atom to the group for testing
    atom_ = atom(name="H", mass=1.008, id_group=1)
    sample_group.atoms = [atom_]
    sample_group.id_tot = ['1']

    # Set up some dummy positions for the atoms
    atom_.position = np.array([[2, 3, 6]])

    # Assert that the the mean z coordinate is calculated correctly
    expected_z = 6.0
    assert sample_group.Extract_z() == expected_z

def test_extract_from_one(sample_group):
    '''
    Test case for the 'Extract_z' method of the 'group' class.

    This test checks if the 'Extract_z' method correctly calculates the mean z coordinate of the group considering 
    two particles of one single type of atom in the group.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class containing only one atom for testing purposes.

    Raises
    ------
    AssertionError
        If the mean z coordinate calculated by the 'Extract_z' method does not match the expected value.
    
    Notes
    -----
    This test adds a single atom of a specific type to the group and assigns its position. Then, it calls the 
    'Extract_z' method to calculate the mean z coordinate of the group and asserts that it matches the expected 
    value.
    '''
    # Add an atom to the group for testing
    atom_ = atom(name="H", mass=1.008, id_group=1)
    sample_group.atoms = [atom_]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms
    atom_.position = np.array([[2, 3, 6], [1, 2, 2]])

    # Assert that the the mean z coordinate is calculated correctly
    expected_z = 4.0
    assert sample_group.Extract_z() == expected_z

def test_extract_from_two(sample_group):
    '''
    Test case for the 'Extract_z' method of the 'group' class.

    This test checks if the 'Extract_z' method correctly calculates the mean z coordinate of the group considering 
    two particles of two different types of atoms in the group.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class containing atoms of different types for testing purposes.

    Raises
    ------
    AssertionError
        If the mean z coordinate calculated by the 'Extract_z' method does not match the expected value.
    
    Notes
    -----
    This test adds a single atom of a specific type to the group and assigns its position. Then, it calls the 
    'Extract_z' method to calculate the mean z coordinate of the group and asserts that it matches the expected 
    value.
    '''
    # Add an atom to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 6], [1, 2, 2]])
    atom1.position = np.array([[2, 3, 8], [1, 2, 0]])

    # Assert that the the mean z coordinate is calculated correctly
    expected_z = 4.0
    assert sample_group.Extract_z() == expected_z

def test_extract_from_zero(sample_group):
    '''
    Test case for the 'Extract_z' method of the 'group' class.

    This test checks if the 'Extract_z' method correctly calculates the mean z coordinates of the group considering 
    two particles of two types of atoms in the group but with z=0.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated mean z coordinate does not match the expected value.

    Notes
    -----
    This test adds two atoms of different types to the group and assigns their positions with z=0. Then, it calls 
    the 'Extract_z' method to calculate the mean z coordinate of the group and asserts that it matches the expected 
    value.
    '''
    # Add atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms with z=0
    atom1.position = np.array([[2, 3, 0], [1, 2, 0]])
    atom1.position = np.array([[2, 3, 0], [1, 2, 0]])

    # Assert that the mean z coordinate is calculated correctly
    assert sample_group.Extract_z() == 0.0

def test_extract_from_negative(sample_group):
    '''
    Test case for the 'Extract_z' method of the 'group' class.

    This test checks if the 'Extract_z' method correctly calculates the mean z coordinates of the group considering 
    two particles of two types of atoms in the group but with negative z.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated mean z coordinate does not match the expected value.

    Notes
    -----
    This test adds two atoms of different types to the group and assigns their positions with z values including 
    negative ones. Then, it calls the 'Extract_z' method to calculate the mean z coordinate of the group and asserts 
    that it matches the expected value.
    '''
    # Add atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    sample_group.atoms = [atom1, atom2]
    sample_group.id_tot = ['1', '2']

    # Set up some dummy positions for the atoms with negative z values
    atom1.position = np.array([[2, 3, -2], [1, 2, 1]])
    atom1.position = np.array([[2, 3, 4], [1, 2, -5]])

    # Assert that the mean z coordinate is calculated correctly
    expected_z = -0.5
    assert sample_group.Extract_z() == expected_z


# Generate tests --------------------------------------------------------------------------------------------------

def test_generation_Ek(sample_group):
    '''
    Test case for the 'Generate' method of the 'group' class.

    This test checks if the 'Generate' method correctly calculates kinetic energy at different timesteps.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the default value for kinetic energy in the first time step is not 0.
    AssertionError
        If the calculated kinetic energy does not match the expected kinetic energy.

    Notes
    -----
    This test initializes multiple atoms with different masses, assigns their positions, and then calculates the 
    kinetic energy at different timesteps using the 'Generate' method.
    Due to the calculation for the velocity, for the first timestep the kinetic energy must be 0, and then only in 
    the second timestep it should be calculated using the velocity. This is due to the fact that to calculate the 
    velocity we need the position of the current timestep and two before, so we need the starting position and the 
    second timestep.
    '''

    # Add atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    atom3 = atom(name="C", mass=12.011, id_group=1 )
    sample_group.atoms = [atom1, atom2, atom3]
    sample_group.id_tot = ['1', '2', '3']

    # Set up initial positions for the atoms
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Set up the transformation matrix
    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity

    # Perform the transformation and calculate kinetic energy at the first timestep
    sample_group.Generate(dt=1, matrix=_matrix)
    expected_ek = 0
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

    # Reset the atomic positions
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Perform the transformation and calculate kinetic energy at the second timestep
    sample_group.Generate(dt=1, matrix=_matrix)
    expected_ek = 0.5 * (1.008 * np.linalg.norm([-0.5, 1.5, 2]) ** 2 + 15.999 * np.linalg.norm([-2.5, 1, -1]) ** 2 + 12.011 * np.linalg.norm([3, -2, -1]) ** 2) * 0.0001036426948415943
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

def test_generation_T(sample_group):
    '''
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly calculates temperature at different timesteps.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated temperature at the first timestep does not match the expected value.
    AssertionError
        If the calculated temperature at the last timestep does not match the expected value.

    Notes
    -----
    This test initializes multiple atoms with different masses, assigns their positions, and then calculates the 
    temperature at different timesteps using the 'Generate' method.
    The first temperature is expected to be zero, and subsequent temperatures are calculated based on the kinetic 
    energy of the group's atoms. The temperature is computed using the formula: 

    .. math::

        T = \\frac{2 E_k}{DOF \\times K_B}

    where:
    - \( E_k \) is the kinetic energy.
    - \( DOF \) is the number of degrees of freedom.
    - \( K_B \) is the Boltzmann constant in eV/K.
    '''
    # Add atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    atom3 = atom(name="C", mass=12.011, id_group=1)
    sample_group.atoms = [atom1, atom2, atom3]
    sample_group.id_tot = ['1', '2', '3']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that the first temperature is zero
    expected_t = 0
    assert np.isclose(sample_group.T[0], expected_t, atol=1e-1)

    # Set up the positions for the atoms again
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)
    
    # Assert that the temperature is calculated correctly
    expected_t = (1.008 * np.linalg.norm([-0.5, 1.5, 2], axis=0) ** 2 + 15.999 * np.linalg.norm([-2.5, 1, -1], axis=0) ** 2 + 12.011 * np.linalg.norm([3, -2, -1]) ** 2) * 0.0001036426948415943 / (9 * 8.617333262145e-5)
    assert np.isclose(sample_group.T[-1], expected_t, atol=2e-1)

def test_forces(sample_group):
    '''
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly calculates the total forces and stores them at different 
    timesteps.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated total force vector at a timestep does not match the expected value.
    AssertionError
        If the stored total force vector does not match the expected value.

    Notes
    -----
    This test initializes multiple atoms with different masses and assigns their positions. It also provides forces 
    for each atom. Then, it calls the 'Generate' method to calculate the total forces acting on the group at 
    different timesteps.
    The total force at each timestep must be the sum of the forces acting on each atom. The calculated total force 
    vector and the stored total force vector are then compared with the expected values.

    '''
    # Add atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom2 = atom(name="O", mass=15.999, id_group=1)
    atom3 = atom(name="C", mass=12.011, id_group=1 )
    sample_group.atoms = [atom1, atom2, atom3]
    sample_group.id_tot = ['1', '2', '3']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Provide forces for each atom
    sample_group.force = [[1, -3, 2], [4, 2, -3], [-2, 3, 5]]

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that the total force vector is calculated correctly
    expected_Ftot = np.array([3, 2, 4])
    assert np.array_equal(sample_group.Ftot, expected_Ftot)

    # Assert that the total force vector is stored correctly
    expected_Ftot_store = np.array([3, 2, 4]).reshape(1, 3)
    assert np.array_equal(sample_group.Ftot_store, expected_Ftot_store)

def test_body(sample_group):
    '''
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly defines the line of the xyz output file at different 
    timesteps.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the calculated body lines for the xyz output file do not match the expected lines.

    Notes
    -----
    This test initializes an atom with predefined attributes and assigns its positions and forces at different 
    timesteps. Then, it calls the 'Generate' method to define the line of the xyz output file.
    The expected body lines are compared with the calculated body lines to ensure that they match.
    '''
    # Add an atom to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom1.N = 2
    sample_group.atoms = [atom1]
    sample_group.id_tot = ['1']

    # Set up some dummy positions and forces for the atom at different timesteps
    atom1.position = np.array([[2, 3, 6], [3, 2, -1]])
    atom1.position_past = np.array([[1, 1.5, 3], [2, 1.5, 0]])
    atom1.position_past_past = np.array([[0, 0, 0], [1, 1, 1]])
    atom1.force = [[1, -3, 2], [0, 2, 4]]
    sample_group.force = [[1, -3, 2], [0, 2, 4]]

    # Define the transformation matrix
    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    calculated_body, d = sample_group.Generate(dt=1, matrix=_matrix)

    # Define the expected body lines for the xyz output file
    expected_body = np.array(['H\t  0.0\t  -1.0\t  6.0\t  0.0\t  0.0\t  0.0\t  1\t  -3\t  2\t  1', 
                     'H\t  -1.0\t  2.0\t  -1.0\t  0.0\t  0.0\t  0.0\t  0\t  2\t  4\t  1'])
                     
    assert np.array_equal(calculated_body, expected_body)

def test_array_reset(sample_group):
    '''
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly resets or updates the storing array in the group instance 
    at each timestep.

    Parameters
    ----------
    sample_group : group
        An instance of the 'group' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the position_past_past attribute of the atom is not updated correctly.
    AssertionError
        If the position_past attribute of the atom is not updated correctly.
    AssertionError
        If the position attribute of the atom is not reset correctly.
    AssertionError
        If the force attribute of the atom is not reset correctly.
    AssertionError
        If the force attribute of the group is not reset correctly.

    Notes
    -----
    This test initializes an atom with predefined attributes and assigns its positions and forces at a specific 
    timestep. Then, it calls the 'Generate' method to update the storing arrays in the group instance.
    The test checks if the position_past_past, position_past, position, and force attributes of the atom, as well 
    as the force attribute of the group, are reset or updated correctly after calling the 'Generate' method.
    '''
    # Add an atom to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    sample_group.atoms = [atom1]
    sample_group.id_tot = ['1']

    # Set up some dummy positions and forces for the atom
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past = np.array([[1, 1.5, 3]])
    atom1.position_past_past = np.array([[0, 0, 0]])
    sample_group.force = [[1, -3, 2]]

    # Define the transformation matrix
    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that position_past_past is updated correctly
    expected_position_past_past = np.array([[1, 1.5, 3]])
    assert np.array_equal(atom1.position_past_past, expected_position_past_past)

    # Assert that position_past is updated correctly
    expected_position_past = np.array([[2, 3, 6]])
    assert np.array_equal(atom1.position_past, expected_position_past)

    # Assert that position is reset correctly
    expected_position = np.array([], dtype=float).reshape(0, 3)
    assert np.array_equal(atom1.position, expected_position)

    # Assert that atom.force is reset correctly
    expected_force = np.array([], dtype=float).reshape(0, 3)
    assert np.array_equal(atom1.force, expected_force)

    # Assert that group.force is reset correctly
    assert np.array_equal(sample_group.force, expected_force)
