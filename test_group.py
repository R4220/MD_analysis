# File: test_group.py
import pytest
import numpy as np
from Codes.class_atom import atom
from Codes.class_group import group 

@pytest.fixture
def sample_group():
    return group(type=['H', 'C', 'P', 'O'], id_group=1)


# initialization test

def test_initialization(sample_group):
    """
    Test case for the initialization of the 'group' class.
    This test checks if the 'group' class is initialized correctly with the provided attributes.
    """
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


# Kinetic_energy tests
    
def test_velocity_storage(sample_group):
    """
    Test case for the 'velocity_storage' method of the 'group' class.
    This test checks if the 'velocity_storage' method correctly stores the velocities of the group's atoms.
    """
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
    """
    Test case for the 'group_velocity' method of the 'group' class.
    This test checks if the 'group_velocity' method correctly calculates the group's velocity.
    """
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
    """
    Test case for the 'group_velocity_store' method of the 'group' class.
    This test checks if the 'group_velocity_store' method correctly stores the group's velocity.
    """
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
    expected_Vtot_store = np.array([[1, 1.5, 0]], dtype=float)
    assert np.array_equal(sample_group.Vtot_store, expected_Vtot_store)

def test_kinetic_energy1(sample_group):
    """
    Test case for the 'Kinetic_energy' method of the 'group' class.
    This test checks if the 'Kinetic_energy' method correctly calculates kinetic energy for a single atom.
    """
    # Add some atoms to the group for testing
    atom1 = atom(name="H", mass=1.008, id_group=1)
    sample_group.atoms = [atom1]
    sample_group.id_tot = ['1']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[10, 0, 0]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the kinetic energy is calculated correctly
    expected_ek = 0
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

def test_kinetic_energy2(sample_group):
    """
    Test case for the 'Kinetic_energy' method of the 'group' class.
    This test checks if the 'Kinetic_energy' method correctly calculates kinetic energy for two atoms with different masses.
    """
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
    """
    Test case for the 'Kinetic_energy' method of the 'group' class.
    This test checks if the 'Kinetic_energy' method correctly calculates kinetic energy for multiple atoms with different masses.
    """
    # Add some atoms to the group for testing
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

    # Assume dt is 1 for simplicity
    sample_group.Kinetic_energy(dt=1)

    # Assert that the kinetic energy is calculated correctly
    expected_ek = 0.5 * (1.008 * np.linalg.norm([-0.5, 1.5, 2]) ** 2 + 15.999 * np.linalg.norm([-2.5, 1, -1]) ** 2 + 12.011 * np.linalg.norm([3, -2, -1]) ** 2) * 0.0001036426948415943
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)


# Generate tests  

def test_generation_Ek(sample_group):
    """
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly calculates kinetic energy at different timesteps.
    """
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

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that the kinetic energy is 0 at the first timestep
    expected_ek = 0
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

    # Set again the atomic positions
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that the kinetic energy is calculated at second timestep
    expected_ek = 0.5 * (1.008 * np.linalg.norm([-0.5, 1.5, 2]) ** 2 + 15.999 * np.linalg.norm([-2.5, 1, -1]) ** 2 + 12.011 * np.linalg.norm([3, -2, -1]) ** 2) * 0.0001036426948415943
    assert np.isclose(sample_group.Ek, expected_ek, atol=1e-4)

def test_generation_T(sample_group):
    """
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly calculates temperature at different timesteps.
    """
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

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that the firt temperature is zero
    expected_t = 0
    assert np.isclose(sample_group.T[0], expected_t, atol=1e-1)

    # Set up again the positions for the atoms
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    atom2.position = np.array([[0, 3, 1]])
    atom2.position_past_past = np.array([[2, 0, 1]])

    atom3.position = np.array([[10, 3, 1]])
    atom3.position_past_past = np.array([[1, 6, 1]])

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)
    
    # Assert that the temperature is calculated correctly
    expected_t = (1.008 * np.linalg.norm([-0.5, 1.5, 2], axis = 0) ** 2 + 15.999 * np.linalg.norm([-2.5, 1, -1], axis = 0) ** 2 + 12.011 * np.linalg.norm([3, -2, -1]) ** 2) * 0.0001036426948415943 / (9 * 8.617333262145e-5)
    assert np.isclose(sample_group.T[-1], expected_t, atol=2e-1)

def test_forces(sample_group):
    """
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly calculates the total forces and store it correctly at different timesteps.
    """
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

    sample_group.force = [[1, -3, 2], [4, 2, -3], [-2, 3, 5]]

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    # Assert that the Ftot is calculated correctly
    expected_Ftot = np.array([3, 2, 4])
    assert np.array_equal(sample_group.Ftot, expected_Ftot)

    # Assert that the total force is stored correctly
    expected_Ftot_store = np.array([3, 2, 4]).reshape(1, 3)
    assert np.array_equal(sample_group.Ftot_store, expected_Ftot_store)

def test_body(sample_group):
    """
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly define the line of the xyz output file different timesteps.
    """
    atom1 = atom(name="H", mass=1.008, id_group=1)
    atom1.N = 2
    sample_group.atoms = [atom1]#, atom2, atom3]
    sample_group.id_tot = ['1']#, '2', '3']


    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 6], [3, 2, -1]])
    atom1.position_past = np.array([[1, 1.5, 3], [2, 1.5, 0]])
    atom1.position_past_past = np.array([[0, 0, 0], [1, 1, 1]])

    atom1.force = [[1, -3, 2], [0, 2, 4]]
    sample_group.force = [[1, -3, 2], [0, 2, 4]]

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    #print('hey', type(np.linalg.inv(_matrix)))

    # Assume dt is 1 for simplicity
    calculated_body = sample_group.Generate(dt=1, matrix=_matrix)
    expected_body = ['H\t  0.0\t  -1.0\t  6.0\t  0.0\t  0.0\t  0.0\t  1\t  -3\t  2\t  1', 'H\t  -1.0\t  2.0\t  -1.0\t  0.0\t  0.0\t  0.0\t  0\t  2\t  4\t  1']
    print('expected :', expected_body)
    print('calculated :', calculated_body)
    assert np.array_equal(calculated_body, expected_body)

def test_array_reset(sample_group):
    """
    Test case for the 'Generate' method of the 'group' class.
    This test checks if the 'Generate' method correctly resets or upgrade the storing array in the group istamnce at each timestep.
    """
    atom1 = atom(name="H", mass=1.008, id_group=1)
    sample_group.atoms = [atom1]#, atom2, atom3]
    sample_group.id_tot = ['1']#, '2', '3']

    # Set up some dummy positions for the atoms
    atom1.position = np.array([[2, 3, 6]])
    atom1.position_past = np.array([[1, 1.5, 3]])
    atom1.position_past_past = np.array([[0, 0, 0]])

    sample_group.force = [[1, -3, 2]]

    _matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    sample_group.DOF = 9

    # Assume dt is 1 for simplicity
    sample_group.Generate(dt=1, matrix=_matrix)

    
    # Assert that position_past_past is changed correctly
    expected_position_past_past = np.array([[1, 1.5, 3]])
    assert np.array_equal(atom1.position_past_past, expected_position_past_past)

    # Assert that position_past is changed correctly
    expected_position_past = np.array([[2, 3, 6]])
    assert np.array_equal(atom1.position_past, expected_position_past)

    # Assert that position is changed correctly
    expected_position = np.array([], dtype=float).reshape(0, 3)
    assert np.array_equal(atom1.position, expected_position)

    # Assert that atom.force is changed correctly
    expected_force = np.array([], dtype=float).reshape(0, 3)
    assert np.array_equal(atom1.force, expected_force)

    # Assert that group.force is changed correctly
    assert np.array_equal(sample_group.force, expected_force)
