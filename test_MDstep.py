import pytest
import numpy as np
from Codes.class_MDstep import MDstep
from Codes.class_graph import graph
from Codes.class_group import group


@pytest.fixture
def sample_MDstep1():
    """
    Fixture to create a sample instance of MDstep for testing purposes.

    This fixture creates an instance of MDstep with sample data for testing purposes.
    It creates a single group instance and initializes an MDstep instance with it.

    Returns
    -------
    MDstep
        A fixture providing a sample instance of MDstep for testing.
    """
    # Create an instance of group with sample data.
    group1 = group(type=['H', 'C'], id_group=0)
    
    # Initialize an MDstep instance with the created group.
    return MDstep([group1])

@pytest.fixture
def sample_MDstep2():
    """
    Fixture to create a sample instance of MDstep for testing purposes.

    This fixture creates an instance of MDstep with multiple sample groups for testing purposes.
    It creates two sample group instances and initializes an MDstep instance with them to simulate a system with 
    more than one group.

    Returns
    -------
    MDstep
        A fixture providing a sample instance of MDstep for testing.
    """
    # Create sample instances of group with sample data.
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Initialize an MDstep instance with the created groups.
    return MDstep([group1, group2])


# Inizialization_test ---------------------------------------------------------------------------------------------

def test_initialization1(sample_MDstep1):
    """
    Test case for the initialization of an MDstep instance with a single group.
    This test verifies that when an MDstep instance is initialized with a single group,
    its attributes are correctly set.

    Parameters
    ----------
    sample_MDstep1 : MDstep
        A fixture providing a sample instance of MDstep for testing.

    Raises
    ------
    AssertionError
        If any of the attributes of the sample MDstep instance do not match the expected values.
    """
    # Verify attribute values of the sample MDstep instance.
    assert np.array_equal(sample_MDstep1.groups[0].type, ['H', 'C'])
    assert sample_MDstep1.groups[0].id_group == 0
    assert sample_MDstep1.n_atoms == 0
    assert sample_MDstep1.n_type == 0
    assert np.array_equal(sample_MDstep1.ax, np.zeros(3))
    assert np.array_equal(sample_MDstep1.ay, np.zeros(3))
    assert np.array_equal(sample_MDstep1.az, np.zeros(3))
    assert sample_MDstep1.U_pot == 0.0
    assert sample_MDstep1.dt == 0.0
    assert sample_MDstep1.N_iteration == 0
    assert sample_MDstep1.alat_to_angstrom == 0.0
    assert sample_MDstep1.Ryau_to_pN == 41193.647367103644
    assert np.array_equal(sample_MDstep1.matrix, [])

def test_initialization2(sample_MDstep2):
    """
    Test case for the initialization of an MDstep instance with multiple groups.

    This test verifies that when an MDstep instance is initialized with multiple groups,
    its attributes are correctly set.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If any of the attributes of the sample MDstep instance do not match the expected values.
    """
    # Verify attribute values of the sample MDstep instance.
    assert np.array_equal(sample_MDstep2.groups[0].type, ['H', 'C'])
    assert np.array_equal(sample_MDstep2.groups[1].type, ['Fe'])
    assert sample_MDstep2.groups[0].id_group == 0
    assert sample_MDstep2.groups[1].id_group == 1
    assert sample_MDstep2.n_atoms == 0
    assert sample_MDstep2.n_type == 0
    assert np.array_equal(sample_MDstep2.ax, np.zeros(3))
    assert np.array_equal(sample_MDstep2.ay, np.zeros(3))
    assert np.array_equal(sample_MDstep2.az, np.zeros(3))
    assert sample_MDstep2.U_pot == 0.0
    assert sample_MDstep2.dt == 0.0
    assert sample_MDstep2.N_iteration == 0
    assert sample_MDstep2.alat_to_angstrom == 0.0
    assert sample_MDstep2.Ryau_to_pN == 41193.647367103644
    assert np.array_equal(sample_MDstep2.matrix, [])


# set_mass test ---------------------------------------------------------------------------------------------------
    
def test_set_mass1(sample_MDstep1):
    """
    Test case for the 'set_mass' method of the 'MDstep' class.

    This test checks if the 'set_mass' method correctly sets the mass for the corresponding atom
    in the group instances of the MDstep object, considering only one group.

    Parameters
    ----------
    sample_MDstep1 : MDstep
        A fixture providing a sample instance of MDstep with a single group for testing.

    Raises
    ------
    AssertionError
        If the mass is not set correctly in the group.
    """
    # Set mass for atoms of specific types in the group.
    sample_MDstep1.set_mass('H', 1.008)
    sample_MDstep1.set_mass('C', 12.011)

    # Assert that the mass has been set correctly in the group.
    assert sample_MDstep1.groups[0].atoms[0].mass == 1.008
    assert sample_MDstep1.groups[0].atoms[1].mass == 12.011

def test_set_mass2(sample_MDstep2):
    """
    Test case for the 'set_mass' method of the 'MDstep' class.

    This test checks if the 'set_mass' method correctly sets the mass for the corresponding atom
    in the group instances of the MDstep object, considering two groups.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the mass is not set correctly in any of the groups.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    sample_MDstep2.set_mass('Fe', 55.845)

    # Assert that the mass has been set correctly in the groups.
    assert sample_MDstep2.groups[0].atoms[0].mass == 1.008
    assert sample_MDstep2.groups[0].atoms[1].mass == 12.011
    assert sample_MDstep2.groups[1].atoms[0].mass == 55.845


# count_group test ------------------------------------------------------------------------------------------------

def test_count_group1_1(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'MDstep' class.

    This test checks the 'count_group' method for correctly counting the number of atoms
    and setting the 'position_past' attribute for an MDstep object with one group and one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the count is not work correctly for the atom in the group.
    AssertionError
        If the position_past is not set correctly for the atom in the group.

    Notes
    -----
    This test initializes one atom and assigns their positions at specific timesteps.
    Then, it calls the 'count_group' method to update the storing array in the group instance.
    The test checks if the number of atoms in the group and the initial positions are set correctly.
    """
    # Set the mass of hydrogen and the conversion factor.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate a line read from the file.
    line = '        1           H tau(   1) = (   0  1   2  )'.split()
    sample_MDstep2.count_group(line)

    # Verify that the number of atoms in the atoms in the group has been incremented correctly.
    expected_N = 1
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N

    # Verify that the initial position has been extracted correctly.
    expected_position_past = np.array([[0.0, 1.0, 2.0]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past)

def test_count_group1_2(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'MDstep' class.

    This test checks the 'count_group' method for counting correctly the number of atoms
    and setting the 'position_past' for an iteration object with one group with two identical atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the number of atoms in the group is not incremented correctly.
    AssertionError
        If the initial position is not extracted correctly.

    Notes
    -----
    This test initializes two atoms with identical attributes and assigns their positions at specific timesteps.
    Then, it calls the 'count_group' method twice to update the storing arrays in the group instance.
    The test checks if the number of atoms in the group and the initial positions are set correctly.
    """
    # Set the mass of hydrogen and the conversion factor.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate lines read from the file.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           H tau(   1) = (   10  -3   4  )'.split()

    # Call the 'count_group' method twice to update the storing arrays in the group instance.
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Verify that the number of atoms in the group has been incremented correctly.
    expected_N = 2
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N

    # Verify that the initial positions have been extracted correctly.
    expected_position_past = np.array([[0.0, 1.0, 2.0], [10, -3, 4]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past)
 
def test_count_group1_2b(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'MDstep' class.

    This test checks the 'count_group' method for counting correctly the number of atoms
    and setting the 'position_past' for an iteration object with one group with two different atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the number of atoms in each atom type of the group is not incremented correctly.

    Notes
    -----
    This test initializes two atoms with different attributes and assigns their positions at specific timesteps.
    Then, it calls the 'count_group' method twice to update the storing arrays in the group instance.
    The test checks if the number of atoms in each atom type of the group is set correctly.
    """
    # Set the mass of hydrogen and carbon and the conversion factor.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate lines read from the file.
    line1 = '        12           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        1           C tau(   1) = (   10  -3   4  )'.split()

    # Call the 'count_group' method twice to update the storing arrays in the group instance.
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Verify that the number of atoms in each atom type of the group has been incremented correctly.
    expected_N = 1
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N
    assert sample_MDstep2.groups[0].atoms[1].N == expected_N
    
    # Verify that the initial position has been extracted correctly
    expected_position_past1 = np.array([[0.0, 1.0, 2.0]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past1)

    expected_position_past2 = np.array([ [10, -3, 4]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].position_past, expected_position_past2)

def test_count_group2_2(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'MDstep' class.

    This test checks the 'count_group' method for counting correctly the number of atoms
    and setting the 'position_past' for an iteration object with two groups, each with one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the number of atoms in each group is not incremented correctly.
    AssertionError
        If the initial positions are not extracted correctly.

    Notes
    -----
    This test initializes two atoms with different attributes and assigns their positions at specific timesteps.
    Then, it calls the 'count_group' method twice to update the storing arrays in the group instance.
    The test checks if the number of atoms in each group and the initial positions are set correctly.
    """
    # Set the mass of hydrogen and iron and the conversion factor.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate lines read from the file.
    line1 = '        12           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        1           Fe tau(   1) = (   10  -3   4  )'.split()

    # Call the 'count_group' method twice to update the storing arrays in the group instance.
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Verify that the number of atoms in each group has been incremented correctly.
    expected_N = 1
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N
    assert sample_MDstep2.groups[1].atoms[0].N == expected_N

    # Verify that the initial positions have been extracted correctly.
    expected_position_past1 = np.array([[0.0, 1.0, 2.0]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past1)

    expected_position_past2 = np.array([[10, -3, 4]])
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].position_past, expected_position_past2)


# set_DOF test ----------------------------------------------------------------------------------------------------
    
def test_set_DOF(sample_MDstep2):
    """
    Test case for the 'set_DOF' method of the 'MDstep' class.

    This test checks the 'set_DOF' method for correctly setting the degrees of freedom
    for the groups in the iteration object.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the degrees of freedom are not calculated correctly for any of the groups.

    Notes
    -----
    This test initializes multiple atoms with specific attributes and assigns their positions at specific timesteps.
    Then, it calls the 'set_DOF' method to calculate the degrees of freedom for each group.
    The test checks if the calculated degrees of freedom match the expected values for each group.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    sample_MDstep2.set_mass('C', 12.011)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate lines read from the file.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        2           Fe tau(   1) = (   10  -3   4  )'.split()
    line3 = '        3           C tau(   1) = (   7   -5   12  )'.split()
    line4 = '        4           Fe tau(   1) = (   8   6   -8  )'.split()
    line5 = '        5           Fe tau(   1) = (   -4.5   -3.3  6.6  )'.split()
    line6 = '        6           H tau(   1) = (   1   1.2   -2.1  )'.split()

    # Call the 'count_group' method to simulate the presence of atoms in the groups.
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)
    sample_MDstep2.count_group(line3)
    sample_MDstep2.count_group(line4)
    sample_MDstep2.count_group(line5)
    sample_MDstep2.count_group(line6)

    # Call the 'set_DOF' method to calculate the degrees of freedom.
    sample_MDstep2.set_DOF()

    # Verify that the degrees of freedom have been calculated correctly.
    assert sample_MDstep2.groups[0].DOF == 9
    assert sample_MDstep2.groups[1].DOF == 6


# forces test -----------------------------------------------------------------------------------------------------
    
def test_forces1_1(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'MDstep' class.

    This test checks the 'forces' method for correctly storing the force for an iteration object
    with one group with one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the force is not correctly added to the group or the atom.

    Notes
    -----
    This test initializes one atom with specific attributes and assigns its position at a specific timestep.
    Then, it simulates reading a line from the file containing the force information for the atom.
    After calling the 'forces' method, the test verifies if the force has been correctly added to the group
    and the corresponding atom.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)

    # Simulate the presence of an atom in the group.
    line_count = '        1           H tau(   1) = (   0  1   2  )'.split()
    sample_MDstep2.count_group(line_count)

    # Simulate a line read from the file.
    line_force = 'atom    1 type  1   force =    0.   1.    2.'.split()
    sample_MDstep2.forces(line_force)

    # Verify that the force has been correctly added to the group and atom.
    expected_force = np.multiply(np.array([[0.0, 1.0, 2.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_force)

def test_forces1_2(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'MDstep' class.

    This test checks the 'forces' method for correctly storing the force for an iteration object
    with one group with two identical atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the force is not correctly added to the group or the atom.

    Notes
    -----
    This test initializes two identical atoms with the same attributes and assigns their positions at specific timesteps.
    Then, it simulates reading lines from the file containing the force information for each atom.
    After calling the 'forces' method, the test verifies if the force has been correctly added to the group
    and the corresponding atoms.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)

    # Simulate the presence of two identical atoms in the group.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           H tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Simulate lines read from the file.
    line1_force = 'atom    1 type  1   force =    0.   1.    2.'.split()
    line2_force = 'atom    12 type  1   force =    5.   -2.    1.'.split()
    sample_MDstep2.forces(line1_force)
    sample_MDstep2.forces(line2_force)

    # Verify that the force has been correctly added to the group and atom.
    expected_force = np.multiply(np.array([[0.0, 1.0, 2.0], [5.0, -2.0, 1.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_force)

def test_forces1_2b(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'MDstep' class.

    This test checks the 'forces' method for correctly storing the force for an iteration object
    with one group containing two different atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the force is not correctly added to the group or the atoms.

    Notes
    -----
    This test initializes two atoms with specific attributes and assigns their positions at specific timesteps.
    Then, it calls the 'forces' method twice to store the forces for each atom.
    The test checks if the forces are correctly added to the group and each atom.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)

    # Simulate the presence of atoms in the group.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           C tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Simulate lines read from the file.
    line_force1 = 'atom    1 type  1   force =    0.   1.    2.'.split()
    line_force2 = 'atom    12 type  1   force =    5.   -2.    1.'.split()
    sample_MDstep2.forces(line_force1)
    sample_MDstep2.forces(line_force2)

    # Verify that the forces have been correctly added to the group and atoms.
    expected_group_force = np.multiply(np.array([[0.0, 1.0, 2.0], [5.0, -2.0, 1.0]]), sample_MDstep2.Ryau_to_pN)
    expected_H_force = np.multiply(np.array([[0.0, 1.0, 2.0]]), sample_MDstep2.Ryau_to_pN)
    expected_C_force = np.multiply(np.array([[5.0, -2.0, 1.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_group_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_H_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].force, expected_C_force)

def test_count_group2_2(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'MDstep' class.

    This test checks the 'forces' method for correctly storing the force for an iteration object
    with two groups, each containing one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the force is not correctly added to the group or the atoms.

    Notes
    -----
    This test initializes two atoms with different attributes and assigns their positions at specific timesteps.
    Then, it calls the 'forces' method twice to update the storing arrays in the group instance.
    The test checks if the force is correctly added to each group and atom.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)

    # Simulate the presence of atoms in the groups.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           Fe tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Simulate lines read from the file.
    line_force1 = 'atom    1 type  1   force =    0.   1.    2.'.split()
    line_force2 = 'atom    12 type  1   force =    5.   -2.    1.'.split()
    sample_MDstep2.forces(line_force1)
    sample_MDstep2.forces(line_force2)

    # Verify that the force has been correctly added to each group and atom.
    expected_H_force = np.multiply(np.array([[0.0, 1.0, 2.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_H_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_H_force)

    expected_Fe_force = np.multiply(np.array([[5.0, -2.0, 1.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[1].force, expected_Fe_force)
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].force, expected_Fe_force)


# positions test in angstrom --------------------------------------------------------------------------------------
    
def test_positions_RDF_store1(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.

    This test checks the 'positions' method for correctly storing the position in both the atom and RDF object,
    considering an iteration object with one group with one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        A fixture providing a sample instance of MDstep with multiple groups for testing.

    Raises
    ------
    AssertionError
        If the position is not correctly added to the atom or the RDF object.

    Notes
    -----
    This test simulates the reading of a line from a file and calls the 'positions' method to store the position
    both in the atom and the RDF object. It then verifies that the position has been correctly added.
    """
    # Set mass for atoms of specific types in the groups.
    sample_MDstep2.set_mass('H', 1.008)

    # Create a sample graph object.
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate a line read from the file.
    line = ['H', '2.0', '3.0', '4.0']

    # Call the 'positions' method to store the position in both the atom and RDF object.
    sample_MDstep2.positions(line, graphs, 1)

    # Verify that the position has been correctly added to the atom.
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0]]))

    # Verify that the position has been correctly added to the graph.
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[2.0, 3.0, 4.0]]))

def test_positions_RDF_store2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.

    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep object 
    with one group containing two identical atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the position is not correctly stored in the atom.
    AssertionError
        If the position is not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including one group with 
    two identical atoms of type 'H'. Then, it simulates reading two lines of positions from a file and calls the 
    'positions' method to store the positions in both the atom and RDF object. Finally, it verifies that the 
    positions are correctly stored in both the atom and RDF object by comparing them with the expected positions.
    """
    # Set the mass for atom type 'H'
    sample_MDstep2.set_mass('H', 1.008)

    # Create a graph object
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate lines read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['H', '1', '-2', '3']

    # Call the positions method to store positions
    sample_MDstep2.positions(line1, graphs, 1)
    sample_MDstep2.positions(line2, graphs, 1)

    # Verify that the position is correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0], [1.0, -2.0, 3.0]]))

    # Verify that the position is correctly stored in the RDF object
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0], [1.0, -2.0, 3.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[2.0, 3.0, 4.0], [1.0, -2.0, 3.0]]))

def test_positions_RDF_store2b(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.

    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep object 
    with one group containing two different atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the position is not correctly stored in the atom.
    AssertionError
        If the position is not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including one group with 
    two different atoms ('H' and 'C'). Then, it simulates reading two lines of positions from a file and calls the 
    'positions' method to store the positions in both the atom and RDF object. Finally, it verifies that the 
    positions are correctly stored in both the atom and RDF object by comparing them with the expected positions.
    """
    # Set the mass for atom types 'H' and 'C'
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)

    # Create a graph object with two types of interactions: H-H and C-H
    graphs = graph("test_file", [5, 5], [['H', 'H'], ['C', 'H']], [100, 100], "output")

    # Simulate lines read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['C', '1', '-2', '3']

    # Call the positions method to store positions for the first atom
    sample_MDstep2.positions(line1, graphs, 1)
    # Verify that the position is correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0]]))

    # Call the positions method to store positions for the second atom
    sample_MDstep2.positions(line2, graphs, 1)
    # Verify that the position is correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].position, np.array([[1.0, -2.0, 3.0]]))
    
    # Verify that the positions are correctly stored in the RDF object for both types of interactions
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[1].at1, np.array([[1.0, -2.0, 3.0]]))
    assert np.array_equal(graphs.type[1].at2, np.array([[2.0, 3.0, 4.0]]))

def test_positions_RDF_store2_2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.

    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep object
    with two groups, each containing one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the position is not correctly stored in the atom.
    AssertionError
        If the position is not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including one group with 
    an 'H' atom and one group with an 'Fe' atom. Then, it simulates reading two lines of positions from a file and 
    calls the 'positions' method to store the positions in both the atom and RDF object. Finally, it verifies that 
    the positions are correctly stored in both the atom and RDF object by comparing them with the expected positions.
    """
    # Set the mass for atom types 'H' and 'Fe'
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)

    # Create a graph object with one type of interaction: H-Fe
    graphs = graph("test_file", [5], [['H', 'Fe']], [100], "output")

    # Simulate lines read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['Fe', '1', '-2', '3']

    # Call the positions method to store positions for the first atom
    sample_MDstep2.positions(line1, graphs, 1)
    # Verify that the position is correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0]]))

    # Call the positions method to store positions for the second atom
    sample_MDstep2.positions(line2, graphs, 1)
    # Verify that the position is correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].position, np.array([[1.0, -2.0, 3.0]]))
    
    # Verify that the positions are correctly stored in the RDF object
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, -2.0, 3.0]]))


# positions test in alat ------------------------------------------------------------------------------------------
    
def test_positions_RDF_alat_store1(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.

    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep object 
    with one group containing one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the position is not correctly stored in the atom.
    AssertionError
        If the position is not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including one group with 
    an 'H' atom. Then, it simulates reading a line of positions from a file and calls the 'positions' method to 
    store the position in both the atom and RDF object. Finally, it verifies that the position is correctly stored 
    in both the atom and RDF object by comparing it with the expected position.
    """
    # Set the mass for the 'H' atom type
    sample_MDstep2.set_mass('H', 1.008)
    
    # Create a graph object with one type of interaction: H-H
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate a line read from the file
    line = ['H', '2.0', '3.0', '4.0']

    # Call the positions method to store the position
    sample_MDstep2.positions(line, graphs, 0.5)
    
    # Verify that the position is correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0]]))
    
    # Verify that the position is correctly stored in the RDF object
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, 1.5, 2.0]]))

def test_positions_RDF_alat_store2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.
    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep object
    with one group containing two identical atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the positions are not correctly stored in the atom.
    AssertionError
        If the positions are not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including one group with 
    two identical 'H' atoms. Then, it simulates reading lines of positions from a file and calls the 'positions' 
    method to store the positions in both the atom and RDF object. Finally, it verifies that the positions are 
    correctly stored in both the atom and RDF object by comparing them with the expected positions.
    """
    # Set the mass for the 'H' atom type
    sample_MDstep2.set_mass('H', 1.008)
    
    # Create a graph object with one type of interaction: H-H
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate lines read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['H', '1', '-2', '3']

    # Call the positions method to store the positions
    sample_MDstep2.positions(line1, graphs, 0.5)
    sample_MDstep2.positions(line2, graphs, 0.5)
    
    # Verify that the positions are correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0], [0.5, -1.0, 1.5]]))
    
    # Verify that the positions are correctly stored in the RDF object
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0], [0.5, -1.0, 1.5]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, 1.5, 2.0], [0.5, -1.0, 1.5]]))

def test_positions_RDF_alat_store2b(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.
    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep object 
    with one group containing two different atoms.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the positions are not correctly stored in the atom.
    AssertionError
        If the positions are not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including one group with 
    two different atom types ('H' and 'C'). Then, it simulates reading lines of positions from a file and calls the
    'positions' method to store the positions in both the atom and RDF object. Finally, it verifies that the 
    positions are correctly stored in both the atom and RDF object by comparing them with the expected positions.
    """
    # Set the mass for the 'H' and 'C' atom types
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    
    # Create a graph object with two types of interactions: H-H and C-H
    graphs = graph("test_file", [5, 5], [['H', 'H'], ['C', 'H']], [100, 100], "output")

    # Simulate lines read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['C', '1', '-2', '3']

    # Call the positions method to store the positions
    sample_MDstep2.positions(line1, graphs, 0.5)
    sample_MDstep2.positions(line2, graphs, 0.5)
    
    # Verify that the positions are correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].position, np.array([[0.5, -1.0, 1.5]]))
    
    # Verify that the positions are correctly stored in the RDF object
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[1].at1, np.array([[0.5, -1.0, 1.5]]))
    assert np.array_equal(graphs.type[1].at2, np.array([[1.0, 1.5, 2.0]]))

def test_positions_RDF_alat_store2_2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'MDstep' class.
    This test checks the correct storing of positions in both the atom and RDF object, considering an MDstep 
    object with two groups, each containing one atom.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes.

    Raises
    ------
    AssertionError
        If the positions are not correctly stored in the atom.
    AssertionError
        If the positions are not correctly stored in the RDF object.

    Notes
    -----
    This test initializes an MDstep object and a graph object with predefined attributes, including two groups
    with one atom each ('H' and 'Fe'). Then, it simulates reading lines of positions from a file and calls the 
    'positions' method to store the positions in both the atom and RDF object. Finally, it verifies that the 
    positions are correctly stored in both the atom and RDF object by comparing them with the expected positions.
    """
    # Set the mass for the 'H' and 'Fe' atom types
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    
    # Create a graph object with one type of interaction: H-Fe
    graphs = graph("test_file", [5], [['H', 'Fe']], [100], "output")

    # Simulate lines read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['Fe', '1', '-2', '3']

    # Call the positions method to store the positions
    sample_MDstep2.positions(line1, graphs, 0.5)
    sample_MDstep2.positions(line2, graphs, 0.5)
    
    # Verify that the positions are correctly stored in the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].position, np.array([[0.5, -1.0, 1.5]]))
    
    # Verify that the positions are correctly stored in the RDF object
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[0.5, -1.0, 1.5]]))


# single_frame test ----------------------------------------------------------------------------------------------

def test_single_frame(sample_MDstep2, monkeypatch):
    """
    Test case for the 'single_frame' method of the 'MDstep' class.

    This test verifies the generation of a single output frame, including lattice information, timestep parameters,
    potential energy, kinetic energy, degrees of freedom, temperature, total force, and other relevant data.

    Parameters
    ----------
    sample_MDstep2 : MDstep
        An instance of the 'MDstep' class with predefined attributes and groups.

    Notes
    -----
    This test sets up a sample MDstep object with predefined attributes for testing. Two groups within the MDstep object
    are also preconfigured with kinetic energy, degrees of freedom, temperature, and total force values.
    The test then calls the 'single_frame' method of the MDstep object to generate a single output frame and verifies
    the correctness of the generated frame.

    Raises
    ------
    AssertionError
        If the generated output frame does not match the expected output format or values.
    """
    # Set up attributes and groups for testing
    sample_MDstep2.n_atoms = 4
    sample_MDstep2.ax = [2, 0, 0]
    sample_MDstep2.ay = [0, 4, 0]
    sample_MDstep2.az = [0, 0, 1]
    sample_MDstep2.dt = 0.001000
    sample_MDstep2.N_iteration = 2
    sample_MDstep2.U_pot = 13.60570398

    # Set up attributes and data for the first group
    gr1 = sample_MDstep2.groups[0]
    gr1.Ek = 1
    gr1.DOF = 2
    gr1.T = [3]
    gr1.force = [[4, 5, 6]]
    gr1.distance_switch = True

    # Set up attributes and data for the second group
    gr2 = sample_MDstep2.groups[1]
    gr2.Ek = 10
    gr2.DOF = 20
    gr2.T = [30]
    gr2.force = [[40, 50, 60]]
    gr2.distance_switch = True
    
    # Generate a single output frame
    text = sample_MDstep2.single_frame()

    # Verify that the output format is correct (you can specify an expected output)
    expected_text = ['4', 'Lattice(Ang)=\"2.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 1.000\" dt(ps)=0.001000 N=2 Epot(eV)=1.000 Ek0(eV)=1.000 DOF0=2 T0(K)=11604.518 Ftot0(pN)=\"4.000, 5.000, 6.000\" Ek1(eV)=10.000 DOF1=20 T1(K)=11604.518 Ftot1(pN)=\"40.000, 50.000, 60.000\"']
    assert np.array_equal(text, expected_text)
