import pytest
import numpy as np
from Codes.class_MDstep import MDstep
from Codes.class_graph import graph
from Codes.class_group import group

@pytest.fixture
def sample_MDstep1():
    # Create an instance of iteration with sample data.
    group1 = group(type=['H', 'C'], id_group=0)
    return MDstep([group1])

@pytest.fixture
def sample_MDstep2():
    # Create an instance of iteration with sample data.
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    return MDstep([group1, group2])
# -----------------------------------------------------------------------------------------------------------------

# Inizialization_test

def test_initialization1(sample_MDstep1):
    """
    Test to verify that the initialization of an iteration with a single group works correctly.
    """
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
    Test to verify that the initialization of an iteration with multiple groups works correctly.
    """
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


# set_mass test
    
def test_set_mass1(sample_MDstep1):
    """
    Test case for the 'set_mass' method of the 'iteration' class.
    This test checks if the 'set_mass' method correctly sets the mass for the corresponding atom in the group instances of the iteration object, considering only one group.
    """
    sample_MDstep1.set_mass('H', 1.008)
    sample_MDstep1.set_mass('C', 12.011)

    # Assert that the mass has been set correctly in the group.
    assert sample_MDstep1.groups[0].atoms[0].mass == 1.008
    assert sample_MDstep1.groups[0].atoms[1].mass == 12.011

def test_set_mass2(sample_MDstep2):
    """
    Test case for the 'set_mass' method of the 'iteration' class.
    This test checks if the 'set_mass' method correctly sets the mass for the corresponding atom in the group instances of the iteration object, considering two groups.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    sample_MDstep2.set_mass('Fe', 55.845)

    # Assert that the mass has been set correctly in the group.
    assert sample_MDstep2.groups[0].atoms[0].mass == 1.008
    assert sample_MDstep2.groups[0].atoms[1].mass == 12.011
    assert sample_MDstep2.groups[1].atoms[0].mass == 55.845


# count_group test

def test_count_group1_1(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'iteration' class.
    Test the count_group method for counting correctly the number of atoms and setting the 'position_past ' for an iteration object with one group with one atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate a line read from the file.
    line = '        1           H tau(   1) = (   0  1   2  )'.split()
    sample_MDstep2.count_group(line)

    # Verify that the number of atoms in the atoms in the group has been incremented correctly
    expected_N = 1
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N

    # Verify that the initial position has been extracted correctly
    expected_position_past = np.array([[0.0, 1.0, 2.0]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past)

def test_count_group1_2(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'iteration' class.
    Test the count_group method for counting correctly the number of atoms and setting the 'position_past ' for an iteration object with one group with two identical atoms.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate a line read from the file.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           H tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Verify that the number of atoms in the atoms in the group has been incremented correctly
    expected_N = 2
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N

    # Verify that the initial position has been extracted correctly
    expected_position_past = np.array([[0.0, 1.0, 2.0], [10, -3, 4]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past)
    
def test_count_group1_2b(sample_MDstep2):
    """
    Test case for the 'count_group' method of the 'iteration' class.
    Test the count_group method for counting correctly the number of atoms and setting the 'position_past ' for an iteration object with one group with two different atoms.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate a line read from the file.
    line1 = '        12           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        1           C tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Verify that the number of atoms in the atoms in the group has been incremented correctly
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
    Test case for the 'count_group' method of the 'iteration' class.
    Test the count_group method for counting correctly the number of atoms and setting the 'position_past ' for an iteration object with two group with one atom each.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate a line read from the file.
    line1 = '        12           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        1           Fe tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Verify that the number of atoms in the atoms in the group has been incremented correctly
    expected_N = 1
    assert sample_MDstep2.groups[0].atoms[0].N == expected_N
    assert sample_MDstep2.groups[1].atoms[0].N == expected_N

    # Verify that the initial position has been extracted correctly
    expected_position_past1 = np.array([[0.0, 1.0, 2.0]])
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position_past, expected_position_past1)

    expected_position_past2 = np.array([ [10, -3, 4]])
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].position_past, expected_position_past2)


# set_DOF test
    
def test_set_DOF(sample_MDstep2):
    """
    Test case for the 'set_DOF' method of the 'iteration' class.
    Test the set_DOF method for an the correct setting of the degree of freedom, accordingly to the notation, for the groups in the iteration object.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    sample_MDstep2.set_mass('C', 12.011)
    sample_MDstep2.alat_to_angstrom = 1

    # Simulate a line read from the file.
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        2           Fe tau(   1) = (   10  -3   4  )'.split()
    line3 = '        3           C tau(   1) = (   7   -5   12  )'.split()
    line4 = '        4           Fe tau(   1) = (   8   6   -8  )'.split()
    line5 = '        5           Fe tau(   1) = (   -4.5   -3.3  6.6  )'.split()
    line6 = '        6           H tau(   1) = (   1   1.2   -2.1  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)
    sample_MDstep2.count_group(line3)
    sample_MDstep2.count_group(line4)
    sample_MDstep2.count_group(line5)
    sample_MDstep2.count_group(line6)

    sample_MDstep2.set_DOF()

    # Verify that the degrees of freedom have been calculated correctly.
    assert sample_MDstep2.groups[0].DOF == 9
    assert sample_MDstep2.groups[1].DOF == 6


# forces test
    
def test_forces1_1(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'iteration' class.
    Test the forces method for the correct storing of the force, considering an iteration object with one group with one atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    line = '        1           H tau(   1) = (   0  1   2  )'.split()
    sample_MDstep2.count_group(line)

    # Simulate a line read from the file.
    line = 'atom    1 type  1   force =    0.   1.    2.'.split()
    sample_MDstep2.forces(line)

    # Verify that the force has been correctly added to the group and atom
    expected_force = np.multiply(np.array([[0.0, 1.0, 2.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_force)

def test_forces1_2(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'iteration' class.
    Test the forces method for the correct storing of the force, considering an iteration object with one group with two identical atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           H tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Simulate a line read from the file.
    line1 = 'atom    1 type  1   force =    0.   1.    2.'.split()
    line2 = 'atom    12 type  1   force =    5.   -2.    1.'.split()
    sample_MDstep2.forces(line1)
    sample_MDstep2.forces(line2)

    # Verify that the force has been correctly added to the group and atom
    expected_force = np.multiply(np.array([[0.0, 1.0, 2.0], [5.0, -2.0, 1-0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_force)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_force)

def test_forces1_2b(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'iteration' class.
    Test the forces method for the correct storing of the force, considering an iteration object with one group with two different atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           C tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Simulate a line read from the file.
    line1 = 'atom    1 type  1   force =    0.   1.    2.'.split()
    line2 = 'atom    12 type  1   force =    5.   -2.    1.'.split()
    sample_MDstep2.forces(line1)
    sample_MDstep2.forces(line2)
    
    # Verify that the force has been correctly added to the group and atom
    expected_grforce = np.multiply(np.array([[0.0, 1.0, 2.0], [5.0, -2.0, 1-0]]), sample_MDstep2.Ryau_to_pN)
    expected_Hforce = np.multiply(np.array([[0.0, 1.0, 2.0]]), sample_MDstep2.Ryau_to_pN)
    expected_Cforce = np.multiply(np.array([[5.0, -2.0, 1-0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_grforce)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_Hforce)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].force, expected_Cforce)

def test_count_group2_2(sample_MDstep2):
    """
    Test case for the 'forces' method of the 'iteration' class.
    Test the forces method for the correct storing of the force, considering an iteration object with two groups with one atom each.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    line1 = '        1           H tau(   1) = (   0  1   2  )'.split()
    line2 = '        12           Fe tau(   1) = (   10  -3   4  )'.split()
    sample_MDstep2.count_group(line1)
    sample_MDstep2.count_group(line2)

    # Simulate a line read from the file.
    line1 = 'atom    1 type  1   force =    0.   1.    2.'.split()
    line2 = 'atom    12 type  1   force =    5.   -2.    1.'.split()
    sample_MDstep2.forces(line1)
    sample_MDstep2.forces(line2)

    # Verify that the force has been correctly added to the group and atom
    expected_Hforce = np.multiply(np.array([[0.0, 1.0, 2.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[0].force, expected_Hforce)
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].force, expected_Hforce)

    expected_Feforce = np.multiply(np.array([[5.0, -2.0, 1.0]]), sample_MDstep2.Ryau_to_pN)
    assert np.array_equal(sample_MDstep2.groups[1].force, expected_Feforce)
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].force, expected_Feforce)
    

# positions test in angstrom
    
def test_positions_RDF_store1(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with one groups with one atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate a line read from the file
    line = ['H', '2.0', '3.0', '4.0']

    sample_MDstep2.positions(line, graphs, 1)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0]]))
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[2.0, 3.0, 4.0]]))

def test_positions_RDF_store2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with one groups with two identical atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate a line read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['H', '1', '-2', '3']

    sample_MDstep2.positions(line1, graphs, 1)
    sample_MDstep2.positions(line2, graphs, 1)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0], [1.0, -2.0, 3.0]]))
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0], [1.0, -2.0, 3.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[2.0, 3.0, 4.0], [1.0, -2.0, 3.0]]))

def test_positions_RDF_store2b(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with one groups with two different atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    graphs = graph("test_file", [5, 5], [['H', 'H'], ['C', 'H']], [100, 100], "output")

    # Simulate a line read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['C', '1', '-2', '3']

    sample_MDstep2.positions(line1, graphs, 1)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0]]))

    sample_MDstep2.positions(line2, graphs, 1)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].position, np.array([[1.0, -2.0, 3.0]]))
    
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[1].at1, np.array([[1.0, -2.0, 3.0]]))
    assert np.array_equal(graphs.type[1].at2, np.array([[2.0, 3.0, 4.0]]))

def test_positions_RDF_store2_2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with two groups with one atom each.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    graphs = graph("test_file", [5], [['H', 'Fe']], [100], "output")

    # Simulate a line read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['Fe', '1', '-2', '3']

    sample_MDstep2.positions(line1, graphs, 1)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[2.0, 3.0, 4.0]]))

    sample_MDstep2.positions(line2, graphs, 1)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].position, np.array([[1.0, -2.0, 3.0]]))
    
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[2.0, 3.0, 4.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, -2.0, 3.0]]))


# positions test in alat
    
def test_positions_RDF_alat_store1(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with one groups with one atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate a line read from the file
    line = ['H', '2.0', '3.0', '4.0']

    sample_MDstep2.positions(line, graphs, 0.5)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0]]))
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, 1.5, 2.0]]))

def test_positions_RDF_alat_store2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with one groups with two identical atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    graphs = graph("test_file", [5], [['H', 'H']], [100], "output")

    # Simulate a line read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['H', '1', '-2', '3']

    sample_MDstep2.positions(line1, graphs, 0.5)
    sample_MDstep2.positions(line2, graphs, 0.5)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0], [0.5, -1.0, 1.5]]))
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0], [0.5, -1.0, 1.5]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, 1.5, 2.0], [0.5, -1.0, 1.5]]))

def test_positions_RDF_alat_store2b(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with one groups with two different atom.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('C', 12.011)
    graphs = graph("test_file", [5, 5], [['H', 'H'], ['C', 'H']], [100, 100], "output")

    # Simulate a line read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['C', '1', '-2', '3']

    sample_MDstep2.positions(line1, graphs, 0.5)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0]]))

    sample_MDstep2.positions(line2, graphs, 0.5)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[1].position, np.array([[0.5, -1.0, 1.5]]))
    
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[1].at1, np.array([[0.5, -1.0, 1.5]]))
    assert np.array_equal(graphs.type[1].at2, np.array([[1.0, 1.5, 2.0]]))

def test_positions_RDF_alat_store2_2(sample_MDstep2):
    """
    Test case for the 'positions' method of the 'iteration' class.
    Test the forces method for the correct storing of the position in both the atom and RDF object, considering an iteration object with two groups with one atom each.
    """
    sample_MDstep2.set_mass('H', 1.008)
    sample_MDstep2.set_mass('Fe', 55.845)
    graphs = graph("test_file", [5], [['H', 'Fe']], [100], "output")

    # Simulate a line read from the file
    line1 = ['H', '2.0', '3.0', '4.0']
    line2 = ['Fe', '1', '-2', '3']

    sample_MDstep2.positions(line1, graphs, 0.5)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[0].atoms[0].position, np.array([[1.0, 1.5, 2.0]]))

    sample_MDstep2.positions(line2, graphs, 0.5)
    # Verify that the position has been correctly added to the atom
    assert np.array_equal(sample_MDstep2.groups[1].atoms[0].position, np.array([[0.5, -1.0, 1.5]]))
    
    # Verify that the position has been correctly added to the graph
    assert np.array_equal(graphs.type[0].at1, np.array([[1.0, 1.5, 2.0]]))
    assert np.array_equal(graphs.type[0].at2, np.array([[0.5, -1.0, 1.5]]))


# single_frame test

def test_single_frame(sample_MDstep2, monkeypatch):
    sample_MDstep2.n_atoms = 4
    sample_MDstep2.ax = [2, 0, 0]
    sample_MDstep2.ay = [0, 4, 0]
    sample_MDstep2.az = [0, 0, 1]
    sample_MDstep2.dt = 0.001000
    sample_MDstep2.N_iteration = 2
    sample_MDstep2.U_pot = 13.60570398

    gr1 = sample_MDstep2.groups[0]
    gr1.Ek = 1
    gr1.DOF = 2
    gr1.T = [3]
    gr1.Ftot = [4, 5, 6]

    gr2 = sample_MDstep2.groups[1]
    gr2.Ek = 10
    gr2.DOF = 20
    gr2.T = [30]
    gr2.Ftot = [40, 50, 60]

    a =1
    b =1
    def mock_generate(gr1, a, b):
        return 'G'
    monkeypatch.setattr(group, 'Generate', mock_generate)
    
    # Generate a single output frame
    text = sample_MDstep2.single_frame()
    # Verify that the output format is correct (you can specify an expected output)
    expected_text = ['4', 'Lattice(Ang)=\"2.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 1.000\" dt(ps)=0.001000 N=2 Epot(eV)=1.000 Ek0(eV)=1.000 DOF0=2 T0(K)=3.000 Ftot0(pN)=\"4.000, 5.000, 6.000\" Ek1(eV)=10.000 DOF1=20 T1(K)=30.000 Ftot1(pN)=\"40.000, 50.000, 60.000\"', 'G', 'G']
    print('expected:\n', expected_text )
    print('\n\n text: \n', text)
    assert np.array_equal(text, expected_text)
