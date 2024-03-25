import pytest
import numpy as np
from Codes.class_MDstep import MDstep
from Codes.class_group import group
from Codes.class_graph import graph
from pwo_into_xyz import configuration, preamble, extract_forces, extract_positions, body, xyz_gen


@pytest.fixture
def sample_MDstep():
    return MDstep(groups=[])

@pytest.fixture
def mocked_input(monkeypatch):
    """
    Fixture that mocks user input for testing purposes.

    This fixture simulates user input by overriding the built-in input() function.
    It provides mock responses to read the input file for the setup.

    Parameters
    ----------
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        The pytest monkeypatch fixture.

    Returns
    -------
    function
        A function that overrides the built-in input() function to provide mock responses.
    """
    # Define logic for simulating user input
    def mock_input(prompt):
        if "Write the input file name:\n" in prompt:
            return "Test/files/Setup.ini"
        else:
            assert 1 == 0
    monkeypatch.setattr('builtins.input', mock_input)


# test config -----------------------------------------------------------------------------------------------------
    
def test_configuration(mocked_input):
    """
    Test case for the 'setup' function.
    This test checks if the 'setup' function correctly extract the values from the setup file.
    """

    filename, outdir, Rmax, atoms, N, groups, filepath = configuration()

    assert filename == 'test_file'
    assert outdir == 'Test/output'
    assert filepath == 'Test/file/test_file'

    # testing the groups
    assert groups[0].id_group == '0'
    assert groups[0].type == ['H', 'C', 'P', 'O']
    assert groups[1].id_group == '1'
    assert groups[1].type == ['Fe2']
    assert groups[2].id_group == '3'
    assert groups[2].type == ['K']
    assert groups[3].id_group == 'test'
    assert groups[3].type == ['P']

    # testing the switching on of the distance boolean of the right groups
    assert groups[0].distance_switch == False
    assert groups[1].distance_switch == True
    assert groups[2].distance_switch == True
    assert groups[3].distance_switch == False

    # testing the reading of RDF parameters
    assert np.array_equal(atoms, [['Fe', 'Fe'], ['H', 'O']])
    assert np.array_equal(Rmax, [7, 6])
    assert np.array_equal(N, [500, 400])


# preamble test
    
def test_preamble(sample_MDstep):
    """
    Test case for the 'preamble' function.
    This test checks if the 'preamble' function correctly extract the values from the first part of the 'pwo' file.
    """

    gr0 = group(['H', 'C'], 0)
    gr1 = group(['Fe'], 1)
    sample_MDstep.groups = [gr0, gr1]

    with open('Test/files/test_preamble.pwo_', 'r') as file:
        preamble(file, sample_MDstep)

    assert sample_MDstep.n_atoms == 6
    assert sample_MDstep.n_type == 3
    assert sample_MDstep.alat_to_angstrom == 0.529177249
    assert np.array_equal(sample_MDstep.ax, [2 * 0.529177249, 1 * 0.529177249, 0])
    assert np.array_equal(sample_MDstep.ay, [2 * 0.529177249, -1 * 0.529177249, 0])
    assert np.array_equal(sample_MDstep.az, [0, 0, 6 * 0.529177249])
    
    assert sample_MDstep.groups[0].atoms[0].name == 'H'
    assert sample_MDstep.groups[0].atoms[1].name == 'C'
    assert sample_MDstep.groups[1].atoms[0].name == 'Fe' 

    assert np.array_equal(sample_MDstep.groups[0].id_tot, [4, 5, 6])
    assert np.array_equal(sample_MDstep.groups[0].atoms[0].id, [5, 6])
    assert np.allclose(sample_MDstep.groups[0].atoms[0].position_past, np.array([[-0.52917725, 0., 0.52917725], [0.52917725, 0., -0.52917725]]).reshape(2, 3), atol=0.01)

    assert np.array_equal(sample_MDstep.groups[0].atoms[1].id, [4])
    assert np.allclose(sample_MDstep.groups[0].atoms[1].position_past, np.array([[0., 0., 0.]]).reshape(1, 3), atol=0.01)

    assert np.array_equal(sample_MDstep.groups[1].id_tot, [1, 2, 3])
    assert np.array_equal(sample_MDstep.groups[1].atoms[0].id, [1, 2, 3])
    assert np.allclose(sample_MDstep.groups[1].atoms[0].position_past, np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]).reshape(3, 3), atol=0.01)

    expected_matrix = np.array([[2 * 0.529177249, 2 * 0.529177249, 0], [1 * 0.529177249, -1 * 0.529177249, 0], [0, 0, 6 * 0.529177249]])
    assert np.array_equal(sample_MDstep.matrix, expected_matrix)
    
# extract_forces test
    
def test_extract_forces(sample_MDstep):
    """
    Test case for the 'extract_force' function.
    This test checks if the 'extract_force' function correctly extract the force values from the the 'pwo' file.
    """
    sample_MDstep.n_atoms = 5

    with open('Test/files/test_extract_forces.pwo_', 'r') as fin:
        extract_forces(fin, sample_MDstep)
        next_line = fin.readline()
        assert 'stop' in next_line
        for line in fin:
            assert not 'force =' in line
    

# estract_positions test
    
def test_extract_positions(sample_MDstep):
    """
    Test case for the 'extract_positions' function.
    This test checks if the 'extract_positions' function correctly extract the positions from the the 'pwo' file.
    """
    sample_MDstep.n_atoms = 5
    graphs = graph('name', [10], [['a', 'b']], [10], 'out')

    with open('Test/files/test_extract_positions.pwo_', 'r') as fin:
        for line in fin:
            if 'ATOMIC_POSITIONS' in line:
                extract_positions(fin, sample_MDstep, graphs, True)
                assert 'stop' in fin.readline()


# body test
    
def test_body(sample_MDstep, monkeypatch):
    """
    Test case for the 'body' function.
    This test checks if the 'body' function correctly extract the values and write on the 'pwo' file.
    """
    sample_MDstep.n_atoms = 5
    group1 = group('Fe', '0')
    group1.distance_switch = True
    group1.DOF = 1
    group2 = group('H', '1')
    group2.distance_switch = True
    group2.DOF = 1
    sample_MDstep.groups = [group1, group2]
    sample_MDstep.dist 

    graphs = graph('filename', [1], [['A', 'B']], [5], 'outdir')

    with open('Test/files/test_body.pwo_', 'r') as fin:
        with open('Test/output/test_output.xyz', 'w') as fout:
            body(fout, fin, sample_MDstep, graphs)
    
    with open('test_output', 'r') as fout:
        line = fout.readline()
        assert line == 'single frame\n'

    assert np.isclose(sample_MDstep.U_pot, -13.605703976, atol=1e-5)
    assert sample_MDstep.dt == 4.8378e-5
    assert sample_MDstep.N_iteration == 2
    assert graphs.T == 300
    