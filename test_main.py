import pytest
import numpy as np
from Codes.class_MDstep import MDstep
from Codes.class_group import group
from Codes.class_graph import graph
from pwo_into_xyz import configuration, preamble, extract_forces, extract_positions, body, xyz_gen


@pytest.fixture
def sample_MDstep():
    """
    Fixture for providing a sample 'MDstep' object for testing purposes.

    Returns
    -------
    MDstep
        An empty instance of the 'MDstep' class, useful for testing functions or methods that rely on 'MDstep' objects.
    """
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

    Raises
    ------
    AssertionError
        If the prompt for the input file name is not recognized.

    Notes
    -----
    This fixture is particularly useful for testing functions or methods that rely on user input.
    It mocks the behavior of the built-in input() function by providing a mock response specifically
    tailored for reading the input file for the setup.
    """
    # Define logic for simulating user input
    def mock_input(prompt):
        if "Write the input file name:\n" in prompt:
            return "Test/files/Setup.ini"
        else:
            assert 1 == 0  # Raise an AssertionError if the prompt is not recognized
    monkeypatch.setattr('builtins.input', mock_input)


# test config -----------------------------------------------------------------------------------------------------
    
def test_configuration(mocked_input):
    """
    Test case for the 'setup' function.

    This test checks if the 'setup' function correctly updates values from the setup file.

    Parameters
    ----------
    mocked_input : function
        A function that mocks user input for testing purposes. This fixture provides mock responses
        to simulate user input for reading the setup file.

    Raises
    ------
    AssertionError
        If the read values do not match the expected values.

    Notes
    -----
    This test case verifies the behavior of the 'setup' function by testing its ability to correctly
    read values from the setup file. It uses the 'mocked_input' fixture to simulate user input
    for reading the setup file. The test checks various aspects including filename, output directory,
    file paths, atom groups, distance switches, and parameters for radial distribution functions (RDF).
    """

    # Calling the 'configuration' function to read setup values
    filename, outdir, Rmax, atoms, N, groups, filepath = configuration()

    # Asserting filename, output directory, and file paths
    assert filename == 'test_file'
    assert outdir == 'Test/output'
    assert filepath == 'Test/file/test_file'

    # Testing the groups
    assert groups[0].id_group == '0'
    assert groups[0].type == ['H', 'C', 'P', 'O']
    assert groups[1].id_group == '1'
    assert groups[1].type == ['Fe2']
    assert groups[2].id_group == '3'
    assert groups[2].type == ['K']
    assert groups[3].id_group == 'test'
    assert groups[3].type == ['P']

    # Testing the switching on of the distance boolean of the right groups
    assert groups[0].distance_switch == False
    assert groups[1].distance_switch == True
    assert groups[2].distance_switch == True
    assert groups[3].distance_switch == False

    # Testing the reading of RDF parameters
    assert np.array_equal(atoms, [['Fe', 'Fe'], ['H', 'O']])
    assert np.array_equal(Rmax, [7, 6])
    assert np.array_equal(N, [500, 400])


# preamble test ---------------------------------------------------------------------------------------------------
    
def test_preamble(sample_MDstep):
    """
    Test case for the 'preamble' function.
    This test checks if the 'preamble' function correctly updates the values from the first part of the 'pwo' file.

    Raises
    ------
    AssertionError
        If any of the extracted values does not match the expected values.

    Notes
    -----
    This test ensures that the 'preamble' function correctly updates the values of the 'MDstep' instance 
    based on the information extracted from the preamble part of the 'pwo' file. It does so by providing 
    a mock 'MDstep' instance with predefined attributes and then calling the 'preamble' function with 
    a mock file object containing the preamble information. Finally, it verifies that the attributes 
    of the 'MDstep' instance have been correctly updated.
    """

    # Create mock 'group' instances
    gr0 = group(['H', 'C'], 0)
    gr1 = group(['Fe'], 1)
    sample_MDstep.groups = [gr0, gr1]

    # Open mock file object
    with open('Test/files/test_preamble.pwo_', 'r') as file:
        # Update attributes using the 'preamble' function
        preamble(file, sample_MDstep)

    # Verify updated attributes
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

    This test verifies whether the 'extract_force' function correctly updates the force values from the 'pwo' file.

    Parameters
    ----------
    sample_MDstep : MDstep
        An instance of the MDstep class representing the molecular dynamics step. 

    Raises
    ------
    AssertionError
        If the function does not correctly stop updating force values from the 'pwo' file or if any line after 
        the force updation contains the term 'force ='.

    Notes
    -----
    This test checks the behavior of the 'extract_force' function by examining its ability to update force values 
    from a 'pwo' file. It uses a sample instance of the MDstep class to hold the extracted data. The test ensures 
    that the function correctly stops updating force values from the file and that no lines after the force 
    update contain the term 'force ='.
    """
    # Set the number of atoms for the sample MDstep
    sample_MDstep.n_atoms = 5

    # Open the test 'pwo' file for force extraction
    with open('Test/files/test_extract_forces.pwo_', 'r') as fin:
        # Call the extract_forces function to extract forces into the sample_MDstep
        extract_forces(fin, sample_MDstep)

        # Check if the next line contains 'stop' indicating the end of force extraction
        next_line = fin.readline()
        assert 'stop' in next_line

        # Check if any line after force extraction contains 'force =', indicating an error in extraction
        for line in fin:
            assert not 'force =' in line


# estract_positions test
    
def test_extract_positions(sample_MDstep):
    """
    Test case for the 'extract_positions' method of the 'MDstep' class.

    This test verifies whether the 'extract_positions' method correctly updates the positions from the 'pwo' file.

    Parameters
    ----------
    sample_MDstep : MDstep
        An instance of the MDstep class representing the molecular dynamics step.

    Raises
    ------
    AssertionError
        If the method does not correctly stop updating position values in the 'pwo' file after reaching the 'stop' line.

    Notes
    -----
    This test checks the behavior of the 'extract_positions' method by examining its ability to update position values 
    from a 'pwo' file. It uses a sample instance of the MDstep class to hold the extracted data. The test ensures that 
    the method correctly stops updating position values in the file after reaching the 'stop' line.
    """
    # Set the number of atoms for the sample MDstep
    sample_MDstep.n_atoms = 5
    graphs = graph('name', [10], [['a', 'b']], [10], 'out')

    # Open the test 'pwo' file for position extraction
    with open('Test/files/test_extract_positions.pwo_', 'r') as fin:
        # Loop through each line in the file
        for line in fin:
            # Check if 'ATOMIC_POSITIONS' is in the line
            if 'ATOMIC_POSITIONS' in line:
                extract_positions(fin, sample_MDstep, graphs, True)
                assert 'stop' in fin.readline()


# body test
    
def test_body(sample_MDstep):
    """
    Test case for the 'body' method of the 'MDstep' class.

    This test verifies whether the 'body' method correctly extracts values and writes to the 'pwo' file.

    Parameters
    ----------
    sample_MDstep : MDstep
        An instance of the MDstep class representing the molecular dynamics step.

    Raises
    ------
    AssertionError
        If the extracted values or the written output do not match the expected values.

    Notes
    -----
    This test checks the behavior of the 'body' method by examining its ability to extract values and write 
    to the 'pwo' file. It uses a sample instance of the MDstep class to hold the extracted data. The test 
    ensures that the method correctly writes output to the specified file and updates attributes in the 
    sample_MDstep instance based on the extracted values.

    Specifically, the test performs the following steps:
    1. Sets the number of atoms for the sample MDstep.
    2. Defines groups and their properties, including atomic species and distance switches.
    3. Opens the test 'pwo' file for extraction and creates a new output file.
    4. Calls the 'body' method to extract values and write to the output file.
    5. Checks the updated attributes in the sample_MDstep instance, including potential energy, time step,
       iteration count, and temperature from the graph instance.

    This comprehensive test ensures that the 'body' method functions correctly, facilitating accurate 
    extraction and writing of data during the molecular dynamics simulation process.
    """
    # Set the number of atoms for the sample MDstep
    sample_MDstep.n_atoms = 5

    # Define groups and their properties
    group1 = group('Fe', '0')
    group1.distance_switch = True
    group1.DOF = 1
    group2 = group('H', '1')
    group2.distance_switch = True
    group2.DOF = 1
    sample_MDstep.groups = [group1, group2]

    # Define a sample graph
    graphs = graph('filename', [1], [['A', 'B']], [5], 'outdir')

    # Open the test 'pwo' file for extraction and create a new output file
    with open('Test/files/test_body.pwo_', 'r') as fin:
        with open('Test/output/test_output.xyz', 'w') as fout:
            body(fout, fin, sample_MDstep, graphs)

    assert np.isclose(sample_MDstep.U_pot, -13.605703976, atol=1e-5)
    assert sample_MDstep.dt == 4.8378e-5
    assert sample_MDstep.N_iteration == 2
    assert graphs.T == 300
