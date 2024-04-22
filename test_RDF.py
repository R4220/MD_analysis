import pytest
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from Codes.class_RDF import RDF
from Codes.class_MDstep import MDstep

@pytest.fixture
def sample_RDF_equal():
    """
    Fixture for generating a sample RDF (Radial Distribution Function) object with identical atom types.

    Returns
    -------
    RDF
        An instance of the 'RDF' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an RDF (Radial Distribution Function) object with the specified parameters:
    - Rmax: Maximum distance considered for RDF calculation (Angstrom).
    - atoms: List of atom types for RDF calculation.
    - N_bin: Number of bins for RDF calculation.
    - filename: Name of the RDF output file.
    - outdir: Directory path for storing the RDF output file.

    This fixture is useful for testing RDF calculations with identical atom types.
    """

    return RDF(Rmax=10.0, atoms=['H', 'H'], N_bin=10, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_different():
    """
    Fixture for generating a sample RDF (Radial Distribution Function) object with different atom types.

    Returns
    -------
    RDF
        An instance of the 'RDF' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an RDF (Radial Distribution Function) object with the specified parameters:
    - Rmax: Maximum distance considered for RDF calculation (Angstrom).
    - atoms: List of atom types for RDF calculation.
    - N_bin: Number of bins for RDF calculation.
    - filename: Name of the RDF output file.
    - outdir: Directory path for storing the RDF output file.

    This fixture is useful for testing RDF calculations with different atom types.
    """

    return RDF(Rmax=10.0, atoms=['H', 'O'], N_bin=10, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_equal_b():
    """
    Fixture for generating a sample RDF (Radial Distribution Function) object with equal atom types and a different 
    number of bins.

    Returns
    -------
    RDF
        An instance of the 'RDF' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an RDF (Radial Distribution Function) object with the specified parameters:
    - Rmax: Maximum distance considered for RDF calculation (Angstrom).
    - atoms: List of atom types for RDF calculation (identical).
    - N_bin: Number of bins for RDF calculation (different from the default value).
    - filename: Name of the RDF output file.
    - outdir: Directory path for storing the RDF output file.

    This fixture is useful for testing RDF calculations with identical atom types but varying parameters.
    """

    return RDF(Rmax=10.0, atoms=['H', 'H'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_different_b():
    """
    Fixture for generating a sample RDF (Radial Distribution Function) object with different atom types and a 
    different number of bins.

    Returns
    -------
    RDF
        An instance of the 'RDF' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an RDF (Radial Distribution Function) object with the specified parameters:
    - Rmax: Maximum distance considered for RDF calculation (Angstrom).
    - atoms: List of atom types for RDF calculation (different).
    - N_bin: Number of bins for RDF calculation (different from the default value).
    - filename: Name of the RDF output file.
    - outdir: Directory path for storing the RDF output file.

    This fixture is useful for testing RDF calculations with different atom types and varying parameters.
    """

    return RDF(Rmax=10.0, atoms=['H', 'O'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_equal_lowRmax():
    """
    Fixture for generating a sample RDF (Radial Distribution Function) object with equal atom types and a low Rmax 
    value.

    Returns
    -------
    RDF
        An instance of the 'RDF' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an RDF (Radial Distribution Function) object with the specified parameters:
    - Rmax: Maximum distance considered for RDF calculation (low value, typically less than the box size).
    - atoms: List of atom types for RDF calculation (equal).
    - N_bin: Number of bins for RDF calculation.
    - filename: Name of the RDF output file.
    - outdir: Directory path for storing the RDF output file.

    This fixture is useful for testing RDF calculations with equal atom types and a low Rmax value, which can be 
    helpful for scenarios where the RDF needs to be calculated within a confined region.
    """

    return RDF(Rmax=2.0, atoms=['H', 'H'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_different_lowRmax():
    """
    Fixture for generating a sample RDF (Radial Distribution Function) object with different atom types and a low 
    Rmax value.

    Returns
    -------
    RDF
        An instance of the 'RDF' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an RDF (Radial Distribution Function) object with the specified parameters:
    - Rmax: Maximum distance considered for RDF calculation (low value, typically less than the box size).
    - atoms: List of atom types for RDF calculation (different).
    - N_bin: Number of bins for RDF calculation.
    - filename: Name of the RDF output file.
    - outdir: Directory path for storing the RDF output file.

    This fixture is useful for testing RDF calculations with different atom types and a low Rmax value, which can 
    be helpful for scenarios where the RDF needs to be calculated within a confined region.
    """

    return RDF(Rmax=2.0, atoms=['H', 'O'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_MDstep():
    """
    Fixture for generating a sample MDstep object for testing.

    Returns
    -------
    MDstep
        An instance of the 'MDstep' class with predefined attributes for testing.

    Notes
    -----
    This fixture initializes an MDstep object with the following predefined attributes:
    - groups: Empty list representing the groups in the MDstep object.
    - N_iteration: Number of iterations.
    - matrix: Transformation matrix for coordinate conversion.
    
    This fixture is useful for testing MDstep-related functionalities and methods.
    """

    step = MDstep(groups=[])
    step.N_iteration = 1
    step.matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    return step


# initialization test ---------------------------------------------------------------------------------------------

def test_initialization_equal(sample_RDF_equal):
    """
    Test case for the initialization of the 'RDF' class.
    
    This test checks if the 'RDF' class is initialized correctly with the provided attributes, considering equal 
    atomic types.
    
    Parameters
    ----------
    sample_RDF_equal : RDF
        An instance of the RDF class with equal atoms type.

    Raises
    ------
    AssertionError
        If any of the attributes of the 'RDF' instance do not match the expected values.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_equal` is correctly initialized with the specified 
    attributes.
    The test verifies the initialization of various attributes of the 'RDF' class instance, including filename, 
    outdir, Rmax, type, N_bin, count, R, dR, norm, equal, at1, N1, at2, N2, and RDF_color. 
    It compares these attributes against expected values to ensure proper initialization.
    """

    # Check if the attributes are initialized correctly
    assert sample_RDF_equal.filename == 'test'
    assert sample_RDF_equal.outdir == 'output'
    assert sample_RDF_equal.Rmax == 10.0
    assert sample_RDF_equal.type == ['H', 'H']
    assert sample_RDF_equal.N_bin == 10
    assert np.array_equal(sample_RDF_equal.count, np.zeros(10))
    assert np.array_equal(sample_RDF_equal.R, np.linspace(0, 10.0, 10))
    assert np.isclose(sample_RDF_equal.dR, 1.1111, atol=1e-4)
    expected_norm = np.multiply([((i + 1.11111111)**3 - i**3) for i in np.linspace(0, 10.0, 10)], np.pi * 4/3)
    assert np.allclose(sample_RDF_equal.norm, expected_norm, atol=1e-4)
    assert sample_RDF_equal.equal == True
    assert np.array_equal(sample_RDF_equal.at1, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_equal.N1 == 0
    assert np.array_equal(sample_RDF_equal.at2, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_equal.N2 == 0
    assert sample_RDF_equal.RDF_color == 'black'

def test_initialization_different(sample_RDF_different):
    """
    Test case for the initialization of the 'RDF' class.
    
    Test the initialization of the RDF class with different atoms type.
    This test checks if the 'RDF' class is initialized correctly with the provided attributes.
    
    Parameters
    ----------
    sample_RDF_different : RDF
        An instance of the RDF class with different atoms type.

    Raises
    ------
    AssertionError
        If any of the attributes of the 'RDF' instance do not match the expected values.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_different` is correctly initialized with the specified 
    attributes.
    The test verifies the initialization of various attributes of the 'RDF' class instance, including filename, 
    outdir, Rmax, type, N_bin, count, R, dR, norm, equal, at1, N1, at2, N2, and RDF_color. 
    It compares these attributes against expected values to ensure proper initialization.
    """

    # Check if the attributes are initialized correctly
    assert sample_RDF_different.filename == 'test'
    assert sample_RDF_different.outdir == 'output'
    assert sample_RDF_different.Rmax == 10.0
    assert sample_RDF_different.type == ['H', 'O']
    assert sample_RDF_different.N_bin == 10
    assert np.array_equal(sample_RDF_different.count, np.zeros(10))
    assert np.array_equal(sample_RDF_different.R, np.linspace(0, 10.0, 10))
    assert np.isclose(sample_RDF_different.dR, 1.1111, atol=1e-4)
    expected_norm = np.multiply([((i + 1.11111111)**3 - i**3) for i in np.linspace(0, 10.0, 10)], np.pi * 4/3)
    assert np.allclose(sample_RDF_different.norm, expected_norm, atol=1e-4)
    assert sample_RDF_different.equal == False
    assert np.array_equal(sample_RDF_different.at1, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_different.N1 == 0
    assert np.array_equal(sample_RDF_different.at2, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_different.N2 == 0
    assert sample_RDF_different.RDF_color == 'black'


# RDF test---------------------------------------------------------------------------------------------------------
    
def test_RDF_equal1(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    Test the calculation of RDF for the same atom, just considering difference along the x-coordinates.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    
    Parameters
    ----------
    sample_RDF_equal_b : RDF
        An instance of the RDF class with equal atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.

    AssertionError
        If the number of atoms in `sample_RDF_equal_b` is not updated correctly.

    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_equal_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The test sets the initial positions of atoms in `sample_RDF_equal_b` and calculates the RDF using the provided 
    MD step object `sample_MDstep`. It then verifies the calculated RDF values against the expected count, checks 
    if the number of atoms is updated correctly, and ensures that the arrays `at1` and `at2` are reset to empty 
    arrays after the RDF calculation.
    """

    # Set initial positions of atoms
    sample_RDF_equal_b.at1 = np.array([[0, 0, 0], [2, 0, 0]])
    
    # Calculate RDF
    sample_RDF_equal_b.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([0], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_equal_b.count, expected_count)
    assert sample_RDF_equal_b.N1 == 2
    assert np.array_equal(sample_RDF_equal_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_equal2(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object. 
    In this case, we consider positions with different coordinates along all three axes.
    
    Parameters
    ----------
    sample_RDF_equal_b : RDF
        An instance of the RDF class with equal atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_equal_b` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_equal_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The test first sets the initial positions of atoms with different coordinates along all three axes in 
    `sample_RDF_equal_b`. Then, it calculates the RDF using the provided MD step object `sample_MDstep`. 
    After the calculation, it verifies the calculated RDF values against the expected count and checks if the 
    number of atoms is updated correctly. Finally, it ensures that the arrays `at1` and `at2` are reset to empty 
    arrays after the RDF calculation.
    """

    # Set initial positions of atoms
    sample_RDF_equal_b.at1 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    
    # Calculate RDF
    sample_RDF_equal_b.RDF(sample_MDstep)

     # Verify the calculated RDF values
    expected_count = np.histogram([1.187434208703792], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_equal_b.count, expected_count)
    assert sample_RDF_equal_b.N1 == 2
    assert np.array_equal(sample_RDF_equal_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_equal3(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object, 
    considering multiple atoms with different positions.
    
    Parameters
    ----------
    sample_RDF_equal_b : RDF
        An instance of the RDF class with equal atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_equal_b` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_equal_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The test sets the initial positions of multiple atoms with different coordinates in `sample_RDF_equal_b` and 
    calculates the RDF using the provided MD step object `sample_MDstep`. After the calculation, it verifies the 
    calculated RDF values against the expected count, checks if the number of atoms is updated correctly, and 
    ensures that the arrays `at1` and `at2` are reset to empty arrays after the RDF calculation.
    """

    # Set initial positions of atoms
    sample_RDF_equal_b.at1 = np.array([[0, 0, 0],[0.5, -2, 4], [1, 3, -0.4]])
    
    # Calculate RDF
    sample_RDF_equal_b.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([1.187434208703792, 2.0615528128088303, 1.469693845669907], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_equal_b.count, expected_count)
    assert sample_RDF_equal_b.N1 == 3
    assert np.array_equal(sample_RDF_equal_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different1(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object, 
    considering different atoms with different coordinates along the x-axis.
    
    Parameters
    ----------
    sample_RDF_different_b : RDF
        An instance of the RDF class with different atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_different_b` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_different_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The test sets the initial positions of atoms in `sample_RDF_different_b`, calculates the RDF using the provided 
    MD step object `sample_MDstep`, and then verifies the calculated RDF values against the expected count. It also 
    checks if the number of atoms is updated correctly and ensures that the arrays `at1` and `at2` are reset to 
    empty arrays after the RDF calculation.
    """

    # Set initial positions of atoms
    sample_RDF_different_b.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_b.at2 = np.array([[2, 0, 0]])
    
    # Calculate RDF
    sample_RDF_different_b.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([0], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_different_b.count, expected_count)
    assert sample_RDF_different_b.N1 == 1
    assert sample_RDF_different_b.N2 == 1
    assert np.array_equal(sample_RDF_different_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different2(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object, 
    considering different atoms with specific coordinates.
    
    Parameters
    ----------
    sample_RDF_different_b : RDF
        An instance of the RDF class with different atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_different_b` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_different_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The test sets the initial positions of atoms in `sample_RDF_different_b`, calculates the RDF using the provided 
    MD step object `sample_MDstep`, and then verifies the calculated RDF values against the expected count. It also 
    checks if the number of atoms is updated correctly and ensures that the arrays `at1` and `at2` are reset to 
    empty arrays after the RDF calculation.
    """

    # Set initial positions of atoms
    sample_RDF_different_b.at1 = np.array([[0.5, -2, 4]])
    sample_RDF_different_b.at2 = np.array([[1, 3, -0.4]])
    
    # Calculate RDF
    sample_RDF_different_b.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([1.187434208703792], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_different_b.count, expected_count)
    assert sample_RDF_different_b.N1 == 1
    assert sample_RDF_different_b.N2 == 1
    assert np.array_equal(sample_RDF_different_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different3(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object, 
    considering different atoms with specific coordinates.
    
    Parameters
    ----------
    sample_RDF_different_b : RDF
        An instance of the RDF class with different atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_different_b` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_different_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The test sets the initial positions of atoms in `sample_RDF_different_b`, calculates the RDF using the provided 
    MD step object `sample_MDstep`, and then verifies the calculated RDF values against the expected count. It also 
    checks if the number of atoms is updated correctly and ensures that the arrays `at1` and `at2` are reset to 
    empty arrays after the RDF calculation.
    """

    # Set initial positions of atoms
    sample_RDF_different_b.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_b.at2 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    
    # Calculate RDF
    sample_RDF_different_b.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([2.0615528128088303, 1.469693845669907], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_different_b.count, expected_count)
    assert sample_RDF_different_b.N1 == 1
    assert sample_RDF_different_b.N2 == 2
    assert np.array_equal(sample_RDF_different_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_equal_lowRmax(sample_RDF_equal_lowRmax, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object, 
    considering equal atoms with different number of bins and lower Rmax.
    
    Parameters
    ----------
    sample_RDF_equal_lowRmax : RDF
        An instance of the RDF class with equal atoms type and different number of bins and lower Rmax.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_equal_lowRmax` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_equal_lowRmax` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The RDF counts are first calculated for the given system configuration. Then, the RDF counts are verified 
    against the expected counts.
    The RDF calculation is performed for a system with lower Rmax, and the RDF counts are validated accordingly.
    """

    # Set initial positions of atoms
    sample_RDF_equal_lowRmax.at1 = np.array([[0, 0, 0],[0.5, -2, 4], [1, 3, -0.4]])
    
    # Calculate RDF
    sample_RDF_equal_lowRmax.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([1.187434208703792, 1.469693845669907], bins=100, range=(0, 2.0))[0]
    assert np.array_equal(sample_RDF_equal_lowRmax.count, expected_count)
    assert sample_RDF_equal_lowRmax.N1 == 3
    assert np.array_equal(sample_RDF_equal_lowRmax.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_lowRmax.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different_lowRmax(sample_RDF_different_lowRmax, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object, 
    considering different atoms with lower Rmax.
    
    Parameters
    ----------
    sample_RDF_different_lowRmax : RDF
        An instance of the RDF class with different atoms type and lower Rmax.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the calculated RDF count does not match the expected count.
    AssertionError
        If the number of atoms in `sample_RDF_different_lowRmax` is not updated correctly.
    AssertionError
        If the arrays `at1` and `at2` are not reset to empty arrays after RDF calculation.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_different_lowRmax` and the MDstep instance `sample_MDstep` 
    are correctly initialized.
    The RDF counts are first calculated for the given system configuration. Then, the RDF counts are verified 
    against the expected counts.
    The RDF calculation is performed for a system with lower Rmax, and the RDF counts are validated accordingly.
    """

    # Set initial positions of atoms
    sample_RDF_different_lowRmax.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_lowRmax.at2 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    
    # Calculate RDF
    sample_RDF_different_lowRmax.RDF(sample_MDstep)

    # Verify the calculated RDF values
    expected_count = np.histogram([ 1.469693845669907], bins=100, range=(0, 2.0))[0]
    assert np.array_equal(sample_RDF_different_lowRmax.count, expected_count)
    assert sample_RDF_different_lowRmax.N1 == 1
    assert sample_RDF_different_lowRmax.N2 == 2
    assert np.array_equal(sample_RDF_different_lowRmax.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_lowRmax.at2, np.array([], dtype=float).reshape(0, 3))


# normalization test ----------------------------------------------------------------------------------------------

def test_normalization_equal(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'normalization' method of the 'RDF' class.
    
    This test checks if the 'normalization' method correctly normalizes the RDF counts based on the system volume 
    and atom counts, considering atoms of equal type.
    
    Parameters
    ----------
    sample_RDF_equal_b : RDF
        An instance of the RDF class with equal atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the normalized RDF counts do not match the expected normalized counts.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_equal_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The RDF counts are first calculated for the given system configuration. Then, the RDF counts are normalized 
    using the `normalization` method of the RDF class. The normalization is done based on the system volume and 
    the number of atoms of type 1. The normalized count is obtained by dividing the RDF counts by the product of 
    the RDF norm and the atomic density of type 1 atoms in the system.
    """

    # Set initial positions of atoms and calculate RDF
    sample_RDF_equal_b.at1 = np.array([[0, 0, 0],[0.5, -2, 4], [1, 3, -0.4]])
    sample_RDF_equal_b.RDF(sample_MDstep)
    expected_count = sample_RDF_equal_b.count
    
    # Normalize RDF counts
    sample_RDF_equal_b.normalization(sample_MDstep)

    # Calculate expected normalized counts
    V = 8 * 10.0 ** 3  # Volume of the system (cubic angstroms)
    rho_1 = V / (3 * 2 * 1)  # Atomic density of type 1 atoms (atoms per cubic angstrom)
    expected_normalized_count = np.divide(expected_count, sample_RDF_equal_b.norm) * rho_1
    
    # Verify the normalized RDF counts
    assert np.array_equal(sample_RDF_equal_b.count, expected_normalized_count)

def test_normalization_different(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'normalization' method of the 'RDF' class.
    
    This test checks if the 'normalization' method correctly normalizes the RDF counts based on the system volume 
    and atom counts.
    
    Parameters
    ----------
    sample_RDF_different_b : RDF
        An instance of the RDF class with different atoms type and different number of bins.
    sample_MDstep : MDstep
        An instance of the MDstep class representing the MD step object.

    Raises
    ------
    AssertionError
        If the normalized RDF counts do not match the expected normalized counts.

    Notes
    -----
    This test assumes that the RDF instance `sample_RDF_different_b` and the MDstep instance `sample_MDstep` are 
    correctly initialized.
    The RDF counts are first calculated for the given system configuration. Then, the RDF counts are normalized 
    using the `normalization` method of the RDF class. The normalization is done based on the system volume and 
    the number of atoms of type 1. The normalized count is obtained by dividing the RDF counts by the product of 
    the RDF norm and the atomic density of type 1 atoms in the system.
    """

    # Set initial positions of atoms and calculate RDF
    sample_RDF_different_b.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_b.at2 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    sample_RDF_different_b.RDF(sample_MDstep)
    expected_count = sample_RDF_different_b.count
    
    # Normalize RDF counts
    sample_RDF_different_b.normalization(sample_MDstep)

    # Calculate expected normalized counts
    V = 8 * 10.0 ** 3  # Volume of the system (cubic angstroms)
    rho_1 = V / (1 * 2 * 1)  # Atomic density of type 1 atoms (atoms per cubic angstrom)
    expected_normalized_count = np.divide(expected_count, sample_RDF_different_b.norm) * rho_1
    
    # Verify the normalized RDF counts
    assert np.array_equal(sample_RDF_different_b.count, expected_normalized_count)
