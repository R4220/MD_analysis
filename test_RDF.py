import pytest
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from Codes.class_RDF import RDF
from Codes.class_MDstep import MDstep

@pytest.fixture
def sample_RDF_equal():
    return RDF(Rmax=10.0, atoms=['H', 'H'], N_bin=10, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_different():
    return RDF(Rmax=10.0, atoms=['H', 'O'], N_bin=10, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_equal_b():
    return RDF(Rmax=10.0, atoms=['H', 'H'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_different_b():
    return RDF(Rmax=10.0, atoms=['H', 'O'], N_bin=100, filename='test', outdir='output')
# -----------------------------------------------------------------------------------------------------------------
@pytest.fixture
def sample_RDF_equal_lowRmax():
    return RDF(Rmax=2.0, atoms=['H', 'H'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_RDF_different_lowRmax():
    return RDF(Rmax=2.0, atoms=['H', 'O'], N_bin=100, filename='test', outdir='output')

@pytest.fixture
def sample_MDstep():
    step = MDstep(groups=[])
    step.N_iteration = 1
    step.matrix = [[2, 0, 0], [0, 4, 0], [0, 0, 1]]
    return step

# graph_esthetic test

def test_graph_aesthetic(sample_RDF_equal, monkeypatch):
    """
    Test case for the 'graph_aesthetic' method of the 'RDF' class.
    This test checks if the 'graph_aesthetic' method correctly reads and updates the default parameters based on the configuration file.
    """
    # Mocking the open function to simulate the 'Setup_graph.txt' content
    fake_config_content = '\n'.join([' # GRAPH VALUES', 'axes.labelsize = 18', 'xtick.labelsize = 16', 'ytick.labelsize = 16', 'legend.fontsize = 16', '# COLORS', 'RDF_color = #425840', 'Energy_sum = True', 'K_energy_color = #b32323', 'U_energy_color = #191971', 'Tot_energy_color = #006400', 'Group_color = #9ACD32 #b32323 #191971 #006400 #b0edef #470303', 'Force_color = #000000', 'Temperature_color = blue'])
    
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: StringIO(fake_config_content))

    sample_RDF_equal.graph_aesthetic()

    assert plt.rcParams['axes.labelsize'] == 18.0
    assert plt.rcParams['xtick.labelsize'] == 16.0
    assert plt.rcParams['ytick.labelsize'] == 16.0
    assert plt.rcParams['legend.fontsize'] == 16.0
    assert sample_RDF_equal.RDF_color == '#425840' 

def test_graph_aesthetic_default(monkeypatch):
    """
    Test case for the 'graph_aesthetic' method of the 'RDF' class.
    This test checks if the 'graph_aesthetic' method correctly reads and use the default parameters since in the configuration file there isn't the informations .
    """
    # Mocking the open function to simulate the 'Setup_graph.txt' content
    fake_config_content = '\n'.join([' # GRAPH VALUES', '# COLORS'])
    
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: StringIO(fake_config_content))

    sample_RDF_default = RDF(Rmax=2.0, atoms=['H', 'O'], N_bin=100, filename='test', outdir='output')

    assert plt.rcParams['axes.labelsize'] == 16.0
    assert plt.rcParams['xtick.labelsize'] == 14.0
    assert plt.rcParams['ytick.labelsize'] == 14.0
    assert plt.rcParams['legend.fontsize'] == 14.0
    assert sample_RDF_default.RDF_color == 'black' 


# initialization test

def test_initialization_equal(sample_RDF_equal):
    """
    Test case for the initialization of the 'RDF' class.
    This test checks if the 'RDF' class is initialized correctly with the provided attributes.
    """
    assert sample_RDF_equal.filename == 'test'
    assert sample_RDF_equal.outdir == 'output'
    assert sample_RDF_equal.Rmax == 10.0
    assert sample_RDF_equal.type == ['H', 'H']
    assert sample_RDF_equal.N_bin == 10
    assert np.array_equal(sample_RDF_equal.count, np.zeros(10))
    assert np.array_equal(sample_RDF_equal.R, np.linspace(0, 10.0, 10))
    assert np.isclose(sample_RDF_equal.dR, 1.1111, atol = 1e-4 )
    expected_norm = np.multiply([((i + 1.11111111)**3 - i**3) for i in np.linspace(0, 10.0, 10)], np.pi * 4/3)
    assert np.allclose(sample_RDF_equal.norm, expected_norm, atol=1e-4)
    assert sample_RDF_equal.condition == True
    assert np.array_equal(sample_RDF_equal.at1, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_equal.N1 == 0
    assert np.array_equal(sample_RDF_equal.at2, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_equal.N2 == 0
    assert sample_RDF_equal.RDF_color == '#425840'

def test_initialization_different(sample_RDF_different):
    """
    Test case for the initialization of the 'RDF' class.
    This test checks if the 'RDF' class is initialized correctly with the provided attributes.
    """
    assert sample_RDF_different.filename == 'test'
    assert sample_RDF_different.outdir == 'output'
    assert sample_RDF_different.Rmax == 10.0
    assert sample_RDF_different.type == ['H', 'O']
    assert sample_RDF_different.N_bin == 10
    assert np.array_equal(sample_RDF_different.count, np.zeros(10))
    assert np.array_equal(sample_RDF_different.R, np.linspace(0, 10.0, 10))
    assert np.isclose(sample_RDF_different.dR, 1.1111, atol = 1e-4 )
    expected_norm = np.multiply([((i + 1.11111111)**3 - i**3) for i in np.linspace(0, 10.0, 10)], np.pi * 4/3)
    assert np.allclose(sample_RDF_different.norm, expected_norm, atol=1e-4)
    assert sample_RDF_different.condition == False
    assert np.array_equal(sample_RDF_different.at1, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_different.N1 == 0
    assert np.array_equal(sample_RDF_different.at2, np.array([], dtype=float).reshape(0, 3))
    assert sample_RDF_different.N2 == 0
    assert sample_RDF_different.RDF_color == '#425840'


# RDF test
    
def test_RDF_equal1(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object. This case consider the same atom in two different cell.
    """
    sample_RDF_equal_b.at1 = np.array([[0, 0, 0], [2, 0, 0]])
    
    sample_RDF_equal_b.RDF(sample_MDstep)

    expected_count = np.histogram([0], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_equal_b.count, expected_count)
    assert sample_RDF_equal_b.N1 == 2
    assert np.array_equal(sample_RDF_equal_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_equal2(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    """
    sample_RDF_equal_b.at1 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    
    sample_RDF_equal_b.RDF(sample_MDstep)

    expected_count = np.histogram([1.187434208703792], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_equal_b.count, expected_count)
    assert sample_RDF_equal_b.N1 == 2
    assert np.array_equal(sample_RDF_equal_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_equal3(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    """
    sample_RDF_equal_b.at1 = np.array([[0, 0, 0],[0.5, -2, 4], [1, 3, -0.4]])
    
    sample_RDF_equal_b.RDF(sample_MDstep)

    expected_count = np.histogram([1.187434208703792, 2.0615528128088303, 1.469693845669907], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_equal_b.count, expected_count)
    assert sample_RDF_equal_b.N1 == 3
    assert np.array_equal(sample_RDF_equal_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different1(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object. This case consider the same atom in two different cell.
    """
    sample_RDF_different_b.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_b.at2 = np.array([[2, 0, 0]])
    
    sample_RDF_different_b.RDF(sample_MDstep)

    expected_count = np.histogram([0], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_different_b.count, expected_count)
    assert sample_RDF_different_b.N1 == 1
    assert sample_RDF_different_b.N2 == 1
    assert np.array_equal(sample_RDF_different_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different2(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    """
    sample_RDF_different_b.at1 = np.array([[0.5, -2, 4]])
    sample_RDF_different_b.at2 = np.array([[1, 3, -0.4]])
    
    sample_RDF_different_b.RDF(sample_MDstep)

    expected_count = np.histogram([1.187434208703792], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_different_b.count, expected_count)
    assert sample_RDF_different_b.N1 == 1
    assert sample_RDF_different_b.N2 == 1
    assert np.array_equal(sample_RDF_different_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different3(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    """
    sample_RDF_different_b.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_b.at2 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    
    sample_RDF_different_b.RDF(sample_MDstep)

    expected_count = np.histogram([2.0615528128088303, 1.469693845669907], bins=100, range=(0, 10.0))[0]
    assert np.array_equal(sample_RDF_different_b.count, expected_count)
    assert sample_RDF_different_b.N1 == 1
    assert sample_RDF_different_b.N2 == 2
    assert np.array_equal(sample_RDF_different_b.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_b.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_equal_lowRmax(sample_RDF_equal_lowRmax, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    """
    sample_RDF_equal_lowRmax.at1 = np.array([[0, 0, 0],[0.5, -2, 4], [1, 3, -0.4]])
    
    sample_RDF_equal_lowRmax.RDF(sample_MDstep)

    expected_count = np.histogram([1.187434208703792, 1.469693845669907], bins=100, range=(0, 2.0))[0]
    assert np.array_equal(sample_RDF_equal_lowRmax.count, expected_count)
    assert sample_RDF_equal_lowRmax.N1 == 3
    assert np.array_equal(sample_RDF_equal_lowRmax.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_equal_lowRmax.at2, np.array([], dtype=float).reshape(0, 3))

def test_RDF_different_lowRmax(sample_RDF_different_lowRmax, sample_MDstep):
    """
    Test case for the 'RDF' method of the 'RDF' class.
    This test checks if the 'RDF' method correctly calculates the RDF based on the provided iteration object.
    """
    sample_RDF_different_lowRmax.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_lowRmax.at2 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    
    sample_RDF_different_lowRmax.RDF(sample_MDstep)

    expected_count = np.histogram([ 1.469693845669907], bins=100, range=(0, 2.0))[0]
    assert np.array_equal(sample_RDF_different_lowRmax.count, expected_count)
    assert sample_RDF_different_lowRmax.N1 == 1
    assert sample_RDF_different_lowRmax.N2 == 2
    assert np.array_equal(sample_RDF_different_lowRmax.at1, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_RDF_different_lowRmax.at2, np.array([], dtype=float).reshape(0, 3))


# normalization test

def test_normalization_equal(sample_RDF_equal_b, sample_MDstep):
    """
    Test case for the 'normalization' method of the 'RDF' class.
    This test checks if the 'normalization' method correctly normalizes the RDF counts based on the system volume and atom counts.
    """
    sample_RDF_equal_b.at1 = np.array([[0, 0, 0],[0.5, -2, 4], [1, 3, -0.4]])
    sample_RDF_equal_b.RDF(sample_MDstep)
    expected_count = sample_RDF_equal_b.count
    sample_RDF_equal_b.normalization(sample_MDstep)

    V = 8 * 10.0 ** 3
    rho_1 = V / (3 * 2 * 1)

    expected_normalized_count = np.divide(expected_count,sample_RDF_equal_b.norm) * rho_1
    assert np.array_equal(sample_RDF_equal_b.count, expected_normalized_count)

def test_normalization_different(sample_RDF_different_b, sample_MDstep):
    """
    Test case for the 'normalization' method of the 'RDF' class.
    This test checks if the 'normalization' method correctly normalizes the RDF counts based on the system volume and atom counts.
    """
    sample_RDF_different_b.at1 = np.array([[0, 0, 0]])
    sample_RDF_different_b.at2 = np.array([[0.5, -2, 4], [1, 3, -0.4]])
    sample_RDF_different_b.RDF(sample_MDstep)
    expected_count = sample_RDF_different_b.count
    sample_RDF_different_b.normalization(sample_MDstep)

    V = 8 * 10.0 ** 3
    rho_1 = V / (1 * 2 * 1)
    
    expected_normalized_count = np.divide(expected_count,sample_RDF_different_b.norm) * rho_1
    assert np.array_equal(sample_RDF_different_b.count, expected_normalized_count)


# plot_RDF test
def test_plot_RDF_equal(sample_RDF_equal_b, monkeypatch):
    """
    Test case for the 'plot_RDF' method of the 'RDF' class.
    This test checks if the 'plot_RDF' method correctly generates and saves the RDF plot.
    """
    # Mocking the plt.savefig function to check if it's called with the correct arguments
    saved_filepath = []

    def mock_savefig(filepath):
        saved_filepath.append(filepath)

    monkeypatch.setattr(plt, 'savefig', mock_savefig)

    # Set count to non-zero values for the test
    sample_RDF_equal_b.count = np.ones(100)  
    sample_RDF_equal_b.plot_RDF()

    assert len(saved_filepath) == 1
    assert saved_filepath[0] == 'output/HH_test.png'

def test_plot_RDF_different(sample_RDF_different_b, monkeypatch):
    """
    Test case for the 'plot_RDF' method of the 'RDF' class.
    This test checks if the 'plot_RDF' method correctly generates and saves the RDF plot.
    """
    # Mocking the plt.savefig function to check if it's called with the correct arguments
    saved_filepath = []

    def mock_savefig(filepath):
        saved_filepath.append(filepath)

    monkeypatch.setattr(plt, 'savefig', mock_savefig)

    # Set count to non-zero values for the test
    sample_RDF_different_b.count = np.ones(100)  
    sample_RDF_different_b.plot_RDF()

    assert saved_filepath[0] == 'output/HO_test.png'