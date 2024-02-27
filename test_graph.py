import numpy as np
import matplotlib.pyplot as plt
import pytest
import os
from io import StringIO
from Codes.class_MDstep import MDstep
from Codes.class_group import group
from Codes.class_graph import graph
from Codes.class_RDF import RDF 

@pytest.fixture
def sample_graph():
    filename = "test_file"
    Rmax = [5.0, 6.0]
    atoms = [['A', 'B'], ['C', 'D']]
    N_bin = [100, 120]
    outdir = "output"
    return graph(filename, Rmax, atoms, N_bin, outdir)

@pytest.fixture
def sample_MDstep2():
    # Create an instance of MDstep with sample data.
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    return MDstep([group1, group2])

# graph_esthetic test

def test_graph_aesthetic(monkeypatch):
    """
    Test case for the 'graph_aesthetic' method of the 'graph' class.
    This test checks if the 'graph_aesthetic' method correctly reads and updates the default parameters based on the configuration file.
    """
    # Mocking the open function to simulate the 'Setup_graph.txt' content
    fake_config_content = '\n'.join([' # GRAPH VALUES', 'axes.labelsize = 18', 'xtick.labelsize = 16', 'ytick.labelsize = 16', 'legend.fontsize = 16', '# COLORS', 'RDF_color = #425840', 'Energy_sum = True', 'K_energy_color = #b32323', 'U_energy_color = #191971', 'Tot_energy_color = #006400', 'Group_color = #000000 #9ACD32 #b32323 #191971 #006400 #b0edef #470303'])
    
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: StringIO(fake_config_content))

    filename = "test_file"
    Rmax = [5.0, 6.0]
    atoms = [['A', 'B'], ['C', 'D']]
    N_bin = [100, 120]
    outdir = "output"
    sample_RDF_aesthetic = graph(filename, Rmax, atoms, N_bin, outdir)

    assert plt.rcParams['axes.labelsize'] == 18.0
    assert plt.rcParams['xtick.labelsize'] == 16.0
    assert plt.rcParams['ytick.labelsize'] == 16.0
    assert plt.rcParams['legend.fontsize'] == 16.0

    assert sample_RDF_aesthetic.RDF_color == '#425840' 
    assert sample_RDF_aesthetic.all_energy == True
    assert np.array_equal(sample_RDF_aesthetic.energy_color, ['#b32323', '#191971', '#006400'])
    assert np.array_equal(sample_RDF_aesthetic.group_color, ['#000000', '#9ACD32', '#b32323', '#191971', '#006400', '#b0edef', '#470303'])

def test_graph_aesthetic_default(monkeypatch):
    """
    Test case for the 'graph_aesthetic' method of the 'graph' class.
    This test checks if the 'graph_aesthetic' method correctly reads and updates the default parameters based on the configuration file.
    """
    # Mocking the open function to simulate the 'Setup_graph.txt' content
    fake_config_content = '\n'.join([' # GRAPH VALUES', '# COLORS'])
    
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: StringIO(fake_config_content))
    filename = "test_file"
    Rmax = [5.0, 6.0]
    atoms = [['A', 'B'], ['C', 'D']]
    N_bin = [100, 120]
    outdir = "output"
    sample_RDF_default = graph(filename, Rmax, atoms, N_bin, outdir)

    assert plt.rcParams['axes.labelsize'] == 16.0
    assert plt.rcParams['xtick.labelsize'] == 14.0
    assert plt.rcParams['ytick.labelsize'] == 14.0
    assert plt.rcParams['legend.fontsize'] == 14.0

    assert sample_RDF_default.RDF_color == 'black'
    assert sample_RDF_default.all_energy == False
    assert np.array_equal(sample_RDF_default.energy_color, ['red', 'blue', 'black'])
    assert np.array_equal(sample_RDF_default.group_color, ['red', 'blue', 'green', 'yellow', 'black', 'purple'])

def test_graph_initialization(sample_graph):
    """
    Test case for the initialization method of the 'graph' class.
    This test checks if the 'graph' class is initialized correctly with the provided attributes.
    """
    
    assert sample_graph.filename == "test_file"
    assert sample_graph.outdir == "output"
    assert np.array_equal(sample_graph.Ek, [])
    assert np.array_equal(sample_graph.Up, [])
    assert np.array_equal(sample_graph.T, [])
    assert np.array_equal(sample_graph.F, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_graph.time, [])


    # Checking the correct creation of the 'RDF' object
    couple1 = sample_graph.type[0]
    assert type(couple1) == RDF  
    assert couple1.Rmax == 5.0
    assert np.array_equal(couple1.type, ['A', 'B'])
    assert couple1.filename == "test_file"
    assert couple1.outdir == "output"
    assert couple1.N_bin == 100

    couple2 = sample_graph.type[1]
    assert type(couple2) == RDF  
    assert couple2.Rmax == 6.0
    assert np.array_equal(couple2.type, ['C', 'D'])
    assert couple2.filename == "test_file"
    assert couple2.outdir == "output"
    assert couple2.N_bin == 120


# extracting_values test
    
def test_extracting_values_Force1():
    """
    Test case for the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' extract correctly the total force acting on the system.
    """
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])

    iter = MDstep([group1, group2])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the mock instance of MDstep
    graph_instance.extracting_values(iter)

    # Verify that the values have been correctly extracted and stored
    assert np.array_equal(graph_instance.F, np.array([[5.0, 7.0, 9.0]]))

def test_extracting_values_Force2():
    """
    Test case for the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' extract correctly the total force acting on the system.
    """
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, -5.0, 6.0])

    iter = MDstep([group1, group2])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the mock instance of MDstep
    graph_instance.extracting_values(iter)

    # Verify that the values have been correctly extracted and stored
    assert np.array_equal(graph_instance.F, np.array([[5.0, -3.0, 9.0]]))

def test_extracting_values_Ek():
    """
    Test case for the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' extract correctly the total kinetic energy of the system.
    """
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])
    
    # Define values to simulate
    group1.Ek = 10.0
    group2.Ek = 15.0

    iter = MDstep([group1, group2])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the mock instance of MDstep
    graph_instance.extracting_values(iter)

    # Verify that the values have been correctly extracted and stored
    assert graph_instance.Ek[-1] == 25.0

def test_extracting_values_Upot():
    """
    Test case for the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' extract correctly the potential energy of the system.
    """
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])
    
    # Define values to simulate
    iter = MDstep([group1, group2])
    iter.U_pot = 30.0

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the mock instance of MDstep
    graph_instance.extracting_values(iter)

    # Verify that the values have been correctly extracted and stored
    assert graph_instance.Up[-1] == 30.0

def test_extracting_values_time():
    """
    Test case for the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' extract correctly the time corresponding to the current time step.
    """
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])
    
    # Define values to simulate
    iter = MDstep([group1, group2])
    iter.dt = 10.0
    iter.N_iteration = 50

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the mock instance of iteration
    graph_instance.extracting_values(iter)

    # Verify that the values have been correctly extracted and stored
    assert graph_instance.time[-1] == 500


# plot_energy test
def test_plot_energy(sample_graph, monkeypatch):
    """
    Test case for the 'plot_energy' method of the 'graph' class.
    This test checks if the 'plot_energy' method correctly generates and saves the energy plot.
    """
    # Mocking the plt.savefig function to check if it's called with the correct arguments
    saved_filepath = []

    def mock_savefig(filepath):
        saved_filepath.append(filepath)

    monkeypatch.setattr(plt, 'savefig', mock_savefig)

    # Set count to non-zero values for the test
    sample_graph.time = np.ones(100)
    sample_graph.Up = np.ones(100)  
    sample_graph.Ek = np.ones(100)
    sample_graph.plot_energy()

    assert saved_filepath[0] == 'output/E_test_file.png'

# plot_forces test
def test_plot_forces(sample_graph, sample_MDstep2,  monkeypatch):
    """
    Test case for the 'plot_forces' method of the 'graph' class.
    This test checks if the 'plot_forces' method correctly generates and saves the forces plot.
    """
    # Mocking the plt.savefig function to check if it's called with the correct arguments
    saved_filepath = []

    def mock_savefig(filepath):
        saved_filepath.append(filepath)

    monkeypatch.setattr(plt, 'savefig', mock_savefig)

    # Set count to non-zero values for the test
    sample_graph.time = [1]
    sample_graph.F = [[1, 1, 1]]
    sample_MDstep2.groups[0].Ftot_store = [[1, 1, 1]]
    sample_MDstep2.groups[1].Ftot_store = [[1, 1, 1]]
    sample_graph.plot_forces(sample_MDstep2)

    assert saved_filepath[0] == 'output/F_test_file.png'

# plot_velocity test
def test_plot_velocity(sample_graph, sample_MDstep2,  monkeypatch):
    """
    Test case for the 'plot_velocity' method of the 'graph' class.
    This test checks if the 'plot_velocity' method correctly generates and saves the velocity plot.
    """
    # Mocking the plt.savefig function to check if it's called with the correct arguments
    saved_filepath = []

    def mock_savefig(filepath):
        saved_filepath.append(filepath)

    monkeypatch.setattr(plt, 'savefig', mock_savefig)

    # Set count to non-zero values for the test
    sample_graph.time = [1, 2]
    sample_MDstep2.groups[0].Vtot_store = [[1, 1, 1]]
    sample_MDstep2.groups[1].Vtot_store = [[1, 1, 1]]
    sample_graph.plot_velocity(sample_MDstep2)

    assert saved_filepath[0] == 'output/V_test_file.png'

# plot_temperature test
def test_plot_temperature(sample_graph, sample_MDstep2,  monkeypatch):
    """
    Test case for the 'plot_temperature' method of the 'graph' class.
    This test checks if the 'plot_temperature' method correctly generates and saves the temperature plot.
    """
    # Mocking the plt.savefig function to check if it's called with the correct arguments
    saved_filepath = []

    def mock_savefig(filepath):
        saved_filepath.append(filepath)

    monkeypatch.setattr(plt, 'savefig', mock_savefig)

    # Set count to non-zero values for the test
    sample_graph.time = [1, 2]
    sample_graph.T = [10, 20]
    sample_MDstep2.groups[0].T = [5, 10]
    sample_MDstep2.groups[1].T = [5, 10]
    sample_graph.plot_temperature(sample_MDstep2)

    assert saved_filepath[0] == 'output/T_test_file.png'

