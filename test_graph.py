import numpy as np
import matplotlib.pyplot as plt
import pytest
from configparser import ConfigParser
from unittest.mock import patch
import os
from io import StringIO
from Codes.class_MDstep import MDstep
from Codes.class_group import group
from Codes.class_graph import graph
from Codes.class_RDF import RDF 

@pytest.fixture
def sample_graph():
    """
    Fixture that provides a sample 'graph' object for testing.
    This fixture initializes a 'graph' object with specific attributes for testing purposes.

    Returns
    -------
    graph
        A 'graph' object initialized with the following attributes:
        - filename: "test_file"
        - Rmax: [5.0, 6.0]
        - atoms: [['A', 'B'], ['C', 'D']]
        - N_bin: [100, 120]
        - outdir: "output"
    """

    filename = "test_file"
    Rmax = [5.0, 6.0]
    atoms = [['A', 'B'], ['C', 'D']]
    N_bin = [100, 120]
    outdir = "output"
    return graph(filename, Rmax, atoms, N_bin, outdir)

# initialization --------------------------------------------------------------------------------------------------
    
def test_graph_initialization(sample_graph):
    """
    Test function to verify the initialization of the 'sample_graph' object.

    Parameters
    ----------
    sample_graph : graph
        A sample 'graph' object initialized for testing.

    Raises
    ------
    AssertionError
        If any of the expected attributes or configurations are not initialized correctly.

    Notes
    -----
    This test checks if the 'sample_graph' object is initialized properly, including the creation of 'RDF' objects.
    It verifies the initialization of attributes such as filename, outdir, Ek, Up, T, F, time, and the creation 
    of 'RDF' objects within the 'type' attribute.

    Raises an AssertionError if any of the expected attributes or configurations are not initialized correctly.
    """
    
    # Check attributes of the 'graph' object
    assert sample_graph.filename == "test_file"
    assert sample_graph.outdir == "output"
    assert np.array_equal(sample_graph.Ek, [])
    assert np.array_equal(sample_graph.Up, [])
    assert np.array_equal(sample_graph.T, [])
    assert np.array_equal(sample_graph.F, np.array([], dtype=float).reshape(0, 3))
    assert np.array_equal(sample_graph.time, [])

    # Check the correct creation of the 'RDF' objects
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


# graph_esthetic test ---------------------------------------------------------------------------------------------

def test_graph_aesthetic(sample_graph):
    """
    Test function to verify the 'graph_aesthetic' method of the 'graph' class.

    Parameters
    ----------
    sample_graph : graph
        A sample 'graph' object initialized for testing.

    Raises
    ------
    AssertionError
        If any of the expected parameters or configurations are not correctly updated.

    Notes
    -----
    This test function executes the 'graph_aesthetic' method of the 'graph' class, which reads parameters from 
    the provided configuration file and updates Matplotlib's default parameters accordingly. It then checks if 
    the parameters are updated correctly.

    Specifically, the test checks:
    - If the method updates the label sizes for axes correctly.
    - If the method updates the label sizes for x-axis ticks correctly.
    - If the method updates the label sizes for y-axis ticks correctly.
    - If the method updates the font size for legend correctly.
    - If the method updates the RDF color for each RDF object correctly.
    - If the method updates the flag for plotting total energy correctly.
    - If the method updates the colors for different energy components correctly.
    - If the method updates the colors for different groups correctly.
    """

    sample_graph.graph_aesthetic('Test/files/graph_aesthetic.ini')

    assert plt.rcParams['axes.labelsize'] == 20.0
    assert plt.rcParams['xtick.labelsize'] == 14.0
    assert plt.rcParams['ytick.labelsize'] == 14.0
    assert plt.rcParams['legend.fontsize'] == 14.0

    for rdf in sample_graph.type:
        assert rdf.RDF_color == '#425840' 
    assert sample_graph.all_energy == True
    assert np.array_equal(sample_graph.energy_color, ['#b32323', '#191971', '#006400'])
    assert np.array_equal(sample_graph.group_color, ['#000000', '#9ACD32', '#b32323', '#191971', '#006400', '#b0edef', '#470303'])

def test_graph_aesthetic_default(sample_graph):
    """
    Test function to verify the 'graph_aesthetic' method of the 'graph' class.

    Parameters
    ----------
    sample_graph : graph
        A sample 'graph' object initialized for testing.

    Raises
    ------
    AssertionError
        If any of the expected parameters or configurations are not correctly updated.

    Notes
    -----
    This test checks if the 'graph_aesthetic' method correctly reads and updates the default parameters based on 
    the default configuration file.

    Specifically, the test checks:
    - If the method updates the label sizes for axes correctly.
    - If the method updates the label sizes for x-axis ticks correctly.
    - If the method updates the label sizes for y-axis ticks correctly.
    - If the method updates the font size for legend correctly.
    - If the method updates the default RDF color correctly.
    - If the method updates the flag for plotting total energy correctly.
    - If the method updates the default colors for different energy components correctly.
    - If the method updates the default colors for different groups correctly.
    """

    sample_graph.graph_aesthetic(False)

    assert plt.rcParams['axes.labelsize'] == 16.0
    assert plt.rcParams['xtick.labelsize'] == 14.0
    assert plt.rcParams['ytick.labelsize'] == 14.0
    assert plt.rcParams['legend.fontsize'] == 14.0

    assert sample_graph.RDF_color == 'black'
    assert sample_graph.all_energy == False
    assert np.array_equal(sample_graph.energy_color, ['red', 'blue', 'black'])
    assert np.array_equal(sample_graph.group_color, ['red', 'blue', 'green', 'yellow', 'black', 'purple'])

def test_graph_aesthetic_not_existing(sample_graph, capfd):
    """
    Test function to verify the 'graph_aesthetic' method of the 'graph' class.

    Parameters
    ----------
    sample_graph : graph
        A sample 'graph' object initialized for testing.
    capfd : fixture
        A fixture provided by pytest to capture stdout and stderr.

    Raises
    ------
    AssertionError
        If any of the expected parameters or configurations are not correctly updated.

    Notes
    -----
    This test checks if the 'graph_aesthetic' method correctly handles the case of not existing file.
    """

    # Verify that the function correctly handles reading a non-existent file
    with pytest.raises(SystemExit):
        sample_graph.graph_aesthetic('false.ini')

    # Capture the output and check if the error message is printed
    out, _ = capfd.readouterr()
    assert "File not found" in out


# extracting_values test -----------------------------------------------------------------------------------------
    
def test_extracting_values_Force1():
    """
    Test function to verify the 'extracting_values' method of the 'graph' class.
    This test validates whether the 'extracting_values' method correctly computes and stores the total force.
    
    Raises
    ------
    AssertionError
        If the total force values extracted by the 'extracting_values' method do not match the expected values.

    Notes
    -----
    This test checks if the 'extracting_values' method correctly calculates the total force acting on the system.
    It does so by creating instances of the 'group' class representing different atom groups in the system
    and simulating their total force values. The 'extracting_values' method of the 'graph' class is then called
    with an instance of 'MDstep' containing these groups. Finally, the test verifies that the total force 
    values calculated and stored by the 'extracting_values' method match the expected values.
    """

    # Create instances of 'group' class
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])

    # Create an instance of 'MDstep' class
    iteration = MDstep([group1, group2])

    # Create an instance of 'graph' class
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the instance of MDstep
    graph_instance.extracting_values(iteration)

    # Verify that the values have been correctly calculated and stored
    assert np.array_equal(graph_instance.F, np.array([[5.0, 7.0, 9.0]]))

def test_extracting_values_Force2():
    """
    Test function to verify the 'extracting_values' method of the 'graph' class.
    In this test, we check if the method correctly calculates and stores the total force also considering negative components of the force.
    
    Raises
    ------
    AssertionError
        If the total force values extracted by the 'extracting_values' method do not match the expected values.

    Notes
    -----
    This test checks if the 'extracting_values' method correctly calculates the total force acting on the system,
    taking into account both positive and negative components of the force. It does so by creating instances 
    of the 'group' class representing different atom groups in the system and simulating their total force values. 
    The 'extracting_values' method of the 'graph' class is then called with an instance of 'MDstep' containing 
    these groups. Finally, the test verifies that the total force values calculated and stored by the 
    'extracting_values' method match the expected values.
    """

    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, -5.0, 6.0])

    iter = MDstep([group1, group2])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the instance of MDstep
    graph_instance.extracting_values(iter)

    # Verify that the values have been correctly extracted and stored
    assert np.array_equal(graph_instance.F, np.array([[5.0, -3.0, 9.0]]))

def test_extracting_values_Ek():
    """
    Test function to verify the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' method correctly calculates and stores the total kinetic energy of the system.
    
    Raises
    ------
    AssertionError
        If the total kinetic energy value extracted by the 'extracting_values' method does not match the expected value.

    Notes
    -----
    This test checks if the 'extracting_values' method correctly calculates the total kinetic energy of the system.
    It does so by creating instances of the 'group' class representing different atom groups in the system
    and simulating their total kinetic energy values. The 'extracting_values' method of the 'graph' class is then called
    with an instance of 'MDstep' containing these groups. Finally, the test verifies that the total kinetic energy 
    value calculated and stored by the 'extracting_values' method matches the expected value.
    """

    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])
    group1.Ek = 10.0
    group2.Ek = 15.0

    iter = MDstep([group1, group2])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the instance of MDstep
    graph_instance.extracting_values(iter)

    # Verify that the value has been correctly extracted and stored
    assert graph_instance.Ek[-1] == 25.0

def test_extracting_values_Upot():
    """
    Test function to verify the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' method correctly calculates and stores the potential energy of the system.

    Raises
    ------
    AssertionError
        If the potential energy value extracted by the 'extracting_values' method does not match the expected value.

    Notes
    -----
    This test checks if the 'extracting_values' method correctly calculates the potential energy of the system.
    It does so by creating instances of the 'group' class representing different atom groups in the system.
    The 'extracting_values' method of the 'graph' class is then called with an instance of 'MDstep' containing
    these groups, along with a simulated potential energy value. Finally, the test verifies that the potential energy
    value calculated and stored by the 'extracting_values' method matches the expected value.
    """

    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    iteration = MDstep([group1, group2])
    iteration.U_pot = 30.0
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the instance of MDstep
    graph_instance.extracting_values(iteration)

    # Verify that the value has been correctly extracted and stored
    assert graph_instance.Up[-1] == 30.0

def test_extracting_values_time():
    """
    Test function to verify the 'extracting_values' method of the 'graph' class.
    This test checks if the 'extracting_values' method correctly calculates and stores the time corresponding to the current time step.
    
    Raises
    ------
    AssertionError
        If the time value extracted by the 'extracting_values' method does not match the expected value.

    Notes
    -----
    This test checks if the 'extracting_values' method correctly calculates the time corresponding to the current time step.
    It does so by creating instances of the 'group' class representing different atom groups in the system.
    The 'extracting_values' method of the 'graph' class is then called with an instance of 'MDstep' containing
    these groups, along with simulated time-related values. Finally, the test verifies that the time value
    calculated and stored by the 'extracting_values' method matches the expected value.
    """
    
    group1 = group(type=['H', 'C'], id_group=0)
    group2 = group(type=['Fe'], id_group=1)
    
    # Define values to simulate
    iteration = MDstep([group1, group2])
    iteration.dt = 10.0
    iteration.N_iteration = 50
    group1.Ftot = np.array([1.0, 2.0, 3.0])
    group2.Ftot = np.array([4.0, 5.0, 6.0])

    # Create an instance of 'graph'
    graph_instance = graph("test_file", [5.0, 6.0], [['H', 'H'], ['C', 'Fe']], [100, 120], "output")

    # Call the extracting_values method with the instance of MDstep
    graph_instance.extracting_values(iteration)

    # Verify that the value has been correctly extracted and stored
    assert graph_instance.time[-1] == 500
