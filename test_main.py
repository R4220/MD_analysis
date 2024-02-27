import os
import pytest
import numpy as np
from io import StringIO
import shutil
from Codes.class_MDstep import MDstep
from Codes.class_group import group
from Codes.class_graph import graph
from Codes.class_RDF import RDF
from pwo_into_xyz import setup, preamble, extract_forces, extract_positions, body, xyz_gen


@pytest.fixture
def sample_MDstep():
    return MDstep(groups=[])


# setup test
    
def test_setup(monkeypatch):
    """
    Test case for the 'setup' function.
    This test checks if the 'setup' function correctly extract the values from the setup file.
    """
    def fake_copy2(src, dst):
        pass
    monkeypatch.setattr(shutil, 'copy2', fake_copy2)
    fake_config_content = '\n'.join(['Filename: folder/test_file', 'Outdir: output', 'Group 0', 'H C', 'Group 1', 'Fe', '# RDF', 'Particles: Fe Fe 7 500'])
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: StringIO(fake_config_content))

    filename, outdir, Rmax, atoms, N, groups, filepath = setup()

    assert filename == 'test_file'
    assert outdir == 'output'
    assert Rmax == 7.0
    assert np.array_equal(atoms, [['Fe', 'Fe']])
    assert N == 500
    assert filepath == 'folder/test_file'

    # testing the groups
    assert groups[0].id_group == 0
    assert groups[0].type == ['H', 'C']
    assert groups[1].id_group == 1
    assert groups[1].type == ['Fe']


# preamble test
    
def test_preamble(sample_MDstep, monkeypatch):
    """
    Test case for the 'preamble' function.
    This test checks if the 'preamble' function correctly extract the values from the first part of the 'pwo' file.
    """
    with open('test_preamble_file.txt', 'w') as file:
        file.write('number of atoms/cell      =           6\n')
        file.write('number of atomic types    =            3\n')
        file.write('celldm(1)=   1.000000  celldm(2)=   0.000000  celldm(3)=   0.000000\n')
        file.write('a(1) = (   2.000000   1.000000   0.000000 )  \n')
        file.write('a(2) = (   2.000000  -1.000000   0.000000 )\n')
        file.write('a(3) = (   0.000000   0.000000   6.000000 )\n')
        file.write('atomic species   valence    mass     pseudopotential\n')
        file.write('H              1.00     1.01000     H ( 1.00)\n')
        file.write('C              4.00    12.00000     C ( 1.00)\n')
        file.write('Fe             8.00     55.84700     Fe( 1.00)\n')
        file.write(' site n.     atom                  positions (alat units)\n')
        file.write('count_group\n')
        file.write('count_group\n')
        file.write('count_group\n')
        file.write('count_group\n')
        file.write('count_group\n')
        file.write('count_group\n')
        file.write('\n')

    def fake_count_group(a, b):
        pass
    monkeypatch.setattr(MDstep, 'count_group', fake_count_group)

    def fake_set_DOF(a):
        pass
    monkeypatch.setattr(MDstep, 'set_DOF', fake_set_DOF)

    gr0 = group(['H', 'C'], 0)
    gr1 = group(['Fe'], 1)
    sample_MDstep.groups = [gr0, gr1]

    with open('test_preamble_file.txt', 'r') as file:
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

    expected_matrix = np.array([[2 * 0.529177249, 2 * 0.529177249, 0], [1 * 0.529177249, -1 * 0.529177249, 0], [0, 0, 6 * 0.529177249]])
    assert np.array_equal(sample_MDstep.matrix, expected_matrix)
    
# extract_forces test
    
def test_extract_forces(sample_MDstep, monkeypatch):
    """
    Test case for the 'extract_force' function.
    This test checks if the 'extract_force' function correctly extract the force values from the the 'pwo' file.
    """
    with open('test_extract_forces_file.txt', 'w') as file:
        file.write('convergence has been achieved in  79 iterations\n')
        file.write('Forces acting on atoms (Ry/au):\n')
        file.write('\n')
        file.write('\n')
        file.write('     negative rho (up, down):  0.212E-02 0.135E-02\n')
        file.write('     atom    1 type  1   force =    -0.00994417   -0.00097200    0.00430182\n')
        file.write('     atom    2 type  1   force =    -0.00146776   -0.00269919   -0.02873328\n')
        file.write('     atom    3 type  1   force =    -0.00693958    0.01240992   -0.01368338\n')
        file.write('     atom    4 type  1   force =     0.00216298    0.00555048   -0.00262624\n')
        file.write('     atom    5 type  1   force =     0.01411466   -0.00877770    0.00041559\n')
        file.write('stop\n')
        file.write('forces\n')
        file.write('\n')
        file.write('\n')
        file.write('\n')
        file.write('\n')
        file.write('\n')
        file.write('\n')
    sample_MDstep.n_atoms = 5

    global array 
    array = []
    def fake_forces(fin, MDstep):
        global array
        array = np.append(array, True)
    monkeypatch.setattr(MDstep, 'forces', fake_forces)

    with open('test_extract_forces_file.txt', 'r') as fin:
        extract_forces(fin, sample_MDstep)
        next_line = fin.readline()
        assert 'stop' in next_line
        for line in fin:
            assert not 'force =' in line
    
    assert np.array_equal(array, [True] * 5)
    

# estract_positions test
    
def test_extract_positions(sample_MDstep, monkeypatch):
    """
    Test case for the 'extract_positions' function.
    This test checks if the 'extract_positions' function correctly extract the positions from the the 'pwo' file.
    """
    with open('test_extract_positions_file.txt', 'w') as file:
        file.write('H 1 2 3\n')
        file.write('H 1 2 3\n')
        file.write('H 1 2 3\n')
        file.write('H 1 2 3\n')
        file.write('H 1 2 3\n')
        file.write('stop\n')
    sample_MDstep.n_atoms = 5

    global array 
    array = []
    def fake_positions(a, line, graphs, bool):
        global array
        array = np.append(array, True)
    monkeypatch.setattr(MDstep, 'positions', fake_positions)

    graphs = 'graph'#graph('filename', [1], ['A', 'B'], [5], 'outdir')

    with open('test_extract_positions_file.txt', 'r') as fin:
        extract_positions(fin, sample_MDstep, graph, True)
        assert 'stop' in fin.readline()
    
    assert np.array_equal(array, [True] * 5)


# body test
    
def test_body(sample_MDstep, monkeypatch):
    """
    Test case for the 'body' function.
    This test checks if the 'body' function correctly extract the values and write on the 'pwo' file.
    """
    with open('test_extract_positions_file.txt', 'w') as file:
        file.write('!    total energy              =   1 Ry\n')
        file.write('     Forces acting on atoms (Ry/au):\n')
        file.write('     Time step             =    1.00 a.u.,  1.9351 femto-seconds\n')
        file.write('     Entering Dynamics:    MDstep =     3\n')
        file.write('ATOMIC_POSITIONS (angstrom)\n')
        file.write('     temperature           =   50 K \n')
        file.write('stop\n')
    sample_MDstep.n_atoms = 5

    global array 
    array = []

    def fake_extract_forces(fin, MDstep):
        global array
        array = np.append(array, True)
    monkeypatch.setattr('pwo_into_xyz.extract_forces', fake_extract_forces)

    def fake_extract_positions(a, line, graphs, bool):
        global array
        array = np.append(array, True)
    monkeypatch.setattr('pwo_into_xyz.extract_positions', fake_extract_positions)

    def fake_single_frame(a):
        return ['single frame']
    monkeypatch.setattr(MDstep, 'single_frame', fake_single_frame)

    def fake_extracting_values(a, iter):
        pass
    monkeypatch.setattr(graph, 'extracting_values', fake_extracting_values)

    def fake_RDF(a, iter):
        pass
    monkeypatch.setattr(RDF, 'RDF', fake_RDF)

    graphs = graph('filename', [1], [['A', 'B']], [5], 'outdir')

    with open('test_extract_positions_file.txt', 'r') as fin:
        with open('test_output', 'w') as fout:
            body(fout, fin, sample_MDstep, graphs)
    
    with open('test_output', 'r') as fout:
        line = fout.readline()
        assert line == 'single frame\n'

    
    assert sample_MDstep.U_pot == 13.605703976
    assert sample_MDstep.dt == 4.8378e-5
    assert sample_MDstep.N_iteration == 3
    assert graphs.T == 50
    assert np.array_equal(array, [True] * 2)
    