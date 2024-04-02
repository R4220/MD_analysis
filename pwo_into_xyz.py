# main
import numpy as np
import shutil
import os
import sys
from configparser import ConfigParser

from Codes.class_MDstep import MDstep
from Codes.class_group import group
from Codes.class_graph import graph


def configuration(input_file):
    '''
    Read configuration parameters from the specified input file and initialize simulation setup.

    Parameters
    ----------
    input_file : str
        The path to the input configuration file.

    Returns
    -------
    tuple
        A tuple containing the following information:
        - filename (str): The name of the file.
        - outdir (str): The output directory.
        - Rmax (list): The maximum distances for radial distribution function calculation.
        - atoms (list): The list of couplesof atoms for RDFs.
        - N (list): The number of bins for the RDFs.
        - groups (list): A list of group instances initialized based on configuration.
        - filepath (str): The filepath for input file.

    Notes
    -----
    This function reads configuration parameters from the specified input file and initializes the simulation setup.
    It extracts information such as the filename, output directory, maximum distance for RDFs, particle types for 
    RDF calculations, number of bins for RDFs, and group instances.
    '''
    config = ConfigParser()

    config.read(input_file)

    filepath = config['SETUP']['filename']
    filename = filepath.split('/')[-1]
    outdir = config['SETUP']['Outdir']

    groups = []
    ids = config["INTERFACE SEPARATION"]["Groups"]
    for gr in config['GROUPS']:
        group_ = group(config['GROUPS'][gr].split(), gr)
        if gr in ids:
            group_.distance_switch = True
        groups = np.append(groups, group_)
    
    atoms = []
    Rmax = []
    N = np.array([], int)
    for rdf in config["RDF"]:
        couple = config["RDF"][rdf].split()
        atoms = np.append(atoms, [couple[0], couple[1]])
        Rmax = np.append(Rmax, float(couple[2]))
        N = np.append(N, int(couple[3]))
    atoms = atoms.reshape(int(len(atoms)/2), 2)
    
    try:
        graph_file = config['GRAPHS']['filename']
    except:
        graph_file = False

    if not os.path.exists(outdir):
            os.makedirs(outdir)
    shutil.copy2(input_file, f'{outdir}')

    return filename, outdir, Rmax, atoms, N, groups, filepath, graph_file


def preamble(fin, step_obj : MDstep) -> None:
    '''
    Process the preamble of the input file and set up the MDstep object.

    Parameters
    ----------
    fin : _io.TextIOWrapper
        The input file.
    step_obj : MDstep
        The MDstep object to be initialized.

    Notes
    -----
    This function processes the preamble of the input file, updating the MDstep object attributes based on the information extracted.
    '''
    preamble_switch = [True] * 8
    matrix_switch = [True] * 3
    while any(preamble_switch):
        line = fin.readline()

        # Extraction of the number of atom in the cell
        if 'number of atoms/cell' in line:
            step_obj.n_atoms = int(line.split()[4])
            preamble_switch[0] = False

        # Extraction of the number of atomic types 
        elif 'number of atomic types' in line:
            step_obj.n_type = int(line.split()[5])
            preamble_switch[1] = False

        # Extraction of the celldim parameter
        elif 'celldm(1)= ' in line:
            step_obj.alat_to_angstrom = float(line.split()[1]) * 0.529177249
            preamble_switch[2] = False

        # Extraction of the first lattice vector
        elif 'a(1)' in line:
            x, y, z = map(float, line.split()[3:6])
            a = np.multiply([x, y, z], step_obj.alat_to_angstrom)
            step_obj.ax = a
            preamble_switch[3] = False
            matrix_switch[0] = False

        # Extraction of the second lattice vector
        elif 'a(2)' in line:
            x, y, z = map(float, line.split()[3:6])
            a = np.multiply([x, y, z], step_obj.alat_to_angstrom)
            step_obj.ay = a
            preamble_switch[4] = False
            matrix_switch[1] = False

        # Extraction of the third lattice vector
        elif 'a(3)' in line:
            x, y, z = map(float, line.split()[3:6])
            a = np.multiply([x, y, z], step_obj.alat_to_angstrom)
            step_obj.az = a
            preamble_switch[5] = False
            matrix_switch[2] = False
        
        # Setting of the mass of the atoms inside the groups
        elif 'atomic species   valence' in line:
            for _ in range(step_obj.n_type):
                line = fin.readline().split()
                step_obj.set_mass(line[0], line[2])
            preamble_switch[6] = False

        # Extraction of the index of atoms and the initial positions
        elif 'site n.' in line:
            for _ in range(step_obj.n_atoms):
                line = fin.readline()
                step_obj.count_group(line.split())
            preamble_switch[7] = False
            step_obj.set_DOF()
        
        if not any(matrix_switch):
            step_obj.matrix = np.column_stack((step_obj.ax, step_obj.ay, step_obj.az))
            matrix_switch = [True] * 3


def extract_forces(fin, step_obj : MDstep) -> None:
    '''
    Extract forces from the input file and update the forces in the MDstep object.

    Parameters
    ----------
    fin : _io.TextIOWrapper
        The input file.
    line : str
        The current line from the input file.
    step_obj : MDstep
        The MDstep object to be updated with forces.


    Notes
    -----
    This function reads the input file from the current line until the end of the forces section.
    For each line containing force information, it updates the forces in the MDstep object.
    '''
    i = 0
    while i < step_obj.n_atoms:
        line = fin.readline()
        if 'force =' in line:
            step_obj.forces(line.split())
            i += 1  


def extract_positions(fin, step_obj : MDstep, RDF_obj, switch : bool) -> None:
    '''
    Extract positions from the input file and update the positions in the MDstep object and RDF histogram.

    Parameters
    ----------
    fin : _io.TextIOWrapper
        The input file.
    line : str
        The current line from the input file.
    step_obj : MDstep
        The MDstep object to be updated with positions.
    RDF_obj : RDF
        The RDF histogram object to be updated with positions.
    switch : bool
        It defines if the conversion from alat unito into angstrom is neaded

    Notes
    -----
    This function reads the input file for the specified number of atoms, extracting their positions.
    For each line containing position information, it updates the positions in the MDstep object and, if applicable, in the RDF histogram.
    '''
    for _ in range(step_obj.n_atoms):
        line = fin.readline().split()
        if switch:
            step_obj.positions(line, RDF_obj, 1.)
        else:
            step_obj.positions(line, RDF_obj, step_obj.alat_to_angstrom)


def body(fout, fin, step_obj : MDstep, graphs : graph) -> None:
    '''
    Extract relevant information from the input file and write output to the output file.

    Parameters
    ----------
    fin : _io.TextIOWrapper
        The output file.
    fin : _io.TextIOWrapper
        The input file.
    step_obj : MDstep
        The MDstep object containing system information.
    graphs : graph
        The graph object in which the plotting of the graphs happens.

    Notes
    -----
    This function reads the input file line by line, extracting relevant information such as potential energy, forces on atoms, time step, iteration number, and atomic positions.
    It updates the information in the MDstep object and writes the corresponding output to the output file.
    Additionally, it calculates the radial distribution function and adds the parameter of the timestep to draw the graph of energies, temperature, and forces over time.

    '''
    generation_switch = [True] * 5
    for line in fin:
        
        # Extraction of the potential energy
        if '!    total energy' in line:
            step_obj.U_pot = float(line.split()[4]) * 13.605703976
            generation_switch[0] = False

        # Extraction of the forces on atoms
        elif 'Forces a' in line:
            extract_forces(fin, step_obj)
            generation_switch[1] = False

        # Extraction of the time step
        elif 'Time step' in line:
            step_obj.dt = float(line.split()[3])  * 4.8378e-5 #ps

        # ExtractionMDstep number
        elif 'Entering' in line:
             step_obj.N_iteration = int(line.split()[4])
             generation_switch[2] = False
             
        # Extraction of the atomic position
        elif 'ATOMIC_POSITIONS' in line:
            extract_positions(fin, step_obj, graphs, 'angstrom' in line)
            generation_switch[3] = False
        
        elif 'temperature    ' in line:
            graphs.T = np.append(graphs.T, float(line.split()[2]))
            generation_switch[4] = False
        
        # Generation of the the text printed on the output file, performing the RDF calculation and extract the values for graphs
        if not any(generation_switch):
            fout.writelines(["%s\n" % i for i in step_obj.single_frame()])
            graphs.extracting_values(step_obj)
            for couples in graphs.type:
                couples.RDF(step_obj)
            generation_switch = [True] * 5


def xyz_gen(fout, fin, RDF_ : list, groups : list, outdir : str, graph_file) -> None:
    '''
    Generate an XYZ file from a PWO file, extracting relevant information and updating the output file.

    Parameters
    ----------
    fout : file
        The output XYZ file object.
    fin : file
        The input PWO file object.
    RDF_ : list
        List containing information needed for RDF calculation.
    groups : list
        List of group instances in the system.
    distance : list
        List of group istances for wich the mean distance is calculated.
    outdir : str
        The output directory for saving files.

    Notes
    -----
    This function checks the input PWO file to extract all the values and generates the corresponding output XYZ file.
    It initializes the iteration object and RDF histogram object, extracts setup information, and then iterates through the file to extract and update information.
    The RDF is calculated, normalized, and plotted at the end within the other graphs.
    '''
    MDstep_obj = MDstep(groups)

    # Extraction the setup informations
    preamble(fin, MDstep_obj)
    
    # Setting of the RDF object
    graphs = graph(RDF_[0], RDF_[1], RDF_[2], RDF_[3], outdir)
    graphs.graph_aesthetic(graph_file)    
    
    # Generation of the configuration at each time
    body(fout, fin, MDstep_obj, graphs)

    for couples in graphs.type:
        couples.normalization(MDstep_obj)
        couples.plot_RDF()
    graphs.plot_energy()
    graphs.plot_forces(MDstep_obj)
    graphs.plot_temperature(MDstep_obj)
    graphs.plot_velocity(MDstep_obj)
    graphs.plot_distance()


if __name__ == "__main__":
    # Extract setup information
    
    _, input_file = sys.argv

    if not os.path.exists(input_file):
        print("File not found")
        exit(0)

    filename, outdir, Rmax, atoms, N, groups, filepath, graph_file = configuration(input_file)
    RDF_ = [filename, Rmax, atoms, N]

    # Open output and input files
    with open(os.path.join(outdir, filename + '.xyz'), "w+") as fout:
        with open(filepath + '.pwo', 'r') as fin:
            # Generate XYZ file from PWO file
            xyz_gen(fout, fin, RDF_, groups, outdir, graph_file)
             