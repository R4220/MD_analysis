import numpy as np
from Codes.class_atom import atom

class MDstep:
    '''
    Represents a single step during molecular dynamics.

    Parameters
    ----------
    groups : list
        List of group instances in the system.

    Attributes
    ----------
    groups : list
        List of group instances in the system.
    n_atoms : int
        Number of atoms in the system.
    n_type : int
        Number of different atom types in the system.
    ax, ay, az : numpy.ndarray
        Lattice vectors.
    U_pot : float
        Potential energy of the system.
    dt : float
        Time interval.
    N_iteration : int
        Number of iterations.
    alat_to_angstrom : float
        Conversion factor from alat unit to angstrom.
    Ryau_to_pN : float
        Conversion factor from Rydberg units to piconewtons.
    matrix : numpy.ndarray
        3x3 transformation matrix for periodic boundary conditions.

    Methods
    -------
    set_mass(_type, _mass)
        Set the mass of atoms in the specified group.
    count_group(line)
        Count the number of atoms for each group and extract the initial position.
    set_DOF()
        Set the degrees of freedom for each group in the system.
    forces(line)
        Extract the force acting on atoms at a specific time.
    positions(line, graphs)
        Extract and store the positions of atoms at a determined time.
    single_frame()
        Generate a list where each element is a line in the output file, representing the current time step.

    '''

    def __init__(self, groups : list):
        '''
        Initialize a System object.

        Parameters
        ----------
        groups : list
            List of group instances in the system.

        Attributes
        ----------
        groups : list
            List of group instances in the system.
        n_atoms : int
            Number of atoms in the system.
        n_type : int
            Number of different atom types in the system.
        ax : numpy.ndarray
            Vector representing the 'ax' lattice vector.
        ay : numpy.ndarray
            Vector representing the 'ay' lattice vector.
        az : numpy.ndarray
            Vector representing the 'az' lattice vector.
        U_pot : float
            Potential energy of the system.
        dt : float
            Time interval.
        N_iteration : int
            Number of iterations.
        alat_to_angstrom : float
            Conversion factor from alat unit to angstrom.
        Ryau_to_pN : float
            Conversion factor from Rydberg units to piconewtons.
        matrix : numpy.ndarray
            3x3 transformation matrix for periodic boundary conditions.

        Notes
        -----
        This constructor sets up a 'MolecularDynamicsStep' object with the specified attributes, including the list of group instances in the system, the number of atoms, the number of different atom types, lattice vectors ('ax', 'ay', 'az'), time interval, and conversion factors.
        The elements of the list 'group' will be updated at each cycle, together with the potential energy and the number of iterations.
        '''
        self.groups = groups
        self.n_atoms = 0
        self.n_type = 0
        self.ax = np.zeros(3)
        self.ay = np.zeros(3)
        self.az = np.zeros(3)
        self.U_pot = 0.0
        self.dt = 0.0
        self.N_iteration = 0
        self.alat_to_angstrom = 0.0
        self.Ryau_to_pN = 41193.647367103644
        self.matrix = []


    def set_mass(self, _type : str, _mass : float) -> None:
        '''
        Set the mass of atoms in the specified group.

        Parameters
        ----------
        _type : str
            Name of the atom.
        _mass : float
            Mass of the atom.

        Notes
        -----
        This method iterates over the groups in the system and adds atoms with the specified name and mass.
        '''
        for gr in self.groups:
            if _type in gr.type:
                gr.atoms = np.append(gr.atoms, atom(_type, _mass, gr.id_group))


    def count_group(self, line : list) -> None:
        '''
        Count the number of atoms for each group and extract the initial position.

        Parameters
        ----------
        line : list
            The read line of the file.

        Notes
        -----
        This method iterates over the groups in the system, counts the number of atoms for each group, and extracts the initial position of the atoms.
        '''
        atom_type = line[1] 
        # Check on the groups
        for gr in self.groups:
            if atom_type in gr.type:
                gr.id_tot = np.append(gr.id_tot, int(line[0]))
                # Check on the group's elements
                for at in gr.atoms:
                    if atom_type == at.name:
                        at.N += 1
                        at.id = np.append(at.id, int(line[0]))
                        pos = np.multiply([float(line[-4]), float(line[-3]), float(line[-2])], self.alat_to_angstrom)
                        at.position_past = np.vstack([at.position_past, pos])
                        break
                break


    def set_DOF(self) -> None:
        '''
        Set the degrees of freedom for each group in the system.

        Notes
        -----
        This method iterates over the groups in the system, calculates the degrees of freedom based on the total number of atoms in the group, and adjusts it if the group's `id_group` is not equal to 0.
        '''
        for gr in self.groups:
            gr.DOF = 3 * len(gr.id_tot)
            if not gr.id_group == 0:
                gr.DOF = gr.DOF - 3


    def forces(self, line : list) -> None:
        '''
        Extract the force acting on atoms at a specific time.

        Parameters
        ----------
        line : list
            The line containing the force acting on the atom.

        Notes
        -----
        This method iterates over the groups in the system, finds the corresponding atom, and adds the force to the atom and group.
        '''
        atom_type = int(line[1])
        # Check on the groups
        for gr in self.groups:
            if atom_type in gr.id_tot:
                gr.force = np.vstack([gr.force, np.multiply([float(line[6]), float(line[7]), float(line[8])], self.Ryau_to_pN)])
                
                # Check on the group's elements
                for at in gr.atoms:
                    if atom_type in at.id:
                        at.force = np.vstack([at.force, np.multiply([float(line[6]), float(line[7]), float(line[8])], self.Ryau_to_pN)])
                        break
                break


    def positions(self, line : list, graphs, conversion : float) -> None:
        '''
        Extract and store the positions of atoms at a determined time.

        Parameters
        ----------
        line : list
            The line containing the atom positions.
        graphs : graph
            The graph instance with which the radial distribution function is calculated.
        conversion : float
            The conversion for the coordinates.

        Notes
        -----
        This method iterates over the groups in the system, finds the corresponding atom, and adds the position to the atom.
        Additionally, it checks the atom type and adds positions to the RDF element into the graph istance 'graphs' (if applicable).
        '''
        atom_type = line[0] 

        pos = np.multiply([float(line[1]), float(line[2]), float(line[3])], conversion)

        # Check if for the atom in line RDF will be calculated
        for couples in graphs.type:
            if couples.type[0] in atom_type:
                couples.at1 = np.vstack([couples.at1, pos])
        for couples in graphs.type:
            if couples.type[1] in atom_type:
                couples.at2 = np.vstack([couples.at2, pos])
        
        # Check on the groups
        for gr in self.groups:
            if atom_type in gr.type:
                # Check on the group's elements
                for at in gr.atoms:
                    if atom_type == at.name:
                        at.position = np.vstack([at.position, pos])
                        break
                break
        

    def single_frame(self) -> list:
        '''
        Generate a list where each element is a line in the output file, representing the current time step.

        Returns
        -------
        text : list
            The list of lines.

        Notes
        -----
        This method prints information about the lattice, time step, number of iterations, and potential energy in the second line of the xyz file.
        It iterates over the groups in the system, adding information about kinetic energy, degrees of freedom, temperature, and total force.
        At the end, it adds the coordinates, velocity, and forces of each atom using the 'generate' method of the group.
        '''    
        # Generate the general information of the time step
        text = [f'{self.n_atoms}', f'Lattice(Ang)=\"{self.ax[0]:.3f}, {self.ax[1]:.3f}, {self.ax[2]:.3f}, {self.ay[0]:.3f}, {self.ay[1]:.3f}, {self.ay[2]:.3f}, {self.az[0]:.3f}, {self.az[1]:.3f}, {self.az[2]:.3f}\" dt(ps)={self.dt:.6f} N={self.N_iteration} Epot(eV)={(self.U_pot / 13.60570398):.3f}']

        # Generate the information for each group  of the time step
        for gr in self.groups:            
            body = gr.Generate(self.dt, self.matrix)
            text[1] = text[1] + f' Ek{gr.id_group}(eV)={gr.Ek:.3f} DOF{gr.id_group}={gr.DOF} T{gr.id_group}(K)={gr.T[-1]:.3f} Ftot{gr.id_group}(pN)=\"{gr.Ftot[0]:.3f}, {gr.Ftot[1]:.3f}, {gr.Ftot[2]:.3f}\"'
            text.extend(body)

        return text
    