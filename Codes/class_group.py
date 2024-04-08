import numpy as np

class group:
    '''
    This class represents a group of atoms used to divide the whole atoms in a molecular dynamics simulation.

    Parameters
    ----------
    type : str
        Atomic type in the group.
    id_group : str
        Identification name of the group.

    Attributes
    ----------
    atoms : list
        List of Atom instances in the group.
    type : str
        Atomic types in the group.
    id_group : str
        Identification name of the group.
    id_tot : ndarray
        Array containing the identification numbers of the atoms inside the group.
    DOF : float
        Degree of freedom of the group.
    Ek : float
        Kinetic energy of the group.
    T : list
        Temperatures of the group.
    Ftot : ndarray
        Total force acting on the group.
    force : ndarray
        Bidimensional array representing the forces against the atoms in the group.
    Ftot_store : ndarray
        Bidimensional array representig the total force against the atopms of the group, for each time step
    Vtot_store : ndarray
        Bidimensional array representig the bulk velocity of the atoms in the group , for each time step
    Vtot : ndarray
        Bidimensional array representing the group velocity of the atoms in the group.
    velocity : ndarray
        Bidimensional array representing the velocity of the atoms in the group.
    velocity_switch : boolean
        Switch to avoid the velocity generation for the first iteration.
    distance_switch : boolean
        Switch to select the group for which the distance is calculated.

    Methods
    -------
    __init__(type, id_group)
        Initialize a Group instance.
    Kinetic_energy(dt)
        Calculate the kinetic energy of the group.
    Extract_z()
        Extract the mean z-coordinate of the group.
    Generate(dt, matrix)
        Generate output data for the group.

    Notes
    -----
    This class provides functionality to simulate molecular dynamics at the group level, including methods to add 
    atoms, calculate kinetic energy, and generate output data.
    '''

    def __init__(self, type: str, id_group: str):
        '''
        Initialize a Group instance.

        Parameters
        ----------
        type : str
            Atomic type in the group.
        id_group : str
            Identification name of the group.

        Attributes
        ----------
        atoms : list
            List of Atom instances in the group.
        type : str
            Atomic types in the group.
        id_group : str
            Identification name of the group.
        id_tot : ndarray
            Array containing the identification numbers of the atoms inside the group.
        DOF : float
            Degree of freedom of the group.
        Ek : float
            Kinetic energy of the group.
        T : list
            Temperatures of the group.
        Ftot : ndarray
            Total force acting on the group.
        force : ndarray
            Bidimensional array representing the forces against the atoms in the group.
        Ftot_store : ndarray
            Bidimensional array representig the total force against the atopms of the group, for each time step
        Vtot_store : ndarray
            Bidimensional array representig the bulk velocity of the atoms in the group , for each time step
        Vtot : ndarray
            Bidimensional array representing the group velocity of the atoms in the group.
        velocity : ndarray
            Bidimensional array representing the velocity of the atoms in the group.
        velocity_switch : boolean
            Switch to avoid the velocity generation for the first iteration.
        distance_switch : boolean
            Switch to select the group for which the distance is calculated.

        Notes
        -----
        This constructor sets up a 'Group' object with the specified attributes, including the group's type and 
        the identification number of the group.
        Additionally, it initializes other attributes like 'atoms', 'id_tot', 'DOF', 'Ek', 'Ftot', 'force', 
        'Vtot', and 'velocity' with default values.
        '''

        self.atoms = []
        self.type = type
        self.id_group = id_group
        self.id_tot = np.array([], dtype=int)
        self.DOF = 0.0
        self.Ek = 0.0
        self.T = []

        self.Ftot = np.array([], dtype=float).reshape(0, 3)
        self.force = np.array([], dtype=float).reshape(0, 3)
        self.Ftot_store = np.array([], dtype=float).reshape(0, 3)

        self.Vtot_store = np.array([], dtype=float).reshape(0, 3)
        self.Vtot = np.array([], dtype=float).reshape(0, 3)
        self.velocity = np.array([], dtype=float).reshape(0, 3)
        self.velocity_switch = False

        self.distance_switch = False

    def Kinetic_energy(self, dt: float) -> None:
        '''
        Calculate the kinetic energy of the group.

        Parameters
        ----------
        dt : float
            Time interval.

        Notes
        -----
        This method calculates the kinetic energy of the group based on the velocities of its atoms. It iterates 
        over each atom in the group, generates the velocity array for each atom type, and computes the total 
        velocity ('Vtot') of the atoms in the group. Then, it removes the mean velocity and calculates the thermal
        kinetic energy ('Ek') using the formula:

        .. math::

            E_k = \\frac{1}{2} \\sum_{i} m_i \\left\\| \\mathbf{v}_i - \\mathbf{V}_{\\text{tot}} \\right\\|^2

        where:
        - \( m_i \) is the mass of the atom.
        - \( \\mathbf{v}_i \) is the velocity vector of the atom.
        - \( \\mathbf{V}_{\\text{tot}} \) is the total velocity of the atoms in the group.

        The calculated kinetic energy is stored in the 'Ek' attribute.
        '''

        self.velocity = np.array([], dtype=float).reshape(0, 3)

        # Generate the velocity array for each atom type
        for at in self.atoms:
            at.generate_velocity(dt)
            self.velocity =  np.vstack([self.velocity, at.velocity])

        # Calculate the mean velocity
        self.Vtot = np.sum(self.velocity, axis=0) / len(self.id_tot)
        self.Vtot_store = np.append(self.Vtot_store, self.Vtot.reshape(1,3), axis=0)

        # Removing the mean velocity to calculate the thermal energy
        self.Ek = 0.
        for at in self.atoms:
            self.Ek += 0.5 * float(at.mass) * np.sum(np.linalg.norm(at.velocity - self.Vtot, axis=1) ** 2) * 0.000103642694841594
        
    def Extract_z(self) -> float:
        '''
        Extract the mean z-coordinate of the group.

        Returns
        -------
        float
            the mean z-coordinate.

        Notes
        -----
        This method calculates the mean z-coordinate by extracting all the z-coordinates from the atoms inside the
        group and then averaging the values.
        '''

        z = []
        for at in self.atoms:
            z = np.append(z, at.position[:, 2])
        return np.mean(z)
        
    def Generate(self, dt: float, matrix: np.ndarray) -> np.ndarray:
        '''
        Generate output data for the group.

        Parameters
        ----------
        dt : float
            Time interval.
        matrix : numpy.ndarray
            Transformation matrix for periodic boundary conditions.

        Returns
        -------
        numpy.ndarray
            Array containing output data for each atom in the group.

        Notes
        -----
        This method calculates the kinetic energy, temperature, and total force of the group. It generates an 
        output line for each atom in the group, including information such as the atom's name, position, velocity, 
        force, and group identification number.
        It also calculates the difference between two previously defined groups.
        To calculate the temperature the method uses the formula:

        .. math::

            T = \\frac{2 E_k}{DOF * K_B}

        where:
        - \( E_k \) is the kinetic energy.
        - \( DOF \) is the number of degrees of freedom.
        - \( K_B \) is the Boltzmann constant in eV/K.
        '''

        # Calculate the kinetic energy
        if self.velocity_switch :
            self.Kinetic_energy(dt)
        else :
            for at in self.atoms:
                at.velocity = np.array([0] * 3 * at.N, dtype=float).reshape(at.N, 3)
            self.velocity_switch = True

        # Calculate temperature
        self.T = np.append(self.T, ((2 * self.Ek) / (self.DOF * 8.617333262145e-5)))

        # Calculate total force
        self.Ftot = np.sum(self.force, axis=0)
        self.Ftot_store = np.append(self.Ftot_store, self.Ftot.reshape(1,3), axis=0)

        # Generate output line
        body = []
        z = 0
        for at in self.atoms:
            rpos = np.dot(np.linalg.inv(matrix), at.position.T).T
            rpos[:, :2] = rpos[:, :2] - np.rint(rpos[:, :2])
            pos = np.dot(matrix, rpos.T).T
            for i in range(at.N):
                    body = np.append(body, f'{at.name}\t  {pos[i][0]}\t  {pos[i][1]}\t  {pos[i][2]}\t  {at.velocity[i][0]}\t  {at.velocity[i][1]}\t  {at.velocity[i][2]}\t  {at.force[i][0]}\t  {at.force[i][1]}\t  {at.force[i][2]}\t  {at.id_group}')

            if self.distance_switch:
                z = self.Extract_z()

        # Reset the arrays for the next time step
            at.position_past_past = at.position_past
            at.position_past = at.position
            at.position = np.array([], dtype=float).reshape(0, 3)
            at.force = np.array([], dtype=float).reshape(0, 3)
        self.force = np.array([], dtype=float).reshape(0, 3)
        return body, z
    