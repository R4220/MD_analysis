import numpy as np

class atom:
   
    """
    This class represents an atomic species used in molecular dynamics simulations.
    This class sets up an 'Atom' object with the specified attributes, including the atom's name, mass, and the identification number of its group.

    Parameters
    ----------
    name : str
        Atomic species.
    mass : float
        Mass of the atomic species.
    id_group : int
        Identification number of the group to which the atoms belong.

    Attributes
    ----------
    name : str
        Atomic species.
    mass : float
        Mass of the atomic species.
    id_group : int
        Identification number of the group to which the atoms belong.
    N : int
        Total count of atoms of the current atomic species.
    id : list
        List of atoms IDs.
    position_past_past : ndarray
        Array representing the second past positions of the atom.
    position_past : ndarray
        Array representing the past positions of the atom.
    position : ndarray
        Array representing the current positions of the atom.
    velocity : ndarray
        Array representing the velocity of the atom.
    force : ndarray
        Array representing the forces acting on the atom.
    """


    def __init__(self, name: str, mass: float, id_group: int):
        
        '''
        This constructor sets up an 'Atom' object with the specified attributes, including the atom's name, mass, and the identification number of its group.
        Additionally, it initializes other attributes like 'DOF', 'N', 'id', 'position_past_past', 'position_past', 'position', 'velocity', and 'force' with default values.

        Parameters
        ----------
        name : str
            Atomic species.
        mass : float
            Mass of the atomic species.
        id_group : int
            Identification number of the group to which the atoms belong.

        Attributes
        ----------
        name : str
            Atomic species.
        mass : float
            Mass of the atomic species.
        id_group : int
            Identification number of the group to which the atoms belong.
        N : int
            Total count of atoms of the current atomic species.
        id : list
            List of atoms IDs.
        position_past_past : ndarray
            Array representing the second past positions of the atom.
        position_past : ndarray
            Array representing the past positions of the atom.
        position : ndarray
            Array representing the current positions of the atom.
        velocity : ndarray
            Array representing the velocity of the atom.
        force : ndarray
            Array representing the forces acting on the atom.
        '''
        self.name = name
        self.mass = mass
        self.id_group = id_group
        self.N = 0
        self.id = []

        self.position_past_past = np.array([], dtype=float).reshape(0, 3)
        self.position_past = np.array([], dtype=float).reshape(0, 3)
        self.position = np.array([], dtype=float).reshape(0, 3)
        self.velocity = np.array([], dtype=float).reshape(0, 3)
        self.force = np.array([], dtype=float).reshape(0, 3)
        
        
    def generate_velocity(self, dt: float) -> None:
        '''
        Generate the bidimensional array representing the velocity of each atom.
        Velocities are calculated as the difference between the current positions and the positions from the second-to-last time step, divided by the time interval.

        Parameters
        ----------
        dt : float
            Time interval.
        '''
        self.velocity = (np.array(self.position, dtype=float) - np.array(self.position_past_past, dtype=float)) / (2 * dt)
