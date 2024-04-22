import numpy as np
import os
import matplotlib.pyplot as plt

class RDF:
    """
    Parameters
    ----------
    Rmax : float
        Maximum value for R.
    atoms : list
        List of atoms.
    N_bin : int
        Number of bins for histogram.
    filename : str
        Name of the file.
    outdir : str
        Directory where the output files will be saved.

    Attributes
    ----------
    filename : str
        Name of the file.
    outdir : str
        Directory where the output files will be saved.
    Rmax : float
        Maximum value for R.
    type : list
        List of atoms.
    N_bin : int
        Number of bins.
    count : numpy.ndarray
        Array of zeros with size N.
    R : numpy.ndarray
        Array containing values from 0 to Rmax with N elements.
    dR : float
        Value of the second element in R.
    norm : numpy.ndarray
        Normalized array based on R values.
    equal : bool
        Condition based on the equality of the two elements in atoms.
    at1 : numpy.ndarray
        Bidimensional array for atoms type 1.
    N1 : int
        Number of elements in at1.
    at2 : numpy.ndarray
        Bidimensional array for atoms type 2.
    N2 : int
        Number of elements in at2.

    Methods:
    --------
    RDF(iteration_obj):
        Calculate the Radial Distribution Function.
    normalization(iteration_obj):
        Normalize RDF counts based on the system volume and atom counts.
    plot_RDF():
        Plot the Radial Distribution Function.
    """

    def __init__(self, Rmax : float, atoms : list, N_bin : int, filename : str, outdir : str):
        """
        Initialize RDF istance.

        Parameters
        ----------
        Rmax : float
            Maximum value for R.
        atoms : list
            List of atoms.
        N_bin : int
            Number of bins for histogram.
        filename : str
            Name of the file.
        outdir : str
            Directory where the output files will be saved.

        Attributes
        ----------
        filename : str
            Name of the file.
        outdir : str
            Directory where the output files will be saved.
        Rmax : float
            Maximum value for R.
        type : list
            List of atoms.
        N_bin : int
            Number of bins.
        count : numpy.ndarray
            Array of zeros with size N.
        R : numpy.ndarray
            Array containing values from 0 to Rmax with N elements.
        dR : float
            Value of the second element in R.
        norm : numpy.ndarray
            Normalized array based on R values.
        equal : bool
            Condition based on the equality of the two elements in atoms.
        at1 : numpy.ndarray
            Bidimensional array for atoms type 1.
        N1 : int
            Number of elements in at1.
        at2 : numpy.ndarray
            Bidimensional array for atoms type 2.
        N2 : int
            Number of elements in at2.
        """

        self.filename = filename
        self.outdir = outdir
        self.Rmax = Rmax
        self.type = atoms
        self.N_bin = N_bin
        self.count = np.zeros(N_bin)
        self.R = np.linspace(0, Rmax, N_bin)
        self.dR = self.R[1]
        self.norm = np.multiply([((i + self.dR)**3 - i**3) for i in self.R[:N_bin]], np.pi * 4 /3)
        self.equal = (atoms[0] == atoms[1])
        self.at1 = np.array([], dtype=float).reshape(0, 3)
        self.N1 = 0
        self.at2 = np.array([], dtype=float).reshape(0, 3)
        self.N2 = 0

        self.RDF_color = 'black'

    def RDF(self, MDstep_obj) -> None:
        """
        Calculate the radial distribution function (RDF) for the group.

        Parameters
        ----------
        MDstep_obj : MDstep
            An MDstep object representing the iteration and containing necessary information.

        Notes
        -----
        This method calculates the distances between the two defined atomic species and adds them to the counting 
        list 'count'. The periodic conditions are used to account for the minimum image criterion. In order to do 
        so, reducted coordinates are also used. If the atomic species are the same, the method considers the 
        distances between atoms of the same type, and if they are different, it calculates the distances between 
        atoms of different types.
        """

        dist = []

        # Equal atomic species
        if self.equal:
            n = len(self.at1)
            rpos = self.at1
            
            # Generate the reducted coordinates
            for i, _pos in enumerate(self.at1):
                    rpos[i] = np.dot(np.linalg.inv(MDstep_obj.matrix), _pos)

            # Calculate the distances
            for k in range(n - 1):
                rdiff = rpos[k+1:] - rpos[k]
                int_pos = np.rint(rdiff)
                rdiff = rdiff - int_pos
                diff = rdiff.copy()
                for i, _rpos in enumerate(rdiff):
                        diff[i] = np.dot(MDstep_obj.matrix, _rpos)
                        r = np.linalg.norm(diff[i])
                        dist.extend(r[r < self.Rmax])

        # Different atomic species
        else:
            rpos1 = self.at1
            rpos2 = self.at2
            
            # Generate the reducted coordinates
            for i, _pos in enumerate(self.at1):
                rpos1[i] = np.dot(np.linalg.inv(MDstep_obj.matrix), _pos)
            for i, _pos in enumerate(self.at2):
                rpos2[i] = np.dot(np.linalg.inv(MDstep_obj.matrix), _pos)

            # Calculate the distances
            for i in rpos1:
                rdiff = i - rpos2
                rdiff = rdiff - np.rint(rdiff)
                diff = rdiff
                for i, _rpos in enumerate(rdiff):
                    diff[i] = np.dot(MDstep_obj.matrix, _rpos)
                    r = np.linalg.norm(diff[i])
                    dist.extend(r[r < self.Rmax])

        self.N1 = len(self.at1)
        self.N2 = len(self.at2)
        self.count += np.histogram(dist, bins=self.N_bin, range=(0, self.Rmax))[0]

        # Reset the arrays for the next time step
        self.at1 = np.array([], dtype=float).reshape(0, 3)
        self.at2 = np.array([], dtype=float).reshape(0, 3)

    def normalization(self, MDstep_obj) -> None:
        """
        Normalize the radial distribution function (RDF).

        Parameters
        ----------
        MDstep_obj : MDstep
            An MDstep object representing the iteration and containing necessary information.

        Notes
        -----
        This method calculates the volume 'V' and uses it to normalize the radial distribution function 'count' by 
        dividing it by the appropriate density ('rho_1').
        """
        
        V = 8 * self.Rmax **3

        if self.equal :
            rho_1 = V / (self.N1 * (self.N1 - 1) * MDstep_obj.N_iteration)
        else:
            rho_1 = V / (self.N1 * self.N2 * MDstep_obj.N_iteration)

        self.count = np.divide(self.count, self.norm) * rho_1
        
    def plot_RDF(self) -> None:
        """
        Plot the radial distribution function (RDF).

        Generates a plot of the RDF based on calculated radial distances and counts and saves it as 
        'RDF_{element1}_{element2}_{filename}.pdf'.

        Notes
        -----
        This method uses Matplotlib to create an RDF plot. The radial distribution function is plotted against 
        radial distances ('r') in angstroms. The plot is saved in PDF format. The plot includes information about 
        the elements used in the RDF calculation.
        """

        fig = plt.figure(figsize=(10, 6.18033988769))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.R, self.count, color = self.RDF_color, label=f'{self.type[0]}-{self.type[1]}')
        ax.set_xlabel('r ($\AA$)')
        ax.set_ylabel('g(r)')
        ax.grid()
        ax.legend()
        filepath = os.path.join(self.outdir, f'{self.type[0]}{self.type[1]}_{self.filename}.pdf')
        plt.savefig(filepath)

