import numpy as np
import os
import matplotlib.pyplot as plt


class RDF: # da modificare con i nuovi grafici
    """
    Radial Distribution Function (RDF) calculator.

    Parameters:
    -----------
    filename : str
        The filename for saving the plot.
    Rmax : float
        Maximum distance for RDF calculation.
    atoms : list of str
        Atom types for which RDF is calculated.
    N : int
        Number of bins for histogram.
    outdir: str
        The directory where the output files will be saved.

    Attributes:
    -----------
    filename : str
        The filename for saving the plot.
    outdir : str
        Directory where the output files will be saved.
    Rmax : float
        Maximum distance for RDF calculation.
    type : list of str
        Atom types for which RDF is calculated.
    N : int
        Number of bins for histogram.
    count : ndarray
        Histogram counts for RDF.
    R : ndarray
        Radial distance array.
    dR : float
        Bin size for histogram.
    norm : ndarray
        Normalization array for RDF calculation.
    condition : bool
        True if RDF is calculated for the same type of atoms, False otherwise.
    at1 : ndarray
        Array to store positions of atoms of type 1.
    N1 : int
        Number of atoms of type 1.
    at2 : ndarray
        Array to store positions of atoms of type 2.
    N2 : int
        Number of atoms of type 2.

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
        condition : bool
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
        self.condition = (atoms[0] == atoms[1])
        self.at1 = np.array([], dtype=float).reshape(0, 3)
        self.N1 = 0
        self.at2 = np.array([], dtype=float).reshape(0, 3)
        self.N2 = 0

        self.RDF_color = 'black'
        self.graph_aesthetic()
        

    def graph_aesthetic(self) -> None:
        '''
        Set aesthetic parameters for Matplotlib plots based on a configuration file.

        This method reads parameters from a configuration file named 'Setup_graph.txt' and updates Matplotlib's default parameters accordingly.
        The configuration file should include specifications for graph parameters such as label sizes, tick sizes, legend font size, and colors.

        Parameters:
        -----------
        self : graph
            An instance of the 'graph' class.

        Notes:
        ------
        This method reads the 'Setup_graph.txt' file and extracts information regarding graph aesthetics, including label sizes, tick sizes, legend font size, and colors.
        It then updates Matplotlib's default parameters to reflect the specified aesthetics.

        The 'Setup_graph.txt' file should contain lines specifying the following:
        - axes.labelsize: Label size for axes.
        - xtick.labelsize: Label size for x-axis ticks.
        - ytick.labelsize: Label size for y-axis ticks.
        - legend.fontsize: Font size for legend.
        - RDF_color: Color for RDF plot.
        '''
        # Default values for label sizes and legend font size
        param = [16, 14, 14, 14] 

        # Read graph parameters from the configuration file
        with open('Setup_graph.txt', 'r') as fgraph:
            for line in fgraph:

                # Graph parameters
                if 'axes.labelsize' in line:
                    param[0] = line.split()[2]
                elif 'xtick.labelsize' in line:
                    param[1] = line.split()[2]
                elif 'ytick.labelsize' in line:
                    param[2] = line.split()[2]
                elif 'legend.fontsize' in line:
                    param[3] = line.split()[2]
                
                # Colors
                elif 'RDF_color' in line:
                    self.RDF_color = line.split()[2]
        
        # Update Matplotlib's default parameters
        parameters = {'axes.labelsize': param[0], 'xtick.labelsize': param[1], 'ytick.labelsize': param[2], 'legend.fontsize': param[3]}
        plt.rcParams.update(parameters)


    def RDF(self, MDstep_obj) -> None:
        """
        Calculate the radial distribution function (RDF) for the group.

        Parameters
        ----------
        MDstep_obj : MDstep
            An MDstep object representing the iteration and containing necessary information.

        Notes
        -----
        This method calculates the distances between the two defined atomic species and adds them to the counting list 'count'.
        The periodic conditions are used to account for the minimum image criterion. In order to do so, reducted coordinates are also used.
        If the atomic species are the same, the method considers the distances between atoms of the same type, and if they are different, it calculates the distances between atoms of different types.
        """
        dist = []

        # Equal atomic species
        if self.condition:
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
        This method calculates the volume 'V' and uses it to normalize the radial distribution function 'count' by dividing it by the appropriate density ('rho_1').
        """
        V = 8 * self.Rmax **3

        if self.condition:
            rho_1 = V / (self.N1 * (self.N1 - 1) * MDstep_obj.N_iteration)
        else:
            rho_1 = V / (self.N1 * self.N2 * MDstep_obj.N_iteration)

        self.count = np.divide(self.count, self.norm) * rho_1
        

    def plot_RDF(self) -> None:
        """
        Plot the radial distribution function (RDF).

        Generates a plot of the RDF based on calculated radial distances and counts and saves it as 'RDF_{element1}_{element2}_{filename}.png'.

        Notes
        -----
        This method uses Matplotlib to create an RDF plot. The radial distribution function is plotted against radial distances ('r') in angstroms.
        The plot is saved in PNG format. The plot includes information about the elements used in the RDF calculation.
        """
        fig = plt.figure(figsize=(10, 6.18033988769))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.R, self.count, color = self.RDF_color, label=f'{self.type[0]}-{self.type[1]}')
        ax.set_xlabel('r ($\AA$)')
        ax.set_ylabel('g(r)')
        ax.grid()
        ax.legend()
        filepath = os.path.join(self.outdir, f'{self.type[0]}{self.type[1]}_{self.filename}.png')
        plt.savefig(filepath)
