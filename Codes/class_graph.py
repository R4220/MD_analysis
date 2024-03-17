import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Codes.class_RDF import RDF

class graph: 
    """
    This class extracts the values to plot graphs and plot them.

    Attributes:
    -----------
    filename : str
        Name of the file.
    outdir : str
        Directory where the output files will be saved.
    type : list
        List containing instances of RDF class for each specified pair of atoms.
    Ek : list
        List to store kinetic energy values.
    Up : list
        List to store potential energy values.
    T : list
        List to store temperature values.
    F : numpy.ndarray
        Bidimensional array to store force values.
    time : list
        List to store time values.
    all_energy : bool
        Flag to indicate whether to include all energy components.
    RDF_color : str
        Color for the Radial Distribution Function (RDF) plot.
    energy_color : list
        List of colors for different energies.
    group_color : list
        List of colors for groups.

    Methods:
    --------
    graph_aesthetic():
        Set aesthetic parameters for Matplotlib plots based on a configuration file.
    extracting_values(iteration_obj):
        Extract and store relevant values from the iteration object.
    plot_energy():
        Plot the graph of kinetic energy and potential energy in function of time.
    plot_forces(iteration_obj):
        Plot the forces acting on the system in function of time.
    plot_velocity(iteration_obj):
        Plot the velocity in function of time.
    plot_temperature():
        Plot the temperature in function of time.
    plot_distance():
        Plot the distance between two groups in function of time.
    """


    def __init__(self, filename : str, Rmax : list, atoms : list, N_bin : list, outdir : str): 
        """
        Initialize an instance of the 'Class_Name' class.

        Parameters
        ----------
        filename : str
            Name of the file.
        Rmax : list
            Maximum value for R for the RDFs.
        atoms : list
            List of couples of atoms.
        N : list
            Number of bins for the RDFs.
        outdir : str
            Directory where the output files will be saved.

        Attributes
        ----------
        filename : str
            Name of the file.
        outdir : str
            Directory where the output files will be saved.
        type : list
            List containing instances of RDF class for each specified pair of atoms.
        Ek : list
            List to store kinetic energy values.
        Up : list
            List to store potential energy values.
        T : list
            List to store temperature values.
        F : numpy.ndarray
            Bidimensional array to store force values.
        distances : list
            List of distances between the selected groups.
        time : list
            List to store time values.
        all_energy : bool
            Flag to indicate whether to include all energy components.
        RDF_color : str
            Color for the Radial Distribution Function (RDF) plot.
        energy_color : list
            List of colors for different energies.
        group_color : list
            List of colors for groups.

        Notes
        -----
        This class is designed to represent a certain type of object, providing various attributes for configuration and calculations.
        """
        self.filename = filename
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        self.type = []
        for i, r in enumerate(Rmax):
            self.type = np.append(self.type, RDF(r, [atoms[i][0], atoms[i][1]], N_bin[i], filename, outdir))

        self.Ek = []
        self.Up = []
        self.T = []
        self.F = np.array([], dtype=float).reshape(0, 3)
        self.distances = []

        self.time = []

        self.all_energy = False
        self.RDF_color = 'black'
        self.energy_color = ['red', 'blue', 'black']
        self.group_color = ['red', 'blue', 'green', 'yellow', 'black', 'purple']

        

    def graph_aesthetic(self) -> None:
        '''
        Set aesthetic parameters for Matplotlib plots based on a configuration file.

        This method reads parameters from a configuration file named 'Setup_graph.txt' and updates Matplotlib's default parameters accordingly.
        The configuration file should include specifications for graph parameters such as label sizes, tick sizes, legend font size, and colors.

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
        - Energy_sum: Flag indicating whether to plot total energy.
        - K_energy_color: Color for kinetic energy plot.
        - U_energy_color: Color for potential energy plot.
        - Tot_energy_color: Color for total energy plot.
        - Group_color: Colors for different groups in force and temperature plots
        '''
        param = [16, 14, 14, 14] # Default values for label sizes and legend font size

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
                    color = line.split()[2]
                
                    for rdf in self.type:
                        rdf.RDF_color = color

                elif 'Energy_sum' in line:
                    self.all_energy = bool(line.split()[2])
                elif 'K_energy_color' in line:
                    self.energy_color[0] = line.split()[2]
                elif 'U_energy_color' in line:
                    self.energy_color[1] = line.split()[2]
                elif 'Tot_energy_color' in line:
                    self.energy_color[2] = line.split()[2]

                elif 'Group_color' in line:
                    _line = line.split()[2:]
                    self.group_color = _line
        
        # Update Matplotlib's default parameters
        parameters = {'axes.labelsize': param[0], 'xtick.labelsize': param[1], 'ytick.labelsize': param[2], 'legend.fontsize': param[3]}
        plt.rcParams.update(parameters)


    def extracting_values(self, step_obj) -> None:
        """
        Extract and store relevant values from the iteration object.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method extracts and stores forces (`Ftot`), kinetic energy (`Ek`), potential energy (`Up`),
        total force (`F`), distances between the groups defined in the setup (`distances`) and simulation time (`time`) from the provided `step_obj` for each group in the system.
        """
        F = np.array([], dtype=float).reshape(0, 3)
        self.Ek = np.append(self.Ek, 0)
        for gr in step_obj.groups:
            F = np.append(F, gr.Ftot.reshape(1,3), axis=0)
            self.Ek[-1] += gr.Ek
                        
        self.Up = np.append(self.Up, step_obj.U_pot)
        self.distances = np.append(self.distances, step_obj.dist)
        self.F = np.append(self.F, np.sum(F, axis=0).reshape(1, 3), axis=0)
        self.time = np.append(self.time, step_obj.dt * step_obj.N_iteration)
        

    def plot_energy(self) -> None:
        """
        Plot the graph of kinetic energy and potential energy in function of time.

        Generates a plot of the kinetic energy and potential energy as a function of time and saves it as an image file named 'E_{filename}.png'.

        Notes
        -----
        This method uses Matplotlib to create a plot of the kinetic energy and potential energy over time. The plot is saved as an image file in PNG format.
        """
        fig = plt.figure(figsize=(10, 6.18033988769))
        axE = fig.add_subplot(1, 1, 1)
        axU = axE.twinx()
        
        axU.plot(self.time[1:], self.Up[1:] * 0.001, color = self.energy_color[1], label='Potential energy')
        axE.plot(self.time[1:], self.Ek[1:], color = self.energy_color[0],  label='Kinetic energy')
        if self.all_energy:
            axU.plot(self.time[1:], (self.Up[1:] + self.Ek[1:]) * 0.001, color = self.energy_color[2], label='Total energy')
        axE.set_xlabel('t (ps)')
        axE.set_ylabel('E$_k$ (eV)')
        axE.legend(loc = 'upper left')
        xticks = axE.get_xticks()
        axE.set_xticks(xticks)
        axE.set_xticklabels([str(int(tick)) for tick in xticks])
        axE.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axE.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axE.get_yticks()
        axE.set_yticks(yticks)
        axE.set_yticklabels([str(int(tick)) for tick in yticks])
        axE.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        axU.set_ylabel('U (keV)')
        axU.legend(loc = 'lower right')
        yticks = axU.get_yticks()
        axU.set_yticks(yticks)
        axU.set_yticklabels([str(int(tick)) for tick in yticks])
        axU.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        filepath = os.path.join(self.outdir, f'E_{self.filename}.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)


    def plot_forces(self, step_obj) -> None:
        """
        Plot the forces acting on the system in function of time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the forces (components along x, y, and z acting on the system over time) and saves it as an image file named 'F_{filename}.png'.
        """
        fig = plt.figure(figsize=(10, 3*6.18033988769 +2))
        
        #ax = fig.add_subplot(1, 1, 1)
        Fx = []
        Fy = []
        Fz = []
        for f in self.F:
                Fx = np.append(Fx, f[0])
                Fy = np.append(Fy, f[1])
                Fz = np.append(Fz, f[2])

        axX = fig.add_subplot(3, 1, 1)
        axX.plot(self.time, Fx * 0.001, color = self.group_color[0], label='F$^{tot}_x$')
        
        axZ = fig.add_subplot(3, 1, 3)
        axZ.plot(self.time, Fz * 0.001, color = self.group_color[0],  label='F$^{tot}_z$')

        axY = fig.add_subplot(3, 1, 2)
        axY.plot(self.time, Fy * 0.001, color = self.group_color[0],  label='F$^{tot}_y$')

        for i, gr in enumerate(step_obj.groups):
            Fx = []
            Fy = []
            Fz = []
            for f in gr.Ftot_store:
                Fx = np.append(Fx, f[0])
                Fy = np.append(Fy, f[1])
                Fz = np.append(Fz, f[2])
            axX.plot(self.time, Fx * 0.001, color = self.group_color[i+1], alpha = 0.5, label=f'{gr.id_group}$_x$')
            axY.plot(self.time, Fy * 0.001, color = self.group_color[i+1], alpha = 0.5, label=f'{gr.id_group}$_y$')
            axZ.plot(self.time, Fz * 0.001, color = self.group_color[i+1], alpha = 0.5, label=f'{gr.id_group}$_z$')

        axX.set_xlabel('t (ps)')
        axX.set_ylabel('F$_x$ (nN)')
        axX.grid()
        axX.legend()
        xticks = axX.get_xticks()
        axX.set_xticks(xticks)
        axX.set_xticklabels([str(int(tick)) for tick in xticks])
        axX.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axX.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axX.get_yticks()
        axX.set_yticks(yticks)
        axX.set_yticklabels([str(int(tick)) for tick in yticks])
        axX.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        axY.set_xlabel('t (ps)')
        axY.set_ylabel('F$_y$ (nN)')
        axY.grid()
        axY.legend()
        xticks = axY.get_xticks()
        axY.set_xticks(xticks)
        axY.set_xticklabels([str(int(tick)) for tick in xticks])
        axY.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axY.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axY.get_yticks()
        axY.set_yticks(yticks)
        axY.set_yticklabels([str(int(tick)) for tick in yticks])
        axY.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        axZ.set_xlabel('t (ps)')
        axZ.set_ylabel('F$_z$ (nN)')
        axZ.grid()
        axZ.legend()
        xticks = axZ.get_xticks()
        axZ.set_xticks(xticks)
        axZ.set_xticklabels([str(int(tick)) for tick in xticks])
        axZ.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axZ.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axZ.get_yticks()
        axZ.set_yticks(yticks)
        axZ.set_yticklabels([str(int(tick)) for tick in yticks])
        axZ.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        filepath = os.path.join(self.outdir, f'F_{self.filename}.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)


    def plot_velocity(self, step_obj) -> None:
        """
        Plot the bulk velocity of the groups in function of time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the bulk velocity (components along x, y, and z acting on the system over time) and saves it as an image file named 'F_{filename}.png'.
        """
        fig = plt.figure(figsize=(10, 3*6.18033988769 +2))
        axX = fig.add_subplot(3, 1, 1)
        axZ = fig.add_subplot(3, 1, 3)
        axY = fig.add_subplot(3, 1, 2)

        for i, gr in enumerate(step_obj.groups):
            Vx = []
            Vy = []
            Vz = []
            for v in gr.Vtot_store:
                Vx = np.append(Vx, v[0])
                Vy = np.append(Vy, v[1])
                Vz = np.append(Vz, v[2])
            axX.plot(self.time[1:], Vx, color = self.group_color[i+1], alpha = 0.5, label=f'{gr.id_group}$_x$')
            axY.plot(self.time[1:], Vy, color = self.group_color[i+1], alpha = 0.5, label=f'{gr.id_group}$_y$')
            axZ.plot(self.time[1:], Vz, color = self.group_color[i+1], alpha = 0.5, label=f'{gr.id_group}$_z$')

        axX.set_xlabel('t (ps)')
        axX.set_ylabel('V$_x$ ($\AA$/ps)')
        axX.grid()
        axX.legend()
        xticks = axX.get_xticks()
        axX.set_xticks(xticks)
        axX.set_xticklabels([str(int(tick)) for tick in xticks])
        axX.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axX.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axX.get_yticks()
        axX.set_yticks(yticks)
        axX.set_yticklabels([str(int(tick)) for tick in yticks])
        axX.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        axY.set_xlabel('t (ps)')
        axY.set_ylabel('V$_y$ ($\AA$/ps)')
        axY.grid()
        axY.legend()
        xticks = axY.get_xticks()
        axY.set_xticks(xticks)
        axY.set_xticklabels([str(int(tick)) for tick in xticks])
        axY.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axY.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axY.get_yticks()
        axY.set_yticks(yticks)
        axY.set_yticklabels([str(int(tick)) for tick in yticks])
        axY.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        axZ.set_xlabel('t (ps)')
        axZ.set_ylabel('V$_z$ ($\AA$/ps)')
        axZ.grid()
        axZ.legend()
        xticks = axZ.get_xticks()
        axZ.set_xticks(xticks)
        axZ.set_xticklabels([str(int(tick)) for tick in xticks])
        axZ.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axZ.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axZ.get_yticks()
        axZ.set_yticks(yticks)
        axZ.set_yticklabels([str(int(tick)) for tick in yticks])
        axZ.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        filepath = os.path.join(self.outdir, f'V_{self.filename}.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)


    def plot_temperature(self, step_obj) -> None:
        """
        Plot the temperature of the system in function of time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the temperature acting on the system over time
        and saves it as an image file named 'T_{filename}.png'.
        """
        fig = plt.figure(figsize=(10, 6.18033988769))
        
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.time[1:], self.T[1:], color = self.group_color[0], label='T$^{tot}$')
        for i, gr in enumerate(step_obj.groups):
            ax.plot(self.time[1:], gr.T[1:], color = self.group_color[i+1], alpha = 0.5, label=f'G. {gr.id_group}')

        ax.set_xlabel('t (ps)')
        ax.set_ylabel('T (K)')
        ax.grid()
        ax.legend(loc = 'upper left')
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(tick)) for tick in xticks])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(tick)) for tick in yticks])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        filepath = os.path.join(self.outdir, f'T_{self.filename}.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)

    def plot_distance(self) -> None:
        """
        Plot the distance between the two defined groups in function of time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the distance between the two defined groups over time
        and saves it as an image file named 'Dist_{filename}.png'.
        """
        fig = plt.figure(figsize=(10, 6.18033988769))
        
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.time, self.distances, color = self.RDF_color)

        ax.set_xlabel('t (ps)')
        ax.set_ylabel('d ($\AA$)')
        ax.grid()
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(tick)) for tick in xticks])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(tick)) for tick in yticks])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        filepath = os.path.join(self.outdir, f'Dist_{self.filename}.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)