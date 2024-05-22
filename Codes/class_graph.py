import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from configparser import ConfigParser

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
    graph_aesthetic(filname):
        Set aesthetic parameters for Matplotlib plots based on a configuration file ('filename').
    extracting_values(iteration_obj):
        Extract and store relevant values from the iteration object.
    plot_energy():
        Plot the graph of kinetic energy and potential energy in function of time.
    plot_forces(iteration_obj):
        Plot the forces acting on the system in function of time.
    plot_velocity(iteration_obj):
        Plot the velocity in function of time.
    plot_temperature(iteration_obj):
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
        N_bin : list
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
        This class is designed to represent a certain type of object, providing various attributes for 
        configuration and calculations.
        During initialization, several parameters are assigned. Additionally, the 'type' list is filled 
        with instances of RDF class.
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
        self.distances = []
        self.F = np.array([], dtype=float).reshape(0, 3)

        self.time = []

        self.all_energy = False
        self.RDF_color = 'black'
        self.energy_color = ['red', 'blue', 'black']
        self.group_color = ['red', 'blue', 'green', 'yellow', 'black', 'purple']

    def graph_aesthetic(self, filename: str) -> None:
        '''
        Set aesthetic parameters for Matplotlib plots based on a configuration file.
        This method reads parameters from a configuration file using the input filename and updates Matplotlib's 
        default parameters accordingly. The configuration file should include specifications for graph parameters 
        such as label sizes, tick sizes, legend font size, and colors.

        Parameters
        ----------
        filename : str
            Name of the configuration file.

        Notes:
        ------
        This method reads the configuration file and extracts information regarding graph aesthetics, including 
        label sizes, tick sizes, legend font size, and colors. It then updates Matplotlib's default parameters to 
        reflect the specified aesthetics.
        The configuration file should contain lines specifying the following:
        - GRAPH VALUES: Section for graph parameter values.
            - axes.labelsize: Label size for axes.
            - xtick.labelsize: Label size for x-axis ticks.
            - ytick.labelsize: Label size for y-axis ticks.
            - legend.fontsize: Font size for legend.
        - COLORS: Section for color specifications.
            - RDF_color: Color for RDF plot.
            - Energy_sum: Flag indicating whether to plot total energy.
            - K_energy_color: Color for kinetic energy plot.
            - U_energy_color: Color for potential energy plot.
            - Tot_energy_color: Color for total energy plot.
            - Group_color: Colors for different groups in force and temperature plots
        '''

        param = [16, 14, 14, 14]  # Default values for label sizes and legend font size
        config = ConfigParser()

        if not filename:
            config.read('Codes/Setup_graph.ini')
        else:
            if os.path.exists(filename):
                # Return the path of the specified file
                config.read(filename)
            else:
                # Print an error message and exit if the file doesn't exist
                print("File not found")
                exit(0)

        # Setup graph values
        try:
            param[0] = config.getint('GRAPH VALUES', 'axes.labelsize')
        except:
            pass

        try:
            param[1] = config.getint('GRAPH VALUES', 'xtick.labelsize')
        except:
            pass

        try:
            param[2] = config.getint('GRAPH VALUES', 'ytick.labelsize')
        except:
            pass

        try:
            param[3] = config.getint('GRAPH VALUES', 'legend.fontsize')
        except:
            pass

        # Define colors
        try:
            for rdf in self.type:
                rdf.RDF_color = config['COLORS']['RDF_color']
        except:
            pass

        try:
            self.all_energy = config.getboolean('COLORS', 'Energy_sum')

            try:
                self.energy_color[2] = config['COLORS']['Tot_energy_color']
            except:
                pass
        except:
            pass

        try:
            self.energy_color[0] = config['COLORS']['K_energy_color']
        except:
            pass

        try:
            self.energy_color[1] = config['COLORS']['U_energy_color']
        except:
            pass

        try:
            self.group_color = config['COLORS']['Group_color'].split()
        except:
            pass

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
        total force (`F`), distances between the groups defined in the setup (`distances`) and simulation time 
        (`time`) from the provided `step_obj` for each group in the system.
        """

        F = np.array([], dtype=float).reshape(0, 3)
        self.Ek = np.append(self.Ek, 0)

        # Loop through each group in the step_obj
        for gr in step_obj.groups:
            # Append the total force (Ftot) for the current group to the F array
            F = np.append(F, gr.Ftot.reshape(1,3), axis=0)
        
            # Increment the kinetic energy (Ek) for this iteration by the kinetic energy of the current group
            self.Ek[-1] += gr.Ek
                    
        # Append the potential energy (Up) for this iteration to the Up array
        self.Up = np.append(self.Up, step_obj.U_pot)
    
        # Append the distance between the specified groups for this iteration to the distances array
        self.distances = np.append(self.distances, step_obj.dist)
        
        # Append the total force (F) for this iteration to the F array
        self.F = np.append(self.F, np.sum(F, axis=0).reshape(1, 3), axis=0)
        
        # Append the simulation time (time) for this iteration to the time array
        self.time = np.append(self.time, step_obj.dt * step_obj.N_iteration)
        
    def plot_energy(self) -> None:
        """
        Plot the graph of kinetic energy and potential energy in function of time.
        Generates a plot of the kinetic energy and potential energy as a function of time and saves it as an image 
        file named 'E_{filename}.pdf'.

        Notes
        -----
        This method uses Matplotlib to create a plot of the kinetic energy and potential energy over time. The 
        plot is saved as an image file in PDF format.
        """

        # Create a figure with a specified size
        fig = plt.figure(figsize=(10, 6.18033988769))
        # Add a subplot to the figure
        axE = fig.add_subplot(1, 1, 1)
        # Create a twin axes sharing the x-axis with axE
        axU = axE.twinx()
        
        # Plot potential energy against time on axU
        axU.plot(self.time[1:], self.Up[1:] * 0.001, color=self.energy_color[1], label='Potential energy')
        # Plot kinetic energy against time on axE
        axE.plot(self.time[1:], self.Ek[1:], color=self.energy_color[0], label='Kinetic energy')
        # Plot total energy if specified
        if self.all_energy:
            axU.plot(self.time[1:], (self.Up[1:] + self.Ek[1:]) * 0.001, color=self.energy_color[2], label='Total energy')
        
        # Set labels and legends for axE
        axE.set_xlabel('t (ps)')
        axE.set_ylabel('E$_k$ (eV)')
        axE.legend(loc='upper left')
        xticks = axE.get_xticks()
        axE.set_xticks(xticks)
        axE.set_xticklabels([str(int(tick)) for tick in xticks])
        axE.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axE.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        yticks = axE.get_yticks()
        axE.set_yticks(yticks)
        axE.set_yticklabels([str(int(tick)) for tick in yticks])
        axE.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Set labels and legends for axU
        axU.set_ylabel('U (keV)')
        axU.legend(loc='lower right')
        yticks = axU.get_yticks()
        axU.set_yticks(yticks)
        axU.set_yticklabels([str(int(tick)) for tick in yticks])
        axU.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        # Save the plot as a PDF file
        filepath = os.path.join(self.outdir, f'E_{self.filename}.pdf')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)

    def plot_forces(self, step_obj) -> None:
        """
        Plot the forces acting on the system over time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the forces (components along x, y, and z) acting on the system over time 
        and saves it as an image file named 'F_{filename}.pdf'.
        """

        # Create a figure with a specified size
        fig = plt.figure(figsize=(10, 3*6.18033988769 + 2))

        # Add subplots for forces along x, y, and z directions
        axX = fig.add_subplot(3, 1, 1)
        axY = fig.add_subplot(3, 1, 2)
        axZ = fig.add_subplot(3, 1, 3)

        # Plot forces for each group separately with transparency
        for i, gr in enumerate(step_obj.groups):
            # Plot forces for the current group along x, y, and z directions with transparency
            axX.plot(self.time, gr.Ftot_store[:, 0] * 0.001, color=self.group_color[i+1], alpha=0.5, label=f'{gr.id_group}$_x$')
            axY.plot(self.time, gr.Ftot_store[:, 1] * 0.001, color=self.group_color[i+1], alpha=0.5, label=f'{gr.id_group}$_y$')
            axZ.plot(self.time, gr.Ftot_store[:, 2] * 0.001, color=self.group_color[i+1], alpha=0.5, label=f'{gr.id_group}$_z$')

        # Plot total forces along x, y, and z directions
        axX.plot(self.time, self.F[:, 0] * 0.001, color=self.group_color[0], label='F$^{tot}_x$')
        axY.plot(self.time, self.F[:, 1] * 0.001, color=self.group_color[0], label='F$^{tot}_y$')
        axZ.plot(self.time, self.F[:, 2] * 0.001, color=self.group_color[0], label='F$^{tot}_z$')

        # Set common x-axis label and grid for all subplots
        for ax in [axX, axY, axZ]:
            ax.set_xlabel('t (ps)')
            ax.grid()
            ax.legend()
            # Set x-axis ticks and their format
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(tick)) for tick in xticks])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # Set the limits and format for y-axis ticks
            ax.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(tick)) for tick in yticks])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Set y-axis labels for each subplot
        axX.set_ylabel('F$_x$ (nN)')
        axY.set_ylabel('F$_y$ (nN)')
        axZ.set_ylabel('F$_z$ (nN)')

        # Save the plot as a PDF file
        filepath = os.path.join(self.outdir, f'F_{self.filename}.pdf')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)

    def plot_velocity(self, step_obj) -> None:
        """
        Plot the bulk velocity of the groups over time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the bulk velocity (components along x, y, and z) of the groups over time 
        and saves it as an image file named 'V_{filename}.pdf'.
        """

        # Create a figure with a specified size
        fig = plt.figure(figsize=(10, 3*6.18033988769 + 2))
        
        # Add subplots for velocities along x, y, and z directions
        axX = fig.add_subplot(3, 1, 1)
        axY = fig.add_subplot(3, 1, 2)
        axZ = fig.add_subplot(3, 1, 3)

        # Plot velocities for each group separately with transparency
        for i, gr in enumerate(step_obj.groups):
            axX.plot(self.time[1:], gr.Vtot_store[:, 0], color=self.group_color[i+1], alpha=0.5, label=f'{gr.id_group}$_x$')
            axY.plot(self.time[1:], gr.Vtot_store[:, 1], color=self.group_color[i+1], alpha=0.5, label=f'{gr.id_group}$_y$')
            axZ.plot(self.time[1:], gr.Vtot_store[:, 2], color=self.group_color[i+1], alpha=0.5, label=f'{gr.id_group}$_z$')

        # Set common x-axis label and grid for all subplots
        for ax in [axX, axY, axZ]:
            ax.set_xlabel('t (ps)')
            ax.grid()
            ax.legend()
            # Set x-axis ticks and their format
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(tick)) for tick in xticks])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # Set the limits and format for y-axis ticks
            ax.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(tick)) for tick in yticks])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Set y-axis labels for each subplot
        axX.set_ylabel(r'V$_x$ ($\AA$/ps)')
        axY.set_ylabel(r'V$_y$ ($\AA$/ps)')
        axZ.set_ylabel(r'V$_z$ ($\AA$/ps)')

        # Save the plot as a PDF file
        filepath = os.path.join(self.outdir, f'V_{self.filename}.pdf')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)

    def plot_temperature(self, step_obj) -> None:
        """
        Plot the temperature of the system over time.

        Parameters
        ----------
        step_obj : MDstep
            An instance of the `MDstep` class containing simulation data.

        Notes
        -----
        This method generates a plot of the temperature over time for the system and saves it as an image file named 
        'T_{filename}.pdf'.
        """

        # Create a figure with a specified size
        fig = plt.figure(figsize=(10, 6.18033988769))
        
        # Add a subplot for temperature
        ax = fig.add_subplot(1, 1, 1)
                
        # Plot temperature for each group separately with transparency
        for i, gr in enumerate(step_obj.groups):
            ax.plot(self.time[1:], gr.T[1:], color=self.group_color[i+1], alpha=0.5, label=f'G. {gr.id_group}')
        
        # Plot total temperature over time
        ax.plot(self.time[1:], self.T[1:], color=self.group_color[0], label='T$^{tot}$')

        # Set labels, legend, and grid for the plot
        ax.set_xlabel('t (ps)')
        ax.set_ylabel('T (K)')
        ax.grid()
        ax.legend(loc='upper left')
        
        # Set x-axis ticks and format
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(tick)) for tick in xticks])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        
        # Set y-axis ticks and format
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(tick)) for tick in yticks])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # Save the plot as a PDF file
        filepath = os.path.join(self.outdir, f'T_{self.filename}.pdf')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)

    def plot_distance(self) -> None:
        """
        Plot the distance between the two defined groups over time.

        Notes
        -----
        This method generates a plot of the distance between the two defined groups over time and saves it as an 
        image file named 'Dist_{filename}.pdf'.
        """

        # Create a figure with a specified size
        fig = plt.figure(figsize=(10, 6.18033988769))
        
        # Add a subplot for distance
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot distance over time
        ax.plot(self.time, self.distances, color=self.RDF_color)

        # Set labels, grid, and formatting for the plot
        ax.set_xlabel('t (ps)')
        ax.set_ylabel(r'd ($\AA$)')
        ax.grid()
        
        # Set x-axis ticks and format
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(tick)) for tick in xticks])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.set_xlim(min(self.time)-0.04, max(self.time)+0.04)
        
        # Set y-axis ticks and format
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(tick)) for tick in yticks])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # Save the plot as a PDF file
        filepath = os.path.join(self.outdir, f'Dist_{self.filename}.pdf')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
