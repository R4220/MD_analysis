# Molecular Dynamics Analysis

Molecular Dynamics Analysis is a Python tool designed to assist in the analysis of output files from a molecular dynamics run in Quantum ESPRESSO. It calculates various properties of the entire system and the groups into which it is divided, including temperature, degree of freedom, kinetic energy, potential energy, forces acting on the atoms, and velocities.

## Usage

This code is used to obtain an initial approximate analysis of a molecular dynamics run performed with Quantum Espresso. In addition to providing various system information, it generates an 'xyz' file to visualize the atomic arrangement at each timestep and graphs.

To run the code print the following lines in the terminal:
    ```bash
    python3.10 pwo_into_xyz.py
    ```

Before running the code, follow these steps:

**Setup.txt:** Place the 'Setup.txt' and the 'pwo' files in the same directory as 'pwo_into_xyz.py'. Inside the 'Setup.txt' file, specify the following entries (the order of the entries doesn't matter):

   i. **Filename:** Specify the name of the 'pwo' file for analysis without the file extension ('_filename'). In the 'Setup.txt' file, write the name of the 'pwo' file without the extension, as follows:
        ```
        Filename: _filename
        ```
    ii. **Output directory:** Specify the directory for output files ('_dirname'). If the folder already exists, it adds the output files there; if it doesn't exist, it creates a new one with that name. In the 'Setup.txt' file, write the name of the directory as follows:
        ```
        Outdir: _dirname
        ```
    iii. **Atomic groups:** Specify the atomic element for each group ('_at1', '_at2') and the group number ('N_group'). The atomic species can be repeated and can be composed by letter and number (e.g., 'Fe3') to be consistent with the typical Quantum Espresso notation. In the 'Setup.txt' file, write the atomic groups as follows:
        ```
        Group _Ngroup
        _at1 _at2
        ```
    iv. **Radial distribution function:** Specify the pairs of atoms for which the radial distribution function will be calculated and plotted ('_at1', '_at2'). In this case, we can specifically select a single type of atom group (e.g., 'Fe3') or all the atoms of the same species (e.g., 'Fe1' and 'Fe2' are both taken into account by setting 'Fe'). Additionally, we need to select the maximum distance at which we calculate the RDF ('_Rmax') and the number of bins in the plot histogram ('_Nbin'). Multiple RDFs can be calculated in the same run. In the 'Setup.txt' file, write the pairs as follows:
        ```
        Particles: _at1 _at2 _Rmax _Nbin
        ```

**Setup_graph.txt:** Place the 'Setup_graph.txt' file in the same directory as 'pwo_into_xyz.py'. Inside the 'Setup_graph.txt' file specify the following entries if you want, otherwise default values are taken (the order of the entries doesn't matter):
    i. **Graph values:** Specify the size of the labels of the axis ('_axsize'), ticks ('_xticksize', '_yticksize'), and legends ('_legendsize'). In the 'Setup_graph.txt' file, write the values as follows:
        ```
        axes.labelsize = _axsize
        xtick.labelsize = _xticksize
        ytick.labelsize = _yticksize
        legend.fontsize = _legendsize
        ```
    Default values are:
        ```
        _axsize = 16
        _xticksize = 14
        _yticksize = 14
        _legendsize = 14
        ```
    ii. **Colors:** Specify the colors of the graphs. In particular, we can define the color of RDF plots ('_RDFcolor'), Kinetic energy ('_kineticcolor'), potential energy ('_potentialcolor'), and the color of the groups ('_gr1color', '_gr2color' ...). In the 'Setup_graph.txt' file, write the values as follows:
        ```
        RDF_color = _RDFcolor
        K_energy_color = _kineticcolor
        U_energy_color = _potentialcolor
        Group_color = _gr1color _gr2color _gr3color _gr4color _gr5color _gr6color
        ```
    If nothing is specified, default values are considered:
        ```
        _RDFcolor == 'black'
        _kineticcolor == 'red'
        _potentialcolor == 'blue'
        _gr1color == 'red'
        _gr2color == 'blue'
        _gr3color == 'green'
        _gr4color == 'yellow'
        _gr5color == 'black'
        _gr6color == 'purple'
        ```
    If the default value is used, the maximum number of groups is six.
    iii. **Total energy:** Specify if the total energy of the system is plotted in the energy graph ('_totbool') and the color of the plot ('_totcolor'). In the 'Setup_graph.txt' file, write the values as follows:
        ```
        Energy_sum = _totbool
        Tot_energy_color = _totcolor
        ```
    The default values are:
        ```
        _totbool == False
        _totcolor == 'black'
        ```

**Returns:** The outputs of the run are placed inside the output directory ('_dirname') and are:

    i. **Output file:** This file is the new file converted into the 'xyz' format. The file stroe each timestep the the molecular dynamics simulation and in particular each timestep is represented by a number of line equal to the total numeber of atoms plus two: in the first line is printed the total number of atoms in the MD simulation, the second one is usually not red from the probrams which read xyz files, and here the information are sotred, than each line represents an atom.
    The first two lines are in this form:
        ```
        _Natoms
        Lattice(Ang)="_ax, _ay, _az, _bx, _by, _bz, _cx, _cy, _cz" dt(ps)=_dt N=_Niteration Epot(eV)=_U Ek_ngr(eV)=_Kn DOF_ngr=_DOFn T_ngr(K)=_Tn Ftot_ngr(pN)="_Fxn, _Fyn, _Fzn"
        ```
    In these two lines, there are some general parameters:
        - '_Natoms': the number of atoms inside the system
        - lattice vectors:
            ```
            a = (ax, ay, az)
            b = (bx, by, bz)
            c = (cx, cy, cz)
            ```
        - '_dt': the time interval between two consecutive timesteps
        - '_Niteration': the number of the current iteration, starting from 1
        - '_U': the potential energy of the system in the current timestep
    Additionally, there are parameters for each group. The number of the group involved is given by '_ngr', and the parameters are:
        - '_Kn': the kinetic energy of the group
        - '_DOFn': the degree of freedom of the group
        - '_Tn': the thermal temperature of the group, removing the velocity of the center of mass
        - the total forces acting on the atoms of the group:
            ```
            Ftot_n = (_Fxn, _Fyn, _Fzn)
            ```
    Following these lines, the atom's information are in the following pattern:
    ```
    _at _x _y _z _vx _vy _vz _fx _fy _fz _ngr
    ```
    where:
        - '_at' represents the atomic species of the considered atom
        - the position in the cell is:
            ```
            R = (_x, _y, _z)
            ```
        - the velocity is:
            ```
            V = (_vx, _vy, _vz)
            ```
        - the force acting on the atom is:
            ```
            F = (_fx, _fy, _fz)
            ```
        - '_ngr' represents the number of the group at which the atom belongs

    ii. **Graphs:** Several graphs are then generated, each of them is named with a preamble that identifies the topic of the graph followed by the filename of the 'pwo' file ('_filename'). The graphs are:

        - Energy plot: named as 'E_filename.png'. It represents the plot of the energy on time during the simulation. Here are plotted the total kinetic energy of the system and the potential energy. Each of them has its own scale (left for the kinetic energy and right for potential energy). In addition, in the 'Setup_graph.txt' file is possible to switch on the plot of the total energy.

        - Temperature plot: named as 'T_filename.png'. It represents the plot of temperature of the whole system and of each group on time during the simulation.

        - Force plot: named as 'F_filename.png'. It represents the plot of the total force acting on the system and on each group depending on time during the simulation. In particular, we have three graphs, one for each coordinate of the force (Fx, Fy, Fz).

        - Velocity plot: named as 'V_filename.png'. It represents the plot of the total velocity of the system and of each group depending on time during the simulation. In particular, we have three graphs, one for each coordinate of the velocity (vx, vy, vz).

        - Radial distribution function plot: named as '_at1_at2_filename.png'. It represents the radial distribution function of the atoms '_at1' and '_at2'.
 

### Dependencies

Install the required Python packages using the following command:

```
bash
pip install numpy os matplotlib shutil
```


#### Files

The Program is divided in multipole files:

    - **'pwo_into_xyz.py':** This serves as the main file for extracting setup parameters and general information about the system. It defines the program's core to extract information from the system at each timestep.

    - **'class_iteration.py':** In this file the class 'iteraton' is defined,  where an object of this class encapsulates all information about the system at a single timestep in molecular dynamics simulation.

    - **'class_group.py':** In this file the class 'group' is defined, representing a group of atoms used to categorize the entire atoms in a molecular dynamics simulation.

    - **'class_atom.py':** In this file the class 'atom' is defined, where an object of this class represents an atomic species utilized in molecular dynamics simulations. It sets up the object with specified attributes, including the atom's name, mass, and the identification number of its group.

    - **'class_graph.py':** In this file the class 'graph' is defined.An object of this class stores information about the molecular dynamics simulation to plot the graphs explained earlier.

    - **'class_RDF.py':** In this file the class 'RDF' is defined.An object of this class represents the radial distribution function of a single pair of atoms and performs the calculation to build the radial distribution function.
# sample-code
