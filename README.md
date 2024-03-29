# Molecular Dynamics Analysis

Molecular Dynamics Analysis is a Python tool designed to assist in the analysis of output files from a molecular dynamics run in Quantum ESPRESSO. It calculates various properties of the entire system and the groups into which it is divided, including temperature, degree of freedom, kinetic energy, potential energy, forces acting on the atoms, and velocities.

## Dependencies

Install the required Python packages using the following command:

```bash
pip install numpy matplotlib
```

If you have not already installed other packages such as `os`, `shutil`, `sys`, and `ConfigParser`, it's not necessary to install them separately as they are standard Python libraries and should already be available in your Python environment. If they are not present, install them using the following command:

```bash
pip install os shutil sys ConfigParser
```

## Usage

This code is used to obtain an initial approximate analysis of a molecular dynamics run performed with Quantum Espresso. In addition to providing various system information, it generates an 'xyz' file to visualize the atomic arrangement at each timestep and graphs.

To run the code, execute the following command in the terminal:

```bash
python pwo_into_xyz.py Setup.txt
```

Where 'Setup.txt' is the setup input file.


### Setup file: 
Place the setup file in the same directory as 'pwo_into_xyz.py'. Inside the 'Setup.txt' file, specify the following entries (the order of the entries doesn't matter):

-   **[SETUP]** 

    Inside this section, you have to define the file to analyze and the directory in which the output are saved:

    -   **Filename:** Specify the name of the 'pwo' file for analysis without the file extension and the path ('_filename'). In the 'Setup.txt' file, write the name of the 'pwo' file without the extension, as follows:

    ```
    Filename = path\_filename
    ```

    -   **Output directory:** Specify the directory for output files ('_dirname'). If the folder already exists, it adds the output files there; if it doesn't exist, it creates a new one with that name. In the 'Setup.txt' file, write the name of the directory as follows:

    ```
    Outdir = _dirname
    ```

-   **[GROUPS]** 

    Here the groups of atoms are defined. To define a group write the line in the following way:
    ```
    _id = _at1 _at2 _at3
    ```
    where '_id' represents the number of the group and '_atN' are the atomic species. The atomic species are composed of letters and numbers (e.g. 'Fe3') to be consistent with the typical Quantum Espresso notation.

-   **[INTERFACE SEPARATION]** 

    Specify the pair of groups for which the distances along the z-coordinate are calculated and plotted ('_id1', '_id2'). In this case, we have to specify the ID numbers of the groups. In the setup file, write the pairs as follows:

    ```
    Groups = _id1 _id2
    ```

-   **[RDF]** 

    Specify the pairs of atoms for which the radial distribution function will be calculated and plotted ('_at1', '_at2'). In this case, we can specifically select a single type of atom group (e.g., 'Fe3') or all the atoms of the same species (e.g., 'Fe1' and 'Fe2' are both taken into account by setting 'Fe'). Additionally, we need to select the maximum distance at which we calculate the RDF ('_Rmax') and the number of bins in the plot histogram ('_Nbin'). Multiple RDFs can be calculated in the same run. In the 'Setup.txt' file, write the pairs as follows:

    ```
    Particles1 = 1_at1 1_at2 1_Rmax 1_Nbin
    Particles2 = 2_at1 2_at2 2_Rmax 2_Nbin
    ...
    ParticlesN = N_at1 N_at2 N_Rmax N_Nbin
    ```

-   **[GRAPHS]**

    Here is specified the input file to decide aesthetic parameters for the graphs. In the 'Setup.txt' file write:
    ```
    filename = aesthetic_filename.txt
    ```
    If this voice is not written the default value is considered (please read the next section).


#### Example
```
[SETUP]
Filename = path\_filename
Outdir = _dirname

[GROUPS]
_id1 = _at1
_id2 = _at2 _at3

[INTERFACE SEPARATION]
Groups = _id1 _id2

[RDF]
couple1 = _at1 _at2  _Rmax _Nbin

[GRAPHS]
filename = aesthetic_filename.txt
```

### Setup graph file:
If you want to change some graphic settings instead of using the default values, you have to use a file placed in the same directory as 'pwo_into_xyz.py'. Inside the setup graph file specify the following entries if you want, otherwise default values are taken (the order of the entries doesn't matter):

-   **[GRAPH VALUES]:** inside this section specify the size of the labels of the axis ('_axsize'), ticks ('_xticksize', '_yticksize'), and legends ('_legendsize'). In the 'Setup_graph.txt' file, write the values as follows:

    ```python
    axes.labelsize = _axsize
    xtick.labelsize = _xticksize
    ytick.labelsize = _yticksize
    legend.fontsize = _legendsize
    ```

    Default values are:

    ```python
    _axsize = 16
    _xticksize = 14
    _yticksize = 14
    _legendsize = 14
    ```

-   **[COLORS]:** in this section specify the colors of the graphs. In particular, we can define the color of RDF plots ('_RDFcolor'), Kinetic energy ('_kineticcolor'), potential energy ('_potentialcolor'), and the color of the groups ('_gr1color', '_gr2color' ...). In the 'Setup_graph.txt' file, write the values as follows:

    ```python
    RDF_color = _RDFcolor
    K_energy_color = _kineticcolor
    U_energy_color = _potentialcolor
    Group_color = _gr1color _gr2color _gr3color _gr4color _gr5color _gr6color
    ```

    If nothing is specified, default values are considered:

    ```python
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

    Inside this section, you can also specify if the total energy of the system is plotted in the energy graph ('_totbool') and the color of the plot ('_totcolor'). In the 'Setup_graph.txt' file, write the values as follows:

    ```python
    Energy_sum = _totbool
    Tot_energy_color = _totcolor
    ```

    The default values are:

    ```python
    _totbool == False
    _totcolor == 'black'
    ```

#### Example
```
[GRAPH VALUES]
axes.labelsize = 16
xtick.labelsize = 14
ytick.labelsize = 14
legend.fontsize = 14

[COLORS]
RDF_color = 'black'
Energy_sum = False
K_energy_color = 'red'
U_energy_color = 'blue'
Tot_energy_color = 'black'
Group_color = 'red' 'blue' 'green' 'yellow' 'black' 'purple'
```

## Returns
The outputs of the run are placed inside the output directory ('_dirname') and are:

-   **Output file:** This file is the new file converted into the 'xyz' format. The file stroe each timestep the the molecular dynamics simulation and in particular each timestep is represented by a number of line equal to the total numeber of atoms plus two: in the first line is printed the total number of atoms in the MD simulation, the second one is usually not red from the probrams which read xyz files, and here the information are sotred, than each line represents an atom.

    The first two lines are in this form:

    ```
    _Natoms
    Lattice(Ang)="_ax, _ay, _az, _bx, _by, _bz, _cx, _cy, _cz" dt(ps)=_dt N=_Niteration Epot(eV)=_U Ek_ngr(eV)=_Kn  DOF_ngr=_DOFn T_ngr(K)=_Tn Ftot_ngr(pN)="_Fxn, _Fyn, _Fzn"
    ```

    In these two lines, there are some general parameters:

    -   '_Natoms': the number of atoms inside the system

    -   lattice vectors:
    ```
    a = (ax, ay, az)
    b = (bx, by, bz)
    c = (cx, cy, cz)
    ```

    -   '_dt': the time interval between two consecutive timesteps

    -   '_Niteration': the number of the current iteration, starting from 1
        
    -   '_U': the potential energy of the system in the current timestep

    Additionally, there are parameters for each group. The number of the group involved is given by '_ngr', and the parameters are:

    -   '_Kn': the kinetic energy of the group

    -   '_DOFn': the degree of freedom of the group

    -   '_Tn': the thermal temperature of the group, removing the velocity of the center of mass

    -   the total forces acting on the atoms of the group:
        ```
        Ftot_n = (_Fxn, _Fyn, _Fzn)
        ```

    Following these lines, the atom's information are in the following pattern:

    ```
    _at _x _y _z _vx _vy _vz _fx _fy _fz _ngr
    ```
    where:

    -   '_at' represents the atomic species of the considered atom

    -   the position in the cell is:
    ```
    R = (_x, _y, _z)
    ```

    -   the velocity is:
    ```
    V = (_vx, _vy, _vz)
    ```

    -   the force acting on the atom is:
    ```
    F = (_fx, _fy, _fz)
    ```

    -'_ngr' represents the number of the group at which the atom belongs

-   **Graphs:** Several graphs are then generated, each of them is named with a preamble that identifies the topic of the graph followed by the filename of the 'pwo' file ('_filename'). The graphs are:

    -   Energy plot: named as 'E_filename.pdf'. It represents the plot of the energy on time during the simulation. Here are plotted the total kinetic energy of the system and the potential energy. Each of them has its own scale (left for the kinetic energy and right for potential energy). In addition, in the 'Setup_graph.txt' file is possible to switch on the plot of the total energy.

    -   Temperature plot: named as 'T_filename.pdf'. It represents the plot of temperature of the whole system and of each group on time during the simulation.

    -   Force plot: named as 'F_filename.pdf'. It represents the plot of the total force acting on the system and on each group depending on time during the simulation. In particular, we have three graphs, one for each coordinate of the force (Fx, Fy, Fz).

    -   Velocity plot: named as 'V_filename.pdf'. It represents the plot of the total velocity of the system and of each group depending on time during the simulation. In particular, we have three graphs, one for each coordinate of the velocity (vx, vy, vz).

    -   Radial distribution function plot: named as '_at1_at2_filename.pdf'. It represents the radial distribution function of the atoms '_at1' and '_at2'.

    -   Distances plot: named as 'Dist_filename.pdf'. It represents the distance between the mean z-coordinates of the two defined groups depending on time.


## Files

The Program is divided in multiple files:

-   **pwo_into_xyz.py:** This serves as the main file for extracting setup parameters and general information about the system. It defines the program's core to extract information from the system at each timestep.

-   **class_iteration.py:** In this file the class 'iteraton' is defined,  where an object of this class encapsulates all information about the system at a single timestep in molecular dynamics simulation.

-   **class_group.py:** In this file the class 'group' is defined, representing a group of atoms used to categorize the entire atoms in a molecular dynamics simulation.

-   **class_atom.py:** In this file the class 'atom' is defined, where an object of this class represents an atomic species utilized in molecular dynamics simulations. It sets up the object with specified attributes, including the atom's name, mass, and the identification number of its group.

-   **class_graph.py:** In this file the class 'graph' is defined.An object of this class stores information about the molecular dynamics simulation to plot the graphs explained earlier.

-   **class_RDF.py:** In this file the class 'RDF' is defined.An object of this class represents the radial distribution function of a single pair of atoms and performs the calculation to build the radial distribution function.

-   **Setup_graph.ini** In this file the delfault values for graphs are setted


## Comments
If there are any problems, please write to me at lorenzo.razzolini@studio.unibo.it. The code is still in development.