trap-sim-tools-python
=====================
UPDATE: There is now a tutorial available if you would like a more in-depth explanation of how the software works.

---------------
1. Introduction
---------------

This package is used to read and post-process electrostatic simulations of ion trap potentials. 

Before you use this package you must use a numerical electrostatics solver which returns the potentials 
of all the electrodes of a trap. These potentials of all electrodes are stored in a text file with format:

X1,Y1,Z1,V1,Ex1,Ey1,Ez1
...
XN,YN,ZN,VN,Exn,Eyn,Ezn

The final three terms correspond to the electric field values which are infrequently used.
You can find example simulations under eurotrap-pt1.txt etc. These were produced using Kilian Singer's 
BEMsolver package. You can also create synthetic "simulations" directly with the test_text function.

You can view example plots for reference here: https://drive.google.com/folderview?id=0B4X2o4phz6b2eTZTVTRScTFtRWs&usp=sharing
These were created using the default made by test_text.
Left of the period is the simulation number, or "g" for the combined simulations.
Please note if the rest is not self-explanatory.

---------------------------------------
2. Preliminaries and naming conventions
---------------------------------------

We have adapted this library to use with surface electrode traps. You should be able to use it with any 
other trap geometry. Our convention for numbering the trap electrodes is: 
  a. Each trap has N electrodes and one ground electrode (the RF electrode is included to the N)
  b. For a linear segmented trap We start counting on the lower left corner. The lower left DC electrode 
     is 1. As you move up along the axis (staying on the same side of the RF rail you count up). When you 
     reach the top, you cross to the right side of the RF rail and start counting at the bottom. When you 
     reach the top, you cross to the inside of the RF rail and count DC electrodes from bottom to top. When 
     you are done counting all the DCs, you count the RF electrode(s). Below is an example with N=12:
      ___________________
     | 05 | |    | | 10 |
     |____| |    | |____|
     | 04 | |    | | 09 |
     |____| |    | |____|
     | 03 | | 11 | | 08 |
     |____| |    | |____|
     | 02 | |    | | 07 |
     |____| |    | |____|
     | 01 | |    | | 06 |
     |____|_|____|_|____|
     
---------------
3. Instructions 
---------------

Day-to-day usage will only involve messing around with analyze_trap and project_parameters.
You will rarely need to dive into the lower level functions.
_______________________________________________________________________________________________________________________
--------------------------
Main scripts and functions
--------------------------
test_text:
	This is a script that generates text files of data in place of BEM-solver. It is used for debugging puposes.
	As of 5/25/14, it generates 2 simulations, each with 14 electrodes, with a saddle point where the two intersect.
	The default requires no permutation. Only edit this function if you are certain of what you are doing, but this 		is less important for test_text than it is for other non-day-to-day usage functions. It only uses 			multipoles up to the second order, so there will not be an escape position to calculate in trap_depth.

analyze_trap: 
    This is the main script which simply calls other scripts. 
    Unlike the matlab code, all parameter controls for the software are in project_paramters.
    
project_parameters: 
    Here you define all the parameters pertaining to the trap simulation, and the system parameters 
    (paths for reading/writing data). This is all the messing around you will need to do. Any attempts 
    to change parameters in the following functions will probably lead to regret and frustration.
    Unliek the matlab code, it includes the fuctionality of set_voltages and most of analyze_trap.
    You should read through the entire script at least once before simulating and be consistent with your values.
    This script also checks for consistency between parameters and converts the Ei,Ui multipole coefficients and
    unit scaling from the experimentally useful external convention to the internal SI and mathematical conventions. 
    
import_data:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    import_data reads the text files which your BEM solver simulations generated, and converts them to a .pkl structures.
    It *does not* make any changes to the structure 'trap'. This is the job of the next function. 
    The main thing to remember is that consecutive BEM solver simulations must have overlapping 
    first and last axial positions, i.e. [Z[last]]_simulation[n] = [Z([first]]_simulation[n+1], where Z is along the 
    trap axis. Each simulation is a "cube" of data points, each with a number of coordinates along Z equal to the full
    lengths of X and Y each, due to BEM-solver limitation. 
    Be sure that you are using the proper permutation in import_data.
    
get_trapping_field:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    get_trapping_field adds the grid and electrostatic potential arrays to build the structure 'trap'. 
    It looks through the data structures which BEM solver generated and creates new data structure with a grid 
    which is centered around the position where you want to trap. You define the trapping position in project_parameters.
    At the end of get_trapping_field, the structure 'trap' should contain a field "potentials", with subfields such as 
    trap.potentials.X, trap.potentials.EL_DC1, trap.potentials.EL_RF, etc. It will also have a field "configuration" that
    stores other relevant parameters as they are used by these scripts.

expand_field:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    expand_field does the spherical harmonic expansion of the potentials for all the used electrodes. It then stores the 
    harmonic expansion coefficients (excluding that of the RF) in the array trap.configuration.multipoleCoefficients.
    Typically, expand_field will also compute all the potentials from the spherical harmonic coefficients and replace the
    original values with the computed ones. This can be useful as a data smoothing step so that the algorithms which use 
    numerical derivatives (e.g. in post_process_trap for pseudopotential calculation, find_saddle, trap_depth) work 		    better. However, this may also mirror the potentials along all axes, which can be confusing if unexpected.
    It also calls dc_potential to create a field of trap "instance" that is referenced by lower order function.
    
trap_knobs:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    trap_knobs calculates the independent multipole control parameters (see G. Littich Master's thesis for an explanation     of what these are). It stores the control parameters in trap.configuration.multipoleControl. It also stores the          kernel vectors of the multipoleControl space in trap.Configuration.multipoleControl, unless the multipoleControls are
    rank deficient. This is also where the electrodeMapping and designation of manualElectrodes becomes relevant,
    as the pseudoinverse is computed with the redundant electrode multipoleCoefficients removed.
    Normally, you are done at this point, you can save the so-called trap.Configuration.multipoleControl array to a file 
    and import it to LabRad. As of 4/4/2014, the array is saved as Multipole_Control_File.txt, 
    under the _post_processed folder.

post_process_trap:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    post_proccees_trap is a function which takes in a set of electrode voltages, an RF frequency, and an amplitude to
    calculate what the trap should look like for the ion. It plots the rf potential, rf pseudopotential, dc potentail, 
    and total trap potential. Furthermore, it calculates the trap frequencies, trap depth, and trap position from pfit. 
    The options for plotting these potentials are: 'no plots', '1d plots', and '2d plots'.
    It updates the trap.instance structure with all these parameters that it has calculated.
    Two obsolete (pre-2009) functionalities of it are: calculating micromotion compensation parameters given a stray 
    electric field, and calculating a stray electric field given micromotion compensated voltages. 
    You will not use these, unless you have an interest in the history of science and technology.
    
---------------------------------
Lower level scripts and functions
---------------------------------
compact_matrix, expand_matrix_el, and expand_matrix mult:
    You should not have to look inside these function, unless you are trying something new (and possibly inappropriate). 
    compact_matrix uses the elctrodeMapping to combine all electrodes that map to the same generator and then removes
    all manual electrodes that are not also redundant by electrodeMapping. The other two functions undo this.
	
d_e:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    d_e is a relic of the older functionalities of post_process_trap, but it is still used in the modern script at one 		    place. There it calculates the compensation for the given stray field.
	
dc_potential:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    dc_potential weights and sums up all electrodes to create a single "instance" field that describes the trap.
    It is primarily used in expand_field and post_proccess_trap.
	
exact_saddle:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    exact_saddle calculates the exact position of the saddle by minimizing the electric field by calling sum_of_e_field.
    If there is no satisfactory saddle point, it states so and then returns the closest candidate. While find_saddle
    finds the index of the saddle along the grid vectors, exact_saddle returns the value of the grid vectors themselves.
    
find_saddle:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate).
    find_saddle takes the gradient (electric field) of a potential field and minimizes it to find the saddle point.
    It returns the index of grid vector corresponding to the saddle, unlike exact_saddle which returns the values.

mesh_slice:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    mesh_slice plots 2D plots of a given potential along the three axes individually. It plots point by point along each     axis, creating an animation with time acting as the relevant spatial dimension. It can also display a title that         changes between frames, but it moves down the image as it animates. It is only called by plot_potential.

p2d:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    p2d fits 2D matrix data to two coordinate matrices. It returns the fit coefficients and the angle between. 
    It is only used by pfit in post_process_trap.  
    
pfit:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    pfit is a helper function for post_process_trap that calculates the trap frequencies by converting the RF to a 
    pseudopotential. It also calls p2d and trap_depth to return their outputs.
    
plot_potential:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    plot_potential is used in many functions, typically for debugging, in order to visualize 3D potentials through plots.
    It calls meshslice to display plot animations.
    
plotN:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    plotN displays the electrode values (excluding the RF) in the geometry given above. It is used for debugging             trap_knobs.
    
spher_harm_exp, spher_harm_cmp, and spher_harm_qlt:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    spher_harm_exp calculates the sperical harmonic expansion coefficients to fit a given 3D potential, spher_harm_cmp 
    does the inverse, and spher_harm_qlt determines the fidelity of the expansion by comparing the potential before and      after. The first is used by sum_of_e_field for exact_saddle optimazation, while the others are only used by              expand_field and debugging. The multipole naming convention is implemented here, unlike the matlab code which            implements it in expand_field.

set_voltages:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    In the matlab code, this is treated as a higher order function but the python code treats this script as lower order
    due to its limited variety of uses. In matlab, it is named multipole_set_dc and not equivalent to set_voltages there.
    It takes multipole parameters or a set of Mathieu parameters, and produces an 1-D array with the voltages you need 
    to apply to all of the electrodes for the appropriate overall field. Entries corresponding to multipole-controlled
    electrodes receive some particular values. If some electrodes are under manual control (per your definition in 
    project_parameters) then these are set to zero. You can add the values to these electrodes at a higher level 
    (as in project_parameters)
    
sum_of_e_field:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    sum_of_e_field calculats the sum of the first three non-constant terms of the spherical harmonic potential, which 
    correspond to the electric field, and divide this sum by the sum of the following coefficients. Toggling debug for 
    this function displays the process of optimization for exact_saddle.
    
trap_depth:
    You should not have to look inside this function, unless you are trying something new (and possibly inappropriate). 
    It takes the gradient of the potential field, much like the saddle functions, but then takes the smallest value some
    distance away from the true saddle point to find the escape position. Toggling debug allows the user to control this 
    minimum distance manually. The grid resolution may not necessarily be high enough to find the escape position. It is 
    only used by pfit in post_proccess_trap.
    
_______________________________________________________________________________________________________________________
-----------------------------------------------------------------------------------------------------------------------
To Do:

* Change the length scales to micron, but keep mm for the E and U
