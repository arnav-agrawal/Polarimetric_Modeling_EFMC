This code is used to compute radar scattering from planetary surfaces. The surface modeled is a rocky lunar regolith. The scattering phase function (i.e. probability to scatter in a certain direction) was computed separately and is provided as an input file in the Inputs folder. The code technically models a 2-layer regolith, but I keep the regolith type the same for both layers, so it's really just one layer.

The code works by shooting photons into the medium and scattering them around until they are either absorbed, scatter a max number of times, or reflected out of the medium. At every step we also ask the question: "If this photon were to scatter in the direction of the detector, with the calculated mean free path (mfp calculated at every scattering event), would it exit the medium and be picked up by the detector?" That is, we ask a hypothetical of the direction, not the sampling distance. If the answer is yes, we use that to add the photon record to the database, but let the photon continue scattering within the medium (remember the direction was a hypothetical). The goal is to record the photon statistics (polarization, number of times scattered, global phase, etc.) at the detector.

With this background in mind, the high level overview of the program is as follows. For each photon: 
- Shoot the photon into the medium (we don't model any sort of Fresnel reflection or other surface effects)
- Repeatedly sample a random free path, move the photon by uniformly sampling the scattering phase function, and test for escape or absorption.
- When a valid exiting contribution is detected, reconstruct the forward and time-reversed path Jones operators and store one photon record in the output file.
- Continue tracing until escape, absorption, or MAX_SCATTER is reached.

And this is how to run the program:
- Compile run_electric_field_mc.cu
    - `nvcc run_electric_field_mc.cu -o "YOUR_EXECUTABLE_NAME_HERE"`
- Run the executable that was just created with flags for the total number of photons to Monte Carlo, the number of photons per batch, the random seed for the MC, and the number of photon records allocated in memory per photon
    - `./"YOUR_EXECUTABLE_NAME_HERE" 50000000 1000000 123456789 3`
    - Recommended to use at least 50 million photons (first argument), 500 thousand photons per batch (second argument), and 3-5 photon records per photon (less than 3 will throw an error)
    - This will create a .bin output file which is processed in the next step
- Compile process_photon_packets.c
    - This file is used to process the .bin output file to compute the coherent summation of electric fields from the recorded photons
    - `gcc process_photon_packets.c -lm -o "YOUR_EXECUTABLE_NAME_HERE"`
- Run the executable that was just created, specifying the input path (first argument) and output path (second argument) in case that has changed from what is in the code
    - `./"YOUR_EXECUTABLE_NAME_HERE" photon_db.bin ./Outputs/my_photon_output.dat`
    - This outputs a .dat file which is human readable
- If you want to see the output as a plot (recommended), run the plot_Phase_vs_CPR.m MATLAB file in the Output_Processing folder
    - Just go into that file and change the name of the file in the first line to the name of the .dat file you just created (i.e. '../Outputs/my_photon_output.dat'
    - And change the file name in the very last line (within exportgraphics) to whatever file you want the plot outputted as (i.e. my_plot.jpg)
- Voila! You can compare the plot to Plot.jpg in the Output_Processing folder to see if it looks similar in shape and to ensure everything ran correctly.
