# Expansion-Microscopy-Analysis-Code
Expansion microscopy analysis code- quantitative and qualitative, 2D and 3D, with distortion mapping, RMSE code analysis and 3D visualisation.

04_RMS code 3D_V3.py -
Main RMSE analysis code. Used for processing pre-aligned pre-/post-ExM image data. Requires: File location, X,Y,Z pixel sizes, expansion factor, minimum points to use to analyse, and the RMSE axis (individual 2D axis/3D/ALL).

06_Plotting_3D.py -
Plots the file output of 04_.

08_Mayavi_3D_ExM_visualiser.py -
3D visualisation code. Used for processing pre-aligned pre-/post-ExM image data into Mayavi rendered 3D space. Requires: File location, X,Y,Z pixel sizes, expansion factor.

Additional Data and Files:
2D data and example:
Geometry-preserving expansion microscopy microplates enable high-fidelity nanoscale distortion mapping - Seehra et al. 2023
DOI: 10.1016/j.xcrp.2023.101719

3D data and worked example - supplementary materials:
Mapping EGFR1 sorting domains in endosomes with a calibrated 3D expansion microscopy toolkit - Shakespeare et al. 2025
DOI: 10.1101/2025.10.01.678490v1
