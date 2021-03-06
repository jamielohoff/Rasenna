# How to modify the cremi dataset for usage with the Rasenna package

To modify the files, use the script like ```python CalculateBoundary.py <path/to/sampleA> <path/to/sampleB> <path/to/sampleC>``` . 
The new files will be created in the same directory and will have the same name with the ending ```<filename>_with_boundaries.h5``` .
You can find the boundaries under the dataframe key ```boundary_maps``` . To use the files, just add the paths in the configuration file of the segmfriends package.

