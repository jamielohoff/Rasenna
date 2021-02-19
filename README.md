# Rasenna
Project that aims to use topological methods for image segmentation/boundary detection on the Cremi dataset. It uses a Unet as backbone with two output branches.
One is used for affinity segmentation and one is used for topological boundary prediction. One could, in principle, also use this package for other applications involving topology.


# Installation

## Overview and preparation of the conda environment

The following packages and their related dependencies are required to install and run the project:
1. https://github.com/elmo0082/segmfriends : My fork of the segmfriends package. This project provides the UNet backbone and dataloaders for the Cremi data.
2. https://github.com/elmo0082/inferno : My fork of the inferno-pytorch package. This project has only minimally modified such that it can interact with my for of speedrun.
3. https://github.com/elmo0082/speedrun : My fork of the speedrun package. I added the ability to log persistence diagrams and draw the corresponding critical points in the prediction image, such that it is much easier to see which topological feature/hole is related to which point in the diagram.
4. https://github.com/elmo0082/pathutils : My fork of the pathutils package. Adjusted such that it runs on every machine and one only has to add ones path. It is required to make the inferencing mode runnable.
5. https://github.com/elmo0082/cremi_python : My fork of the cremi package provided by the cremi project. Has been modified such that inferencing is possible using connected components.
Install all those packages in the given order 

## Installation of Rasenna

The Rasenna package of 

# Usage


# Aknowledgements

