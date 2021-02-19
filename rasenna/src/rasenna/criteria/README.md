# How to use Rasenna with a segmfriends + inferno Backbone

## Setup
This guide describes the use of the Rasenna package together with the segmfriends package and inferno. This is the intended use of this project, as it was designed solely for this purpose. The use of the package is plain simple, as segmfriends is designed such that the loss is outsourced to a separate file.
The details of how the code works are layed out in the commens within the code.

As the user probably knows, segmfriends can be extensively configured using a .yml configuration file.
To tell the interpreter and inferno which loss function to use and where it can be found, just change the criterion to:
```YAML
criterion:
  loss_name: "rasenna.criteria.CustomLossFunctions.TopologicalLoss"
  kwargs: {"SD_weight": 1.0, "topo_weight": 1.0, "topo_SD_weight": 1.0, "pretraining": False}
```
The keyword args and hyperparameters are explained in ```CustomLossFunction.py```. If you want your loss to do something else or to add some logging, just modify ```CustomLossFunction.py```.
There is one other thing that has to be done. We have to tell segmfriends to add another output branch to the UNet for the topological loss:

```YAML
model_class: confnets.models.MultiScaleInputMultiOutputUNet
model_kwargs:
  depth: 3
  in_channels: 1
  output_branches_specs: 
    global:
      activation: "Sigmoid"
      out_channels: 20
    0:
      depth: 0
    1: # Additional branch for topological loss
      depth: 0
      out_channels: 1
```
Note that if you use our fork of the segmfriends package, these configurations are already provided and it should run out of the box.

## Training

After the setup is complete, we want to run an experiment. Assuming that all paths in the .yml config have been configured properly, such that the dataloaders find the HDF5 cremi files, we can start training by switching to the ```segmfriends``` base directory and use:
```shell
CUDA_VISIBLE_DEVICES=<device id> python experiments/cremi/train_affinities.py <your experiment name> --inherit <config name>.yml
```
Weights, logging etc. are saved to ```segmfriends/experiments/cremi/runs/<your experiment name>```. The logging can be accessed via tensorboard.
If our fork of segmfriends is used, the following configuration files are provided, which should work out of the box:
| Name | What it does|
|SD_train_affs_multiout.yml| The basic configuration file to make everything work |
|SD_train_affs_2D.yml|A configuration which only uses affinities in the  xy-plane, such that slices cannot exchange information. |

## Inference
