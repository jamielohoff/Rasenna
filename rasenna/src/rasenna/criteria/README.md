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
Note that if you use my fork of the segmfriends package, these configurations are already provided and it should run out of the box.

## Training



## Inference
