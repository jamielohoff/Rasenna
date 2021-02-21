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

### Simple Training
After the setup is complete, we want to run an experiment. Assuming that all paths in the .yml config have been configured properly, such that the dataloaders find the HDF5 cremi files, we can start training by switching to the ```segmfriends``` base directory and use:
```
CUDA_VISIBLE_DEVICES=<device ID> python experiments/cremi/train_affinities.py <your experiment name> --inherit <config name>.yml
```
Weights, checkpoints, logging etc. are saved to ```segmfriends/experiments/cremi/runs/<your experiment name>```. The logging can be accessed via tensorboard.
If our fork of segmfriends is used, the following configuration files are provided, which should work out of the box:
| Name | What it does|
|-----|---------------------------------------------------|
|SD_train_affs_multiout.yml| The basic configuration file to make everything work |
|SD_train_affs_2D.yml|A configuration which only uses affinities in the  xy-plane, such that slices cannot exchange information. |

### Checkpointing

To resume/start a training from a given checkpoint, use
```
CUDA_VISIBLE_DEVICES=<device ID> python experiments/cremi/train_affinities.py <your new experiment name> 
--inherit <your new config name>.yml 
--config.model.model_kwargs.loadfrom <path/to/pytorch checkpoint file>
```
One could also use configflags instead of an entirely new configuration file if only one or two parameters change, e.g. 
```
CUDA_VISIBLE_DEVICES=<device ID> python experiments/cremi/train_affinities.py <your new experiment name> 
--inherit <config name>.yml 
--config.model.model_kwargs.loadfrom <path/to/pytorch checkpoint file> 
--config.model.criterion.kwargs.topo_weight 0.5
```
would change the weight of the topological loss to 0.5.

### Pretraining
To pretrain a network using only the affinities channel and Sorensen-Dice and no topological loss, set ```"pretraining": True``` in the criterion keyword arguments.
When the pretraining is complete, you can use the pretrained model by loading it from a checkpoint.

## Inferencing

If your model has converged, you can run the inferencing algorthm by executing
```
CUDA_VISIBLE_DEVICES=<device ID> python experiments/cremi/infer.py <name of your inferencing run> 
--inherit <path/to/your/pytorch/checkpoint> 
--update0 <your inference config name>.yml 
--config.inference.index_output 1 
--config.inference.threshold 0.5
```
in the ```segmfriends``` parent directory. This code predicts a given dataset using your trained network and outputs a score using connected components.
When you use our segmfriends package, there is a file called ``` infer_config.yml ``` which is used to configure the inference. Most of the parameters are self-explanatory or explained in the comments.

For more information on inference, the [cremi_python](https://github.com/elmo0082/cremi_python) package provides some Jupyter notebooks and examples on how to do inferencing using the provided tools.
