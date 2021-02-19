# How to use Rasenna with a segmfriends + inferno Backbone

This guide describes the use of the Rasenna package together with the segmfriends package and inferno. This is the intended use of this project, as it was designed solely for this purpose. The use of the package is plain simple, as segmfriends is designed such that the loss is outsourced to a separate file.
As the user probably knows, segmfriends can be extensively configured using a .yml configuration file.

```YAML
# --- Modifications to the model to add an additional output branch for boundary prediction using persistent homology --- #
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
      1:
        depth: 0
        out_channels: 1
# ----------------------------------------------------------------------------------------------------------------------- #
```
Note that if you use my fork of the segmfriends package, these configurations are already provided and it should run out of the box.
