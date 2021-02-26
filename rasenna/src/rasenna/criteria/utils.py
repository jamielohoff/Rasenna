import torch
import numpy as np 
import scipy as scp
import matplotlib.pyplot as plt
import cv2
from speedrun.log_anywhere import log_scalar, log_image, log_persistence_diagram


def draw_arrows_and_persistence_diagram(input, target, topo_grad):
    """
    Function that provides the appropriate logging for the topological gradients and persistence diagram.

    :param input: numpy array of shape NxN
            Slice of the input image/probability map from which e calculate the topological features.
    :param target: numpy array of shape NxN
            Slice of the target image/boundary map from which e calculate the topological features. 
    :param grad: tensor of shape NxN
            Gradients calculated through the use of persistent homology.
    """
    assert len(topo_grad) % 2 == 0
    # white = 1 = boundary
    # black = 0 = hole

    rgb_in_pic = [1 - input, 1 - input, 1 - input]
    log_image('output/prediction', torch.stack(rgb_in_pic, dim=0))
    value = (rgb_in_pic, topo_grad)
    log_persistence_diagram('output/barcodes', value)
    log_image('output/target', torch.stack([1 - target], dim=0))


def prepare_target(target):
    """
    Function that performs some data engineering to prepare the affinity and boundary branch. 
    """
    # Create boundary mask from 0th channel to drop parts outside
    # the boundary from the topology computation
    boundary_mask = torch.where(target[:, 0, :, :, :].cpu() >= 1.0, torch.tensor(1.0), torch.tensor(0.0))

    # calculate contours of the mask to get a border to make the 
    # persitent homology calculation meaningful
    contours = get_contours(boundary_mask)

    # performed the data engineering form Alberto's code and the affinity channel
    # In particular, it replaces 
    # - neurofire.criteria.loss_transforms.RemoveSegmentationFromTarget
    # - segmfriends.transform.volume.ApplyAndRemoveMask: {first_invert_target: True}
    target = target[:, 1:, :, :, :]
    seperating_channel = target.size(1) // 2
    mask = target[:, seperating_channel:]
    target = target[:, :seperating_channel]
    mask.requires_grad = False

    # invert target for proper affinity prediction
    target = 1. - target

    # mask prediction and target with mask
    return target * mask, boundary_mask[0], contours

def get_contours(input):
    """
    Function that uses the cv2-package to calculate the boundaries of a given image/mask.

    :param target: numpy array of shape NxN
            Two-dimensional numpy array that contains the image/mask.
    """
    contours = []
    for i in range(input.shape[1]):
        img = np.zeros((input.shape[2], input.shape[3]))
        contour, hierarchy = cv2.findContours(input[0, i].numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contour, -1, (1, 1, 1), 3)
        contours.append(torch.from_numpy(img))
    return torch.stack(contours, dim=0)


