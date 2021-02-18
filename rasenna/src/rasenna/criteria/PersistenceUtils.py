import os
import numpy as np
import torch.nn as nn
import torch
import time
import math
import logging
import matplotlib.pyplot as plt
from .utils import draw_arrows_and_persistence_diagram

# TODO fix imports 
import imp
imp.load_dynamic('PersistencePython', os.path.dirname(__file__) + '/PersistencePython.so')
# '/home_sdb/jgrieser_tmp/anaconda3/envs/segmFr/lib/PersistencePython.so')
from PersistencePython import cubePers

def compute_persistence_2DImg(f, dimension=1.0, threshold=0.4):
    """
    Copied from Hu's code under https://github.com/HuXiaoling/TopoLoss. 
    This function computes the persistence diagram of a 2D probability map and the corresponding critical points,
    given a threshold.

    :param f: numpy array of shape NxN
            This is the input image/probability map from which e calculate the topological features.
    :param dimension: float
            Controls the dimension of the topoological features, that we want to calculate.
    :param threshold: float
            When calculating topological features, we have to say, which pixels belong to the boundary 
            and which pixels belong to the voids/holes. Everything above the threshold is boundary, 
            everything below is considered to belong to the hole.
            We have the following color code: Black = 0 = hole, white = 1 = boundary
    """
    # f has to be 2D function
    assert len(f.shape) == 2  
    dim = 2
    # Pad the function with a few pixels of maximum values
    # This way one can compute the 1D topology as loops
    # Remember to transform back to the original coordinates when finished
    padwidth = 3
    padvalue = min(f.min(), 0.0)
    f_padded = np.pad(f[3:-3, 3:-3], padwidth, 'constant', constant_values=padvalue)
    
    start = time.time()

    # Call persistence code to compute diagrams
    # Loads PersistencePython.so (compiled from C++); should be in current dir
    persistence_result = cubePers(np.reshape(f_padded, f_padded.size).tolist(), list(f_padded.shape), threshold) 

    end = time.time()
    print('cubePers time elapsed ' + str(end - start))

    # Only take 1-dim topology, first column of persistence_result is dimension
    persistence_result_filtered = np.array(list(filter(lambda x: x[0] == float(dimension), persistence_result)))

    # Persistence diagram (second and third columns are coordinates)
    # Check if filtration is not empty
    if len(persistence_result_filtered.shape) < 2:
        dgm = np.array([])
        birth_cp_list = np.array([])
        death_cp_list = np.array([])
    else:
        dgm = persistence_result_filtered[:, 1:3]

        # Critical points
        birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
        death_cp_list = persistence_result_filtered[:, 4 + dim:]

        # When mapping back, shift critical points back to the original coordinates
        birth_cp_list = birth_cp_list - padwidth + 3
        death_cp_list = death_cp_list - padwidth + 3
    return dgm, birth_cp_list, death_cp_list


def compute_dgm_force(lh_dgm, gt_dgm):
    """
    Copied from Hu's code under https://github.com/HuXiaoling/TopoLoss and heavily modified by author.
    This function calculates the diagram "forces", i.e. the whether a critical point has to be lifted or subsided.
    It translates into whether a point in the persistence diagram has to be mapped to (0,1), i.e. the 
    topological feature has to be preserved and become more emanent or is mapped to the diagonal and 
    the corresponding feature has to vanish.
    This is basically nothing more than the calculation of the gradients.
    """
    # get persistence list from both diagrams
    lh_pers = lh_dgm[:, 1] - lh_dgm[:, 0]
    gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]

    # more lh dots than gt dots
    assert lh_pers.size >= gt_pers.size

    # check to ensure that all gt dots have persistence 1
    tmp = gt_pers > 0.999
    # assert tmp.sum() == gt_pers.size

    gt_n_holes = gt_pers.size  # number of holes in gt

    # get "perfect holes" - holes which do not need to be fixed, i.e., find top
    # lh_n_holes_perfect indices
    # check to ensure that at least one dot has persistence 1; it is the hole
    # formed by the padded boundary
    # if no hole is ~1 (ie >.999) then just take all holes with max values
    tmp = lh_pers > 0.999  # old: assert tmp.sum() >= 1
    if tmp.sum() >= 1:
        # n_holes_to_fix = gt_n_holes - lh_n_holes_perfect
        lh_n_holes_perfect = tmp.sum()
        idx_holes_perfect = np.argpartition(lh_pers, -lh_n_holes_perfect)[-lh_n_holes_perfect:]
    else:
        idx_holes_perfect = np.where(lh_pers == lh_pers.max())[0]

    # find top gt_n_holes indices
    idx_holes_to_fix_or_perfect = np.argpartition(lh_pers, -gt_n_holes)[-gt_n_holes:]

    # the difference is holes to be fixed to perfect
    idx_holes_to_fix = list(set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

    # remaining holes are all to be removed
    idx_holes_to_remove = list(set(range(lh_pers.size)) - set(idx_holes_to_fix_or_perfect))

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # TODO values below this are small dents so dont fix them; tune this value?
    pers_thd = 0.05
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / math.sqrt(2.0)

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def compute_loss(force_list, idx_holes_to_fix, idx_holes_to_remove):
    """
    Copied from Hu's code under https://github.com/HuXiaoling/TopoLoss and heavily modified by author.
    This function calculates the loss of of given topological 2-Space using the algorithm presented in Hu's paper.
    To do this, it uses the previously calculated "forces" which quantify the relevance of a topological feature and 
    its associated critical points.
    By relevance we mean whether the critical point belongs to a feature that we want to see in the ground truth 
    or if it is noise or a wrong prediction.
    """
    loss = 0.0

    for idx in idx_holes_to_fix:
        loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
    for idx in idx_holes_to_remove:
        loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2

    return loss

def compute_loss_and_gradient(input, target, threshold, slice_index, loss_dict, gradient_dict):
    """
    Copied from Hu's code under https://github.com/HuXiaoling/TopoLoss and heavily modified by author.
    This is a function to parallelize the computation of the loss and gradient for each slice.
    """
    # computation of the diagrams of prediction (input) and ground truth (target)
    input_dgms, input_birth_cp, input_death_cp = compute_persistence_2DImg(input, dimension=1, threshold=threshold)
    target_dgms, target_birth_cp, target_death_cp = compute_persistence_2DImg(target, dimension=1, threshold=threshold)

    if (input_dgms.size != 0) and (target_dgms.size != 0):
        # compute forces acting on critical points
        force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(input_dgms, target_dgms)

        # compute loss
        loss = compute_loss(force_list, idx_holes_to_fix, idx_holes_to_remove)

        ###############################
        #-Computation of the gradient-#
        ###############################
        
        # each birth/death crit pt of a persistence dot to move corresponds to a row
        # each row has 3 values: x, y coordinates, and the force (increase/decrease)
        topo_grad = np.zeros([2 * (len(idx_holes_to_fix) + len(idx_holes_to_remove)), 4])
        counter = 0

        # calculate gradients according to the algorithm presented by Hu et al.
        for idx in idx_holes_to_fix:
            topo_grad[counter] = [input_birth_cp[idx, 1], input_birth_cp[idx, 0], force_list[idx, 0], input_dgms[idx, 0]]
            counter = counter + 1
            topo_grad[counter] = [input_death_cp[idx, 1], input_death_cp[idx, 0], force_list[idx, 1], input_dgms[idx, 1]]
            counter = counter + 1
        for idx in idx_holes_to_remove:
            topo_grad[counter] = [input_birth_cp[idx, 1], input_birth_cp[idx, 0], force_list[idx, 0], input_dgms[idx, 0]]
            counter = counter + 1
            topo_grad[counter] = [input_death_cp[idx, 1], input_death_cp[idx, 0], force_list[idx, 1], input_dgms[idx, 1]]
            counter = counter + 1

        # logging of  the persistence diagrams and the gradients
        draw_arrows_and_persistence_diagram(input, target, topo_grad)

        """
        Topo_grad contains the coordinates/pixel positions of the critical points 
        as well as the value of the gradient at the respective point in the format
        [x, y, gradient].
        We have to convert this into a format pytorch can use, i.e. we have to 
        create a 2x2 matrix that contains the gradients at the respective positions
        and uses the x,y-positions as indices.

        Thus we get a [length, width]-matrix with gradients as entries.
        """
        topo_grad[:, 2] = topo_grad[:, 2] * 2

        gradients = np.zeros((input.shape[0], input.shape[1]))

        # Populate gradient matrix
        for pos in topo_grad:
            gradients[int(pos[1]), int(pos[0])] = pos[2]
    else:
        loss = 0.0
        gradients = np.zeros((input.shape[0], input.shape[1]))

    """
    In-place modification of loss_dict and gradient_dict.
    This is necessary due to the usage of multiprocessing.
    Only then, the gradients will be in the right order. 
    """
    loss_dict[slice_index] = loss
    gradient_dict[slice_index] = torch.from_numpy(gradients)

