import numpy as np
import torch.nn as nn
import torch
import time
import math
import matplotlib.pyplot as plt
from .utils import draw_arrows_and_persistence_diagram

# TODO fix imports 
import imp
imp.load_dynamic('PersistencePython', '/home_sdb/jgrieser_tmp/anaconda3/envs/segmFr/lib/PersistencePython.so')
from PersistencePython import cubePers

def compute_persistence_2DImg(f, dimension, threshold):
    # TODO find out how to adjust the threshold 0.4 seems to be a good value
    """
    Copied from Hu's code...
    compute persistence diagram in a 2D function (can be N-dim) and critical pts
    only generate 1D homology dots and critical points
    """
    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # Pad the function with a few pixels of maximum values
    # This way one can compute the 1D topology as loops
    # Remember to transform back to the original coordinates when finished
    # Black = 1, white = 0
    padwidth = 5
    padvalue = min(f.min(), 0.0)
    f_padded = np.pad(f[3:-3, 3:-3], padwidth, 'constant', constant_values=padvalue)

    # Call persistence code to compute diagrams
    # Loads PersistencePython.so (compiled from C++); should be in current dir
    persistence_result = cubePers(np.reshape(f_padded, f_padded.size).tolist(), list(f_padded.shape), threshold) 

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
    '''
    Copied from Hu's code...
    TODO Write a comment on what this piece of code does.
    '''
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
    # TODO fix documentation
    '''
    This function calculates the loss of of given topological 2-Space 
    '''
    loss = 0.0

    for idx in idx_holes_to_fix:
        loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
    for idx in idx_holes_to_remove:
        loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2

    return loss


def compute_loss_and_gradient(input, target, threshold):
    '''
    This is a function to parallelize the computation of the loss and gradient for each slice.
    '''
    ############################
    #-Computation of the loss--#
    ############################

    input_dgms, input_birth_cp, input_death_cp = compute_persistence_2DImg(input, dimension=1, threshold=threshold)
    target_dgms, target_birth_cp, target_death_cp = compute_persistence_2DImg(target, dimension=1, threshold=threshold)

    if (input_dgms.size != 0) and (target_dgms.size != 0):

        force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(input_dgms, target_dgms)

        loss = compute_loss(force_list, idx_holes_to_fix, idx_holes_to_remove)

        ################################
        #-Computation of the gradient--#
        ################################
        # each birth/death crit pt of a persistence dot to move corresponds to a row
        # each row has 3 values: x, y coordinates, and the force (increase/decrease)
        topo_grad = np.zeros([2 * (len(idx_holes_to_fix) + len(idx_holes_to_remove)), 4])
        counter = 0

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

        # logging
        draw_arrows_and_persistence_diagram(input, target, topo_grad)

        """
        topo_grad contains the coordinates/pixel positions of the critical points 
        as well as the value of the gradient at the respective point in the format
        [x, y, gradient]
        we have to convert this into a format pytorch can use, i.e. we have to 
        create a 2x2 matrix that contains the gradients at the respective positions
        and uses the x,y-positions as indices

        thus we get a [length, width]-matrix with gradients as entries
        """
        topo_grad[:, 2] = topo_grad[:, 2] * 2

        gradients = np.zeros((input.shape[0], input.shape[1]))

        # Populate gradient matrix
        for pos in topo_grad:
            gradients[int(pos[1]), int(pos[0])] = pos[2]
    else:
        loss = 0.0
        gradients = np.zeros((input.shape[0], input.shape[1]))

    return [loss, gradients]

