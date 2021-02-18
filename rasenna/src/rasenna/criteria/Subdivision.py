import numpy as np
import torch.nn as nn
import torch
import time
import math
from torch.autograd import Function
from .PersistenceUtils import compute_persistence_2DImg, compute_dgm_force


# --------------------------------------------------- #
# Code has never been used or refactored!
# --------------------------------------------------- #

class SubdivisionLossFunction(Function):
    """
    This loss-function calculates the topological loss of a gray-scale image/input in 2.5 dimensions, 
    i.e. 1d homology on 3d voxel slices in z-direction. 
    It is based on the code of Xiaoling Hu's paper --Topology-Preserving Deep Image Segmentation--
    You can find the respective code under https://github.com/HuXiaoling/TopoLoss.
    """

    @staticmethod
    def forward(ctx, input, target):
        """
        Forward pass of topological loss function used to calculate the topological loss between input and target.
        The loss is computed separately for each slice in z-direction and then added up. 
        Not only does the algorithm calculate the loss, but it does also calculate the critical points, 
        i.e. the points that --probably-- need to be fixed to get the right number of holes and thus reduce the loss.
        """
        ctx.x_size = input.shape[1]
        ctx.y_size = input.shape[2]

        loss = 0.0
        dgm_list = []

        for i in range(0, len(input)):
            input_tiles = subdivide(input.cpu().detach()[i], (2,2))
            target_tiles = subdivide(target.cpu().detach()[i], (2,2))
            tile_list = []
            for input_tile, target_tile in zip(input_tiles, target_tiles):
                print(input_tile.shape, target_tile.shape)
                _loss, dgms = compute_loss(input_tile, target_tile)
                loss += _loss 
                tile_list.append(dgms)
            dgm_list.append(tile_list)
        ctx.dgm_list = dgm_list
        
        return torch.tensor(loss)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of topological loss used to calculate the gradient of the topological loss function.
        To calculate the gradient, we need the critical points and their position in the 2d image.
        We again calculate each gradient separately for each slice and then merge them into one 3d voxel, which is then used for backpropagation.
        """ 
        dgm_list = ctx.dgm_list

        grad_list = []

        for tile_list in dgm_list:
            tiles = []
            for list_slice in tile_list:
                tiles.append(compute_gradient(list_slice, ctx.x_size, ctx.y_size))
            grad_list.append(torch.from_numpy(assemble(tiles, (ctx.x_size, ctx.y_size))))
        
        # Stack all gradients along z-direction
        return torch.stack(grad_list, dim=0).cuda(), None


def subdivide(slc, shape):
    '''
    Subdivides the input slice slc into the given shape and returns a list of the tiles.
    '''
    #assert (tile_shape.shape = (1,1)), 'Please provide length and width of the tile!'
    
    x_length = int(slc.shape[0] / shape[0])
    y_length = int(slc.shape[1] / shape[1])

    tile_list = []

    for x_tile in range(0, shape[0]):
        for y_tile in range(0, shape[1]):
            tile = slc[ x_tile*x_length:(x_tile + 1)*x_length, y_tile*y_length:(y_tile + 1)*y_length ]
            tile_list.append(tile)

    return tile_list


def assemble(tile_list, shape):
    '''
    Assembles the tile_list to a matrix of the given shape.
    '''
    output = np.zeros(shape)
    tile_shape = tile_list[0].shape

    x_tiles = int(shape[0] / tile_shape[0])
    y_tiles = int(shape[1] / tile_shape[1])

    for j in range(0, y_tiles):
        for i in range(0, x_tiles):
            output[ i*tile_shape[0]:(i + 1)*tile_shape[0], j*tile_shape[1]:(j + 1)*tile_shape[1] ] = tile_list[i + j]

    return output



def compute_loss(input, target):
    '''
    TODO
    '''
    loss = 0.0

    for i in range(0, len(input)):
        input_dgms, input_birth_cp, input_death_cp = compute_persistence_2DImg(input, dimension=1)
        target_dgms, target_birth_cp, target_death_cp = compute_persistence_2DImg(target, dimension=1)

        force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(input_dgms, target_dgms)
        for idx in idx_holes_to_fix:
            loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
        for idx in idx_holes_to_remove:
            loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2

    # save critical points and persistence diagrams for back-propagation
    return torch.tensor(loss), (input_dgms, input_birth_cp, input_death_cp, target_dgms)


def compute_gradient(list_slice, x_size, y_size):
    '''
    Computes the gradient of a given tile
    '''
    lh_dgm, lh_bcp, lh_dcp, gt_dgm = list_slice

    force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(lh_dgm, gt_dgm)

    # each birth/death crit pt of a persistence dot to move corresponds to a row
    # each row has 3 values: x, y coordinates, and the force (increase/decrease)
    topo_grad = np.zeros([2 * (len(idx_holes_to_fix) + len(idx_holes_to_remove)), 3])
    counter = 0
    for idx in idx_holes_to_fix:
        topo_grad[counter] = [lh_bcp[idx, 1], lh_bcp[idx, 0], force_list[idx, 0]]
        counter = counter + 1
        topo_grad[counter] = [lh_dcp[idx, 1], lh_dcp[idx, 0], force_list[idx, 1]]
        counter = counter + 1
    for idx in idx_holes_to_remove:
        topo_grad[counter] = [lh_bcp[idx, 1], lh_bcp[idx, 0], force_list[idx, 0]]
        counter = counter + 1
        topo_grad[counter] = [lh_dcp[idx, 1], lh_dcp[idx, 0], force_list[idx, 1]]
        counter = counter + 1

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

    gradients = np.zeros((x_size, y_size))

    for pos in topo_grad:
        gradients[int(pos[0]), int(pos[1])] = pos[2]
    
    return gradients
    



