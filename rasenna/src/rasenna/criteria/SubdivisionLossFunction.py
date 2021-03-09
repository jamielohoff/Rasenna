import numpy as np
import torch.nn as nn
import multiprocessing as mp
import torch
import time
import math
from torch.autograd import Function
from .PersistenceUtils import loss_and_gradient
from .utils import draw_arrows_and_persistence_diagram


class SubdivisionLossFunction(Function):
    """
    This loss-function calculates the topological loss of a gray-scale image/input in 2.5 dimensions, 
    i.e. 1D homology on 3D voxel slices in z-direction. 
    It is based on the code of Xiaoling Hu's paper --Topology-Preserving Deep Image Segmentation--
    You can find the respective code under https://github.com/HuXiaoling/TopoLoss.

    This function is parallelized in such a way that for a 3D voxel where we calculate the homology along the z-axis slices,
    for each slice, we create another process, leading to a highly parallelized version, where all slices are almost calculated at once.
    ==> Computation of 6 slices requires 6 cores, but only takes the time for 1 slice
    """

    @staticmethod
    def forward(ctx, input, target):
        """
        Forward pass of topological loss function used to calculate the topological loss between input and target.
        The loss is computed separately for each slice in z-direction and then added up. 
        Not only does the algorithm calculate the loss, but it does also calculate the critical points, 
        i.e. the points that --probably-- need to be fixed to get the right number of holes and thus reduce the loss.
        From this, we then can also calculate the gradient/jacobian of the topological loss.
        """
        loss = 0.0
        grad_list = None
        threshold = 0.4
        jobs = []
        grid = (2, 2)

        # Setting up the multiprocessing manager
        manager = mp.Manager()
        loss_dict = manager.dict()
        gradient_dict = manager.dict()

        # Create as many separate processes as there are slices in z-direction for 
        # Maximum parallelization as persistent homology is calculated only in xy-plane
        for i in range(0, len(input)):  
            # We have to invert the values to be able to use Hu's persistent homology package
            _input = 1 - input.cpu().detach()[i]
            _target = 1 - target.cpu().detach()[i]
            p = mp.Process(target=compute_loss_and_gradient, args=(grid, _input, _target, threshold, 
                                                                          i, loss_dict, gradient_dict))
            jobs.append(p)
            p.start()

        # Wait for all processes to finish, then retrieve information and terminate processes
        for proc in jobs:
            proc.join()

        # Sum up losses and create list of gradient tensors
        grad_list = []
        for i in range(0, len(input)):
            loss += loss_dict[i]
            grad_list.append(gradient_dict[i])

        ctx.grad_list = grad_list

        return torch.tensor(loss)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of topological loss. As the gradient is already calculated in the forward pass, 
        this is just a simple matter of exctaction from the context manager.
        """         
        # Stack all gradient tensors in list along z-direction
        return torch.stack(ctx.grad_list, dim=0), None


def compute_loss_and_gradient(grid, input, target, threshold, slice_index, loss_dict, gradient_dict):
    """
    Function that uses a subdivision scheme in order to accelerate the computation and increase its precision.
    The input image is subdivided into a grid of smaller tiles and then the persistent homology is computed on these tiles.
    After that, the are reassembled and the gradient matrix is reconstructed.

    :param grid: tuple of integers
        Number of cells and columns (cells, columns).
    :param input: pytorch.tensor
        Input tensor containing the prediction of shape (x-size, y-size).
    :param target: pytorch.tensor
        Target tensor containing the ground truth of shape (x-size, y-size).
    :param threshold: float
        Threshold for persistent homology calculation.
    :param slice_index: int
        Specifies the slice from which to log the prediction, gradients and persistent homology for display in tensorboard.
        Also is used to specify the position of the loss and gradient of this slice in the loss_dict and gradient_dict.
    :param loss_dict: dict
        Dictionary containing all the losses from the different slice computed in separated processes. 
        Is modified inplace to avoid memory corruption etc. to allow parallelization.
    :param gradient_dict: dict
        Dictionary containing all the gradients from the different slice computed in separated processes. 
        Is modified inplace to avoid memory corruption etc. to allow parallelization.
    """
    # Pixel width and height of the the image of course has to be divisible by the number of grid tiles
    assert input.shape[0] % grid[0] == 0 and input.shape[1] % grid[1] == 0, "Input shape not divisible by grid size!"
    assert target.shape[0] % grid[0] == 0 and target.shape[1] % grid[1] == 0, "Target shape not divisible by grid size!"
    m, n = grid
    M = int(input.shape[0]/m)
    N = int(target.shape[0]/n)

    loss = 0.0
    gradients = np.zeros((input.shape[0], input.shape[1]))
    topo_grad = []
    """
    Go through all the tiles iteratively and calculate the persistent homology as well as the gradient matrix
    and then reassemble the whole gradient matrix from the tiles. 
    Also, create the data required for logging.
    """
    for k in range(0, m*n):
        i = int(k % m)
        j = int(k / m)

        _input = input[M*i : M*(i+1),  N*j : N*(j+1)].numpy()
        _target = target[M*i : M*(i+1), N*j  : N*(j+1)].numpy()

        _loss, _topo_grad = loss_and_gradient(_input, _target, threshold) 

        # Populate gradient matrix
        if _topo_grad.shape != (0,):
            for pos in _topo_grad:
                gradients[int(M*i + pos[1]), int(N*j + pos[0])] = pos[2]
                # topo_grad is ad dictionary only needed for logging of he persistence diagram
                # Thus, we should only construct it if we need it.
                if slice_index == 3:
                    topo_grad.append(np.array([pos[0] + N*j, pos[1] + M*i, pos[2], pos[3]]))
        loss += _loss

    #  Logging 
    if slice_index == 3:
        draw_arrows_and_persistence_diagram(input, target, np.array(topo_grad))
    
    """
    In-place modification of loss_dict and gradient_dict.
    This is necessary due to the usage of multiprocessing.
    Only then, the gradients will be in the right order. 
    """
    loss_dict[slice_index] = loss
    gradient_dict[slice_index] = torch.from_numpy(gradients)

