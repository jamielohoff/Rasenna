import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import multiprocessing as mp
import torch
import time
import math
from torch.autograd import Function
from .PersistenceUtils import compute_loss_and_gradient

class TopologicalLossFunction(Function):
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

        # setting up the multiprocessing manager
        manager = mp.Manager()
        loss_dict = manager.dict()
        gradient_dict = manager.dict()

        # create as many separate processes as there are slices in z-direction for 
        # maximum parallelization as persistent homology is calculated only in xy-plane
        for i in range(0, len(input)):  
            # We have to invert the values to be able to use Hu's persistent homology package
            p = mp.Process(target=compute_loss_and_gradient, args=(1 - input.cpu().detach()[i], 
                                                                    1 - target.cpu().detach()[i], 
                                                                    threshold, i, loss_dict, gradient_dict))
            jobs.append(p)
            p.start()

        # wait for all processes to finish, then retrieve information and terminate processes
        for proc in jobs:
            proc.join()

        # sum up losses and create list of gradient tensors
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

