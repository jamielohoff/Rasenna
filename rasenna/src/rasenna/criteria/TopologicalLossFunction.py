import numpy as np
import torch.nn as nn
import torch
import time
import math
from torch.autograd import Function
from .PersistenceUtils import compute_loss_and_gradient
from .Subdivision import subdivide, assemble
from speedrun.log_anywhere import log_scalar, log_image
import multiprocessing as mp

class TopologicalLossFunction(Function):
    """
    This loss-function calculates the topological loss of a gray-scale image/input in 2.5 dimensions, 
    i.e. 1d homology on 3d voxel slices in z-direction. 
    It is based on the code of Xiaoling Hu's paper --Topology-Preserving Deep Image Segmentation--
    You can find the respective code under https://github.com/HuXiaoling/TopoLoss.
    """

    @staticmethod
    def forward(ctx, input, target, threshold=0.4, use_multiprocessing=False):
        """
        Forward pass of topological loss function used to calculate the topological loss between input and target.
        The loss is computed separately for each slice in z-direction and then added up. 
        Not only does the algorithm calculate the loss, but it does also calculate the critical points, 
        i.e. the points that --probably-- need to be fixed to get the right number of holes and thus reduce the loss.
        """
        loss = 0.0
        grad_list = []

        # log images boundary probability and ground truth
        ctx.pred_pic = input[3]
        ctx.target_pic = 1 - target[3]

        if use_multiprocessing == True:
            pool = mp.Pool(input.shape[0])
            print('Amount of workers:', input.shape[0])

            # We have to invert the ground truth (target) to be able to use persistent homology 1 - target.cpu().detach()[i]
            slices = zip(input.cpu().detach()[i], 1 - target.cpu().detach()[i])
            results = [pool.apply(compute_loss_and_gradient, args=((input_slice, target_slice, threshold))) for input_slice, target_slice in slices]
            loss = sum([entry[0] for entry in results])
            grad_list = [torch.from_numpy(entry[1]) for entry in results]

        else:
            for i in range(0, len(input)):  
                # We have to invert the values to be able to use persistent homology
                _loss, _gradient = compute_loss_and_gradient(input.cpu().detach()[i], 1 - target.cpu().detach()[i], threshold)
                loss += _loss
                grad_list.append(torch.from_numpy(_gradient))
        
        ctx.grad_list = grad_list

        return torch.tensor(loss)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of topological loss used to calculate the gradient of the topological loss function.
        To calculate the gradient, we need the critical points and their position in the 2d image.
        We again calculate each gradient separately for each slice and then merge them into one 3d voxel, which is then used for backpropagation.
        """ 
        grad = ctx.grad_list[3].cuda()
        indices = torch.nonzero(grad, as_tuple=True)
        pic = ctx.pred_pic.index_put(indices=indices, values=torch.tensor(0.0))

        red_pic = pic + grad
        green_pic = pic
        blue_pic = pic
        rgb_in_pic = [red_pic.float(), green_pic.float(), blue_pic.float()]

        log_image('output/prediction_and_gradient', torch.stack(rgb_in_pic, dim=0))
        log_image('output/target', torch.stack([ctx.target_pic], dim=0))
        
        # Stack all gradients along z-direction
        return torch.stack(ctx.grad_list, dim=0).cuda(), None

