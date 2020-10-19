import numpy as np
import torch.nn as nn
import torch
import time
import math
from torch.autograd import Function
from .PersistenceUtils import compute_persistence_2DImg, compute_dgm_force
from .Subdivision import subdivide, assemble
from speedrun.log_anywhere import log_scalar, log_image

class TopologicalLossFunction(Function):
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

        # log images boundary prob, boundary map
        ctx.pred_pic = input[3]
        ctx.target_pic = target[3]

        for i in range(0, len(input)):
            input_dgms, input_birth_cp, input_death_cp = compute_persistence_2DImg(input.cpu().detach()[i], dimension=1)
            target_dgms, target_birth_cp, target_death_cp = compute_persistence_2DImg(target.cpu().detach()[i], dimension=1)

            # save critical points and persistence diagrams for back-propagation
            dgm_list.append((input_dgms, input_birth_cp, input_death_cp, target_dgms))

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(input_dgms, target_dgms)
            for idx in idx_holes_to_fix:
                loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
            for idx in idx_holes_to_remove:
                loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2

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

        for list_slice in dgm_list:
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
            topo_grad[:, 2] = topo_grad[:, 2] * -2 # TODO clarify the role of the minus sign here!!!

            gradients = np.zeros((ctx.x_size, ctx.y_size))

            for pos in topo_grad:
                gradients[int(pos[1]), int(pos[0])] = pos[2]            
            grad_list.append(torch.from_numpy(gradients))
        
        grad = grad_list[3].cuda()
        indices = torch.nonzero(grad, as_tuple=True)
        pic = ctx.pred_pic.index_put(indices=indices, values=torch.tensor(0.0))

        red_pic = pic + grad
        green_pic = pic
        blue_pic = pic
        rgb_in_pic = [red_pic.float(), green_pic.float(), blue_pic.float()]

        log_image('output/prediction_and_gradient', torch.stack(rgb_in_pic, dim=0))
        log_image('output/target', torch.stack([ctx.target_pic], dim=0))
        
        # Stack all gradients along z-direction
        return torch.stack(grad_list, dim=0).cuda(), None

