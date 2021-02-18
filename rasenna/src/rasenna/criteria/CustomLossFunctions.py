import numpy as np
import torch.nn as nn
import torch
import time
import logging 
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from rasenna.criteria.TopologicalLossFunction import TopologicalLossFunction
from speedrun.log_anywhere import log_scalar, log_image
from rasenna.criteria.utils import prepare_target

class TopologicalLoss(nn.Module):
    """
    Computes a weighted loss scalar which uses a balance between the affinity channel and the boundary channel of the model.
    On the boundary channel, we use a Sorensen-Dice loss while on the boundary channel, 
    we use combination of topological and Sorensen-Dice loss.
    """
    def __init__(self, SD_weight, topo_weight, topo_SD_weight, enable_logging=True, pretraining=False, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param SD_weight: float
            This controls the weighting of the affinity channel Sorensen-Dice vs. boundary branch losses
        :param topo_weight: float
            This controls the weighting of topological loss on the boundary branch.
        :param topo_SD_weight: float
            This controls the weighting of Sorensen-Dice loss on the boundary branch.
        :param pretraining: bool
            If you want to pretrain your model on the affinity branch with a SD only, set this falg to true.

        :param weight: torch.FloatTensor or torch.cuda.FloatTensor
            Class weights: Applies only if `channelwise = True`.
        :param channelwise: bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(TopologicalLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

        self.SD_weight = SD_weight
        self.topo_weight = topo_weight
        self.topo_SD_weight = topo_SD_weight
        self.pretraining = pretraining

        self.SDLoss = SorensenDiceLoss()
        self.TopoLoss = TopologicalLossFunction.apply

        # initialise logging
        if enable_logging:
            self.log = logging.getLogger()
            self.log.setLevel(logging.INFO)


    def forward(self, input, target):
        """
        Forward pass of the loss. It returns a loss which is a weighted linear combination of the three losses.
        The 1st channel of input and target are the boundary parts,
        while the 0th channel contains the stuff required for the affinity computations.
        """
        # target for topological part
        boundary = target[1] 
        # do some data engineering 
        target, boundary_mask, boundary_contour = prepare_target(target[0]) 

        loss = 0.0
        sorensen_dice_loss = 0.0
        topological_loss = 0.0
        topo_sorensen_dice_loss = 0.0

        if not self.pretraining:
            # create boundary map and boundary probabilies form input and target tensors
            start = time.time()
            b_map = boundary[0,0,:,:,:].float().detach().cpu()
            maximum = torch.max(b_map)
            boundary_map = torch.where(b_map < maximum, torch.tensor(0.0), torch.tensor(1.0)).detach()
            boundary_prob = input[1][0, 0, :, :, :].cpu() * boundary_mask.cpu() + boundary_contour

            boundary_prob_ax = torch.stack([boundary_prob], dim=0)
            boundary_map_ax = torch.stack([boundary_map], dim=0)
            data_end = time.time()

            # Losses on the boundary branch
            topological_loss = self.TopoLoss(boundary_prob, boundary_map).cuda()
            end = time.time()
            self.log.info('Time elapsed to calculate topological features and gradient: ' + str(end - start) + ' s')
            topo_sorensen_dice_loss = boundary_map_ax.shape[1] + self.SDLoss(boundary_prob_ax, boundary_map_ax).cuda()

        # Sorensen-Dice loss on affinity channel
        print(input[0].shape, target.shape)
        sorensen_dice_loss = target.shape[1] + self.SDLoss(input[0], target)

        # logging of the different losses in tensorboardX
        log_scalar('training_loss/SorensenDiceAffinityBranch', sorensen_dice_loss)
        log_scalar('training_loss/SorensenDiceBoundaryBranch', topo_sorensen_dice_loss)
        log_scalar('training_loss/TopologicalLoss', topological_loss)

        if self.SD_weight > 1e-6:
            loss += self.SD_weight * sorensen_dice_loss

        if self.topo_weight > 1e-6 and self.pretraining == False:
            loss += self.topo_weight * topological_loss
        
        if self.topo_SD_weight > 1e-6 and self.pretraining == False:
            loss += self.topo_SD_weight * topo_sorensen_dice_loss

        self.log.info('Affinity SD Loss (weighted): ' + str(self.SD_weight * sorensen_dice_loss))
        self.log.info('Boundary SD Loss (weighted): ' + str(self.topo_SD_weight * topo_sorensen_dice_loss))
        self.log.info('Topological Loss (weighted): ' + str(self.topo_weight * topological_loss))
        self.log.info("Loss: " + str(loss))

        return loss

