import pytorch


import numpy as np
import torch.nn as nn
import torch
import time
from topologylayer.nn import LevelSetLayer2D
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss

class TopologicalLoss(nn.Module):
    """
    Topological Loss implementation. Do more commenting
    For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, g_factor, gridsize, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param g: float
            This controls the weighting of SorensenDice vs. CrossEntropy Loss.
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

        # add input such that level set layer is easier to modify
        print('Initializing LevelSetLayer2D...')
        start_time = time.time()
        self.topolayer = LevelSetLayer2D(size=(gridsize, gridsize), maxdim=1, sublevel=False)
        elapsed_time = time.time() - start_time
        print('Done after', elapsed_time,'seconds.')

        self.g_factor = g_factor

    def forward(self, input, target):
        """
        Parameters
        ----------
        :param input:      torch.FloatTensor or torch.cuda.FloatTensor
        :param target:     torch.FloatTensor or torch.cuda.FloatTensor
        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        boundary_map = torch.bitwise_or(target[0,0,:,:,:].bool(), target[0,1,:,:,:].bool())
        boundary_map = torch.bitwise_or(boundary_map, target[0,2,:,:,:].bool())
        boundary_prob = (1/3) * (input[0,0,:,:,:] + input[0,1,:,:,:] + input[0,2,:,:,:])

        SD_Loss = SorensenDiceLoss()
        boundary_prob_dgms, issublevelset = self.topolayer(boundary_prob[0].float())
        boundary_map_dgms, issublevelset_ = self.topolayer(boundary_map[0].float())

        print(target.shape)
        
        loss = SD_Loss(input, target)
        return loss

    def calculate_dmgs(self, boundary_map):
        return 0
