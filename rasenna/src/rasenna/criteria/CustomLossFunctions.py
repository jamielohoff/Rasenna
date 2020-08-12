import numpy as np
import torch.nn as nn
import torch
import time
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from rasenna.criteria.TopologicalLossFunction import TopologicalLossFunction

class CombinedLoss(nn.Module):
    """
    Computes a weighted loss scalar which uses a balance between a SoerensenDice Loss on all 20 Channels
    and a CrossEntropy Loss on the first three channels between the input and the target. 
    The second loss is used to compute the boundary probabilities.
    For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, g_factor, weight=None, channelwise=True, eps=1e-6):
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
        super(CombinedLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps
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
        CE_Loss = nn.BCELoss()

        # Remove when we do not need it anymore
        print(CE_Loss(boundary_prob.view(-1), boundary_map.view(-1).float()))
        
        loss = SD_Loss(input, target) + self.g_factor * CE_Loss(boundary_prob.view(-1), boundary_map.view(-1).float())
        return loss

class TopologicalLoss(nn.Module):
    """
    Computes a weighted loss scalar which uses a balance between two SorensenDice Losses, 
    one for all 20 channels and one acting only on the first three channels to compute boundary probabilities.
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, g_factor, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param g: float
            This controls the weighting of SorensenDice vs. other Sore Loss
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
        self.g_factor = g_factor

        self.SDLoss = SorensenDiceLoss()
        self.TopoLoss = TopologicalLossFunction.apply

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

        print('Shapes: ', boundary_prob.shape)

        # Remove when we do not need it anymore
        topological_loss = self.TopoLoss(boundary_prob, boundary_map.float()).cuda()
        sorensen_dice_loss = self.SDLoss(input, target)
        
        loss = sorensen_dice_loss + self.g_factor * topological_loss

        print('Losses:', topological_loss, sorensen_dice_loss, loss)

        return loss

