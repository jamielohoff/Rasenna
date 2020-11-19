import numpy as np
import torch.nn as nn
import torch
import time
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from rasenna.criteria.TopologicalLossFunction import TopologicalLossFunction
from speedrun.log_anywhere import log_scalar
from rasenna.criteria.Subdivision import SubdivisionLossFunction

class TopologicalLoss(nn.Module):
    """
    Computes a weighted loss scalar which uses a balance between Sorensen-Dice loss and topological loss, 
    one for all 20 channels and one acting only on the first three channels to compute boundary probabilities.
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, g_factor, threshold=0.4, use_multiprocessing=False, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param g_factor: float
            This controls the weighting of Sorensen-Dice vs. topological loss
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

        Calculates the weighted sum of Sorensen-Dice loss and topological loss.
        """
        loss = 0.0

        if self.g_factor != 0.0:
            boundary_map = torch.bitwise_or(target[0,0,:,:,:].bool(), target[0,1,:,:,:].bool())
            boundary_map = torch.bitwise_or(boundary_map, target[0,2,:,:,:].bool()).float().detach()
            boundary_prob = (1/3) * (input[0,0,:,:,:] + input[0,1,:,:,:] + input[0,2,:,:,:]).detach()

            # We have to not put a minus sign here, otherwise computation screws up
            sorensen_dice_loss = self.SDLoss(input, target)
            topological_loss = self.TopoLoss(boundary_prob, boundary_map).cuda()

            # Logging of the different losses in tensorboardX
            log_scalar('training_loss/SorensenDice', sorensen_dice_loss)
            log_scalar('training_loss/Topological', topological_loss)
            
            loss = sorensen_dice_loss + self.g_factor * topological_loss
            print('Topological Loss:', topological_loss, 'Sorensen-Dice Loss:', sorensen_dice_loss, "Loss:", loss)
        else:
            loss = self.SDLoss(input, target)
            log_scalar('training_loss/SorensenDice', loss)
            print("Sorensen-Dice Loss:", loss)

        return loss

class TopologicalBCELoss(nn.Module):
    """
    Computes a weighted loss scalar which uses a balance between binary crossentropy loss and topological loss, 
    one for all 20 channels and one acting only on the first three channels to compute boundary probabilities.
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, g_factor, threshold=0.4, use_multiprocessing=False, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param g_factor: float
            This controls the weighting of SorensenDice vs. other Sore Loss
        :param weight: torch.FloatTensor or torch.cuda.FloatTensor
            Class weights: Applies only if `channelwise = True`.
        :param channelwise: bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(TopologicalBCELoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps
        self.g_factor = g_factor

        self.BCELoss = nn.BCELoss(weight=weight)
        self.TopoLoss = TopologicalLossFunction.apply

    def forward(self, input, target):
        """
        Parameters
        ----------
        :param input:      torch.FloatTensor or torch.cuda.FloatTensor
        :param target:     torch.FloatTensor or torch.cuda.FloatTensor
        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        loss = 0.0

        boundary_map = torch.bitwise_or(target[0,0,:,:,:].bool(), target[0,1,:,:,:].bool())
        boundary_map = torch.bitwise_or(boundary_map, target[0,2,:,:,:].bool()).detach
        boundary_prob = (1/3) * (input[0,0,:,:,:] + input[0,1,:,:,:] + input[0,2,:,:,:]).detach

        binary_cross_entropy_loss = self.BCELoss(input.flatten(), target.flatten())
        topological_loss = self.TopoLoss(boundary_prob, boundary_map.float()).cuda()

        # logging of the different losses in tensorboardX
        log_scalar('training_loss/BCE', binary_cross_entropy_loss)
        log_scalar('training_loss/Topological', topological_loss)
        
        loss = binary_cross_entropy_loss + self.g_factor * topological_loss
        print('Topological Loss:', topological_loss, 'BCE Loss:', binary_cross_entropy_loss, "Loss:", loss)

        return loss

class TopologicalLossLogged(nn.Module):
    """
    A loss function that only returns the value of the Sorensen-Dice loss but also calculates the topological loss for comparison.
    """
    def __init__(self, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param g_factor: float
            This controls the weighting of SorensenDice vs. topological loss
        :param weight: torch.FloatTensor or torch.cuda.FloatTensor
            Class weights: Applies only if `channelwise = True`.
        :param channelwise: bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(TopologicalLossLogged, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

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
        loss = 0.0

        boundary_map = torch.bitwise_or(target[0,0,:,:,:].bool(), target[0,1,:,:,:].bool())
        boundary_map = torch.bitwise_or(boundary_map, target[0,2,:,:,:].bool()).detach().float()
        boundary_prob = (1/3) * (input[0,0,:,:,:] + input[0,1,:,:,:] + input[0,2,:,:,:]).detach()

        # We have to not put a minus sign here, otherwise computation screws up
        sorensen_dice_loss = self.SDLoss(input, target)
        topological_loss = self.TopoLoss(boundary_prob, boundary_map).cuda().detach()

        # Logging of the different losses in tensorboardX
        log_scalar('training_loss/SorensenDice', sorensen_dice_loss)
        log_scalar('training_loss/Topological', topological_loss)
        
        loss = sorensen_dice_loss
        print('Topological Loss:', topological_loss, 'Sorensen-Dice Loss:', sorensen_dice_loss, "Loss:", loss)

        return loss

class SubdivisionLoss(nn.Module):
    """
    A loss function that uses the subdivision method/functioin to calculate the topological loss of a voxel. 
    The topological loss is then weighted against the Sorensen-Dice loss.
    """
    def __init__(self, g_factor, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        :param g_factor: float
            This controls the weighting of SorensenDice vs. other Sore Loss
        :param weight: torch.FloatTensor or torch.cuda.FloatTensor
            Class weights: Applies only if `channelwise = True`.
        :param channelwise: bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SubdivisionLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps
        self.g_factor = g_factor

        self.SDLoss = SorensenDiceLoss()
        self.SubdivisionLoss = SubdivisionLossFunction.apply

    def forward(self, input, target):
        """
        Parameters
        ----------
        :param input:      torch.FloatTensor or torch.cuda.FloatTensor
        :param target:     torch.FloatTensor or torch.cuda.FloatTensor
        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        loss = 0.0

        boundary_map = torch.bitwise_or(target[0,0,:,:,:].bool(), target[0,1,:,:,:].bool())
        boundary_map = torch.bitwise_or(boundary_map, target[0,2,:,:,:].bool())
        boundary_prob = (1/3) * (input[0,0,:,:,:] + input[0,1,:,:,:] + input[0,2,:,:,:])

        # We have to invert the values to be able to use persistent homology
        boundary_prob = 1 - boundary_prob
        boundary_map = 1 - boundary_map.float()

        # we have to put a minus sign here, see https://www.jeremyjordan.me/semantic-segmentation/
        sorensen_dice_loss = 1 - self.SDLoss(input, target)
        topological_loss = self.SubdivisionLoss(boundary_prob, boundary_map.float()).cuda()

        # logging of the different losses in tensorboardX
        log_scalar('training_loss/SorensenDice', sorensen_dice_loss)
        log_scalar('training_loss/Topological', topological_loss)
        
        loss = sorensen_dice_loss + self.g_factor * topological_loss
        print('Topological Loss:', topological_loss, 'Sorensen-Dice Loss:', sorensen_dice_loss, "Loss:", loss)

        return loss
