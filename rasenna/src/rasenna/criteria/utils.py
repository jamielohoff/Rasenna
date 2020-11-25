import torch
import numpy as np 
import scipy as scp
import matplotlib.pyplot as plt
from speedrun.log_anywhere import log_scalar, log_image, log_figure

def draw_dots(image, grad):
    pos_grad = torch.where(grad > 0, grad, torch.zeros_like(grad))
    neg_grad = torch.where(grad < 0, grad, torch.zeros_like(grad))
    indices = torch.nonzero(grad, as_tuple=True)

    img = image.index_put(indices=indices, values=torch.tensor(0.0))

    red_channel = img - neg_grad
    green_channel = img + pos_grad
    blue_channel = img
    rgb_in_pic = [red_channel.float(), green_channel.float(), blue_channel.float()]

    return torch.stack(rgb_in_pic, dim=0)


def draw_persistence_diagram(dgm, colors):
    fig, ax = plt.subplots()
    ax.scatter(dgm[:][0], dgm[:][1])

    ax.set(xlabel='Birth time', ylabel='Death time', title='Persistence diagram')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid()
    return fig


def draw_arrow(image, position, color, grad):
    """
    Function that adds an arrow to a dataset at the position (i,j) with colormap (r,g,b)
    """
    sign = 1
    arrow_offsets = [[0, 0], [1,0], [2, 0], 
                     [3, 0], [4, 0], [5, 0], 
                     [1, 1], [1, -1], [2, 2], 
                     [2, -2], [2, 1], [2, -1], [6, 0]]

    if grad < 0:
        sign = -1

    y = int(position[0])
    x = int(position[1])
    for offset in arrow_offsets:
        image[0][(x + sign*offset[0]) % image[0].shape[0]][(y + offset[1]) % image[0].shape[0]] = color[0]
        image[1][(x + sign*offset[0]) % image[1].shape[0]][(y + offset[1]) % image[1].shape[0]] = color[1]
        image[2][(x + sign*offset[0]) % image[2].shape[0]][(y + offset[1]) % image[2].shape[0]] = color[2]
    return image


def draw_arrows_and_persistence_diagram(input, target, topo_grad):
    assert len(topo_grad) % 2 == 0
    # white = 1 = boundary
    # black = 0 = other stuff
    rgb_in_pic = [1 - input, 1 - input, 1 - input]

    fig, ax = plt.subplots()
    x = np.linspace(0, 1.0, 5)
    ax.plot(x, x, 'k-')
    ax.set_xlabel('Birth time')
    ax.set_ylabel('Death time')

    for j in range(int(len(topo_grad) / 2)):
        i = 2*j
        birth_arrow = [topo_grad[i, 0], topo_grad[i, 1]]
        death_arrow = [topo_grad[i+1, 0], topo_grad[i+1, 1]]

        color = np.random.rand(3)
        rgb_in_pic = draw_arrow(rgb_in_pic, birth_arrow, color, topo_grad[i, 2])
        rgb_in_pic = draw_arrow(rgb_in_pic, death_arrow, color, topo_grad[i+1, 2])

        ax.scatter(topo_grad[i,3], topo_grad[i+1,3], color=(color[0], color[1], color[2]))

    log_image('output/prediction_and_gradient', torch.stack(rgb_in_pic, dim=0))
    log_image('output/target', torch.stack([1 - target], dim=0))
    log_figure('output/barcodes', fig)

