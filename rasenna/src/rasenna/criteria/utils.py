import torch
import numpy as np 

def draw_arrow(data, position, color, shape='up'):
    """
    Function that adds an arrow to a dataset at the position (i,j) with colormap (r,g,b)
    """
    sign = 1
    arrow_offsets = [[0, 0], [1,0], [2, 0], 
                     [3, 0], [4, 0], [5, 0], 
                     [1, 1], [1, -1], [2, 2], 
                     [2, -2], [2, 1], [2, -1], [6, 0]]

    if shape == 'down':
        sign = -1

    for offset in arrow_offsets:
        data[position[0] + sign*offset[0], position[1] + offset[1], 0] = color[0]
        data[position[0] + sign*offset[0], position[1] + offset[1], 1] = color[1]
        data[position[0] + sign*offset[0], position[1] + offset[1], 2] = color[2]
    return data

def draw_arrows(data, points, color, shape):
    """
    Function that draws multiple arrows of same shape at different positions.
    """
    for point in points:
        data = draw_arrow(data, point, color, shape)
    return data




