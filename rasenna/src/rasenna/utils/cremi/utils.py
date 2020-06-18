import h5py
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def open_hdf_file(filename, key=None):
    """
    Opens .h5- and .hdf-files and returns all data under the given key as a numpy array.
    When no key is specified, the function returns the entire dataset

    Parameters
    ----------
    :param filename: Path to h5py.Dataset
    """
    with h5py.File(filename, 'r') as f:
        if key is not None:
            assert key in f.keys()
            data = np.array(f[key])
            return data
        else:
            print('Please specify a key!')


def get_hdf_keys(filename, print_keys=False):
    """
    Function that returns all keys of a given h5-file.
    To print the keys on the console, set print_keys=True.

    Parameters
    ----------
    filename : Path to HDF5-Dataset
    """
    with h5py.File(filename, 'r') as f:
        keys = f.keys()
    if print_keys:
        print("Keys: %s" % keys)
    return keys


# Code copied from: https://github.com/cremi/cremi_python/blob/master/cremi/evaluation/border_mask.py
def create_border_mask(input_data, target, max_dist, background_label, axis=0):
    """
    Overlay a border mask from input_data with background_label onto target data.
    A pixel is part of a border if one of its 4-neighbors has different label.
    
    Parameters
    ----------
    input_data : h5py.Dataset or numpy.ndarray - Input data containing neuron ids.
    target : h5py.Datset or numpy.ndarray - Target on which the borders from the input data is overlayed.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    background_label : int - Border mask will be overlayed using this label.
    axis : int - Axis of iteration (perpendicular to 2d images for which mask will be generated)
    """
    sl = [slice(None) for d in range(len(target.shape))]

    for z in range(target.shape[axis]):
        sl[axis] = z
        border = create_border_mask_2d(input_data[tuple(sl)], max_dist)
        target_slice = target[tuple(sl)] if isinstance(target, h5py.Dataset) else np.copy(target[tuple(sl)])
        target_slice[border] = background_label
        target[tuple(sl)] = target_slice


# Code copied from: https://github.com/cremi/cremi_python/blob/master/cremi/evaluation/border_mask.py
def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.
    
    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)
    
    padded = np.pad(image, 1, mode='edge')
    
    border_pixels = np.logical_and(
        np.logical_and( image == padded[:-2, 1:-1], image == padded[2:, 1:-1] ),
        np.logical_and( image == padded[1:-1, :-2], image == padded[1:-1, 2:] )
        )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
        )

    return distances <= max_dist


def extract_random_slices(dataset, depth, height, width, nsamples):
    """
    Function to extract 3-dimensional slices from a given
    dataset with resolution depth x height x width.

    nsamples: number of "depth x height x width"-slices
    """
    data = []
    for i in range(0, nsamples):
        z = np.random.randint(0, dataset.shape[0] - depth + 1)
        x = np.random.randint(0, dataset.shape[1] - height + 1)
        y = np.random.randint(0, dataset.shape[2] - width + 1)
        data.append(dataset[z:z + depth, x:x + height, y:y + width])
    data = np.array(data)

    intervals = [(z, z + depth), (x, x + height), (y, y + width)]

    return data, intervals


def save_everywhere(filename, data):
    plt.imsave(filename, data)

