import h5py as h5py
import numpy as np
from shutil import copyfile
from os.path import isfile
from utils import create_border_mask_2d, open_hdf_file


def get_image_boundary(image, id, maxdist=1):
    """
    Function to calculate the boundaries of the segmented image based on the raw data and the neuron id's
    """
    assert image.shape == id.shape

    width  = image.shape[0]
    height  = image.shape[1]

    boundaries = np.zeros(image.shape, dtype=np.int32)
    for i in range(0, width):
        for j in range(0,height):
            if id[i, (j-maxdist)%height] != id[i, j] or id[i, (j+maxdist)%height] != id[i, j] or id[(i + maxdist) % width, j] != id[i, j] or id[(i - maxdist) % width, j] != id[i, j]:
                if id[i,j] != 0:
                    boundaries[i, j] = 1
    
    return boundaries


def create_boundary_map(path='', output_file=''):
    """
    :param path: Path to the Cremi files
    :param 
    """
    print('Loading...')
    raw_data = open_hdf_file(path, key='volumes/raw')
    ids = open_hdf_file(path, key='volumes/labels/neuron_ids')

    print('Calculating...')
    data = np.zeros(raw_data.shape)
    i = 0
    for img,img_ids in zip(raw_data, ids):
        print('Image', i+1 , 'of', raw_data.shape[0])
        boundary = get_image_boundary(img, img_ids, maxdist=3)
        data[i] = boundary
        i += 1

    print('Writing to', output_file,'...')
    with h5py.File(output_file, 'w') as f:
        dset = f.create_dataset('boundary_maps', data=data)
    print('Done!')

def merge_cremi(cremi_file='', boundary_file='', output_file=''):
    boundaries = open_hdf_file(boundary_file, key='boundary_maps')

    print('Copying', cremi_file,'to', output_file, '...')
    if isfile(output_file):
        print('File already exists.')
    else:
        copyfile(cremi_file, output_file)
    print('Done.\n Beginning merge with boundary map...')
    with h5py.File(output_file, 'r+') as f:
        assert 'volumes/labels/boundaries' not in f.keys(), "Data has already been merged!"
        dset = f.create_dataset('volumes/labels/boundary_maps', data=boundaries, dtype='int32')
    print('Done.')

def modify_full_cremi(croppedFiles=None):
    fullpath = '/export/home/jgrieser/datasets/CREMI/'
    paddedFiles = ['paddedData/sample_A', 'paddedData/sample_B', 'paddedData/sample_C']

    # Use the files for the neuronal network also as source files for th boundaries
    if croppedFiles is None:
        croppedFiles = paddedFiles

    for crop_file in croppedFiles:
        if not isfile(fullpath + crop_file + '_boundary_maps.h5'):
            create_boundary_map(fullpath + crop_file + '.h5', output_file=fullpath + crop_file + '_boundary_maps.h5')
        else:
            print('File exists, skipping...')
            continue
    
    for pad_file,bound_file in zip(paddedFiles, croppedFiles):
        if not isfile(fullpath + pad_file + '_with_boundaries.h5'):
            merge_cremi(cremi_file=fullpath + pad_file + '.h5', 
                        boundary_file=fullpath + crop_file+'_boundary_maps.h5',
                        output_file=fullpath + pad_file + '_with_boundaries.h5')
        else:
            print('File exists, skipping...')
            continue

