from copy import deepcopy
from torch.utils.data.dataloader import DataLoader, default_collate
import numpy as np

try:
    from inferno.io.core import ZipReject, Concatenate
    from inferno.io.transform import Compose, Transform
    from inferno.io.transform.generic import AsTorchBatch
    from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
    from inferno.io.transform.image import RandomRotate, ElasticTransform
except ImportError:
    raise ImportError("CremiDataset requires inferno")

try:
    from neurofire.datasets.loader import RawVolume, SegmentationVolume, RawVolumeWithDefectAugmentation
    from neurofire.transform.artifact_source import RejectNonZeroThreshold
    from neurofire.transform.volume import RandomSlide
except ImportError:
    raise ImportError("CremiDataset requires neurofire")

from ..utils.various import yaml2dict
from ..transform.volume import MergeExtraMasks, DuplicateGtDefectedSlices, DownSampleAndCropTensorsInBatch, \
    ReplicateTensorsInBatch
from ..transform.affinities import affinity_config_to_transform, Segmentation2AffinitiesDynamicOffsets


class CremiDataset(ZipReject):
    """
    Load the following volumes:

    - Raw image with defect augmentation ('raw' entry in `volume_config`)
    - GT segmentation volume ('segmentation' entry in `volume_config`)
    - Optionally, another label volume containing boundary masks, defect masks, glia masks, etc...
        ('extra_masks' entry in `volume_config`)

    In order, the following transformations/augmentations are applied, according to `master_config` and
    `defect_augmentation_config`:

        - Add defected slices, black slices and/or artifacts to the raw data (with option to keep track of which slice
            contains artifacts)
        - Normalize raw
        - Flip/rotate
        - On (tracked) defected slices, predict GT of the previous slice
        - Elastic transformation
        - Random slide in the xy plane and crop to certain shape to avoid border artifacts (also from elastic
                transform)
        - Create multiple cropped/downscaled versions of the input (for multi-scale UNet architecture)
        - Compute binary affinity targets from GT labels given certain offsets
                (taking care of ignore_label, boundary and glia labels)
        - Crop some extra pad to avoid boundary artifacts (`crop_after_target`)
    """

    def __init__(self, name, volume_config, slicing_config,
                 defect_augmentation_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert isinstance(defect_augmentation_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        volume_config = deepcopy(volume_config)

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))

        # check if we have special dict entries for names in the defect augmentation
        # slicing config
        augmentation_config = deepcopy(defect_augmentation_config)
        for slicing_key, slicing_item in augmentation_config['artifact_source']['slicing_config'].items():
            if isinstance(slicing_item, dict):
                new_item = augmentation_config['artifact_source']['slicing_config'][slicing_key][name]
                augmentation_config['artifact_source']['slicing_config'][slicing_key] = new_item

        raw_volume_kwargs.update({'defect_augmentation_config': augmentation_config})
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        ignore_slice_list = deepcopy(augmentation_config.get("ignore_slice_list", None))
        if isinstance(ignore_slice_list, dict):
            ignore_slice_list = ignore_slice_list.get(name, [])
        self.raw_volume = RawVolumeWithDefectAugmentation(name=name,
                                                          ignore_slice_list=ignore_slice_list,
                                                          **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name,
                                                      **segmentation_volume_kwargs)

        volumes_to_load = [self.raw_volume, self.segmentation_volume]

        # Load additional masks:
        self.extra_masks_volume = None
        if volume_config.get('extra_masks', False):
            extra_masks_kwargs = dict(volume_config.get('extra_masks'))
            extra_masks_kwargs.update(slicing_config)
            self.extra_masks_volume = SegmentationVolume(name=name,
                                                         **extra_masks_kwargs)
            volumes_to_load.append(self.extra_masks_volume)

        rejection_threshold = volume_config.get('rejection_threshold', 0.5)
        super().__init__(*volumes_to_load,
                         sync=True, rejection_dataset_indices=1,
                         rejection_criterion=RejectSingleLabelVolumes(1.0, rejection_threshold,
                                                                      defected_label=master_config.get('duplicate_GT_defected_slices', {}).get('defect_label', -1)))
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else deepcopy(master_config)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose()

        transforms.add(MergeExtraMasks(defects_label=self.master_config.get('defects_label', 3)))

        if self.master_config.get('random_flip', False):
            transforms.add(RandomFlip3D())
            transforms.add(RandomRotate())

        transforms.add(DuplicateGtDefectedSlices(
            defects_label=self.master_config.get('defects_label', 3),
            ignore_label=self.master_config.get('ignore_label', 0))
        )

        # Elastic transforms can be skipped by
        # setting elastic_transform to false in the
        # yaml config file.
        if self.master_config.get('elastic_transform'):
            elastic_transform_config = self.master_config.get('elastic_transform')
            if elastic_transform_config.get('apply', False):
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

        # random slide augmentation
        if self.master_config.get('random_slides') is not None:
            random_slides_config = deepcopy(self.master_config.get('random_slides'))
            ouput_shape = random_slides_config.pop('shape_after_slide', None)
            max_misalign = random_slides_config.pop('max_misalign', None)
            transforms.add(RandomSlide(
                output_image_size=ouput_shape, max_misalign=max_misalign,
                **random_slides_config))

        # Replicate and downscale batch:
        nb_inputs = 1
        if self.master_config.get("downscale_and_crop") is not None:
            ds_config = self.master_config.get("downscale_and_crop")
            apply_to  = [conf.pop('apply_to') for conf in ds_config]
            nb_inputs = (np.array(apply_to) == 0).sum()
            transforms.add(ReplicateTensorsInBatch(apply_to))
            for indx, conf in enumerate(ds_config):
                transforms.add(DownSampleAndCropTensorsInBatch(apply_to=[indx], order=None, **conf))

        # Check if to compute binary-affinity-targets from GT labels:
        if self.master_config.get("affinity_config") is not None:
            affs_config = deepcopy(self.master_config.get("affinity_config"))
            global_kwargs = affs_config.pop("global", {})

            aff_transform = Segmentation2AffinitiesDynamicOffsets if affs_config.pop("use_dynamic_offsets", False) \
                else affinity_config_to_transform

            for input_index in affs_config:
                affs_kwargs = deepcopy(global_kwargs)
                affs_kwargs.update(affs_config[input_index])
                transforms.add(aff_transform(apply_to=[input_index+nb_inputs], **affs_kwargs))

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.master_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        master_config = config.get('master_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   defect_augmentation_config=defect_augmentation_config,
                   master_config=master_config)


class CremiDatasets(Concatenate):
    def __init__(self, names,
                 volume_config,
                 slicing_config,
                 defect_augmentation_config,
                 master_config=None):
        # Make datasets and concatenate
        if names is None:
            datasets = [CremiDataset(name=None,
                                     volume_config=volume_config,
                                     slicing_config=slicing_config,
                                     defect_augmentation_config=defect_augmentation_config,
                                     master_config=master_config)]
        else:
            datasets = [CremiDataset(name=name,
                                     volume_config=volume_config,
                                     slicing_config=slicing_config,
                                     defect_augmentation_config=defect_augmentation_config,
                                     master_config=master_config)
                        for name in names]
        super().__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   defect_augmentation_config=defect_augmentation_config,
                   slicing_config=slicing_config, master_config=master_config)


class RejectSingleLabelVolumes(object):
    def __init__(self, threshold, threshold_zero_label=1.,
                 defected_label=None):
        """
        :param threshold: If the biggest segment takes more than 'threshold', batch is rejected
        :param threshold_zero_label: if the percentage of non-zero-labels is less than this, reject
        """
        self.threshold = threshold
        self.threshold_zero_label = threshold_zero_label
        self.defected_label = defected_label

    def __call__(self, fetched):
        counts = np.bincount(fetched.flatten())
        # Check if we should reject:
        return ((float(np.max(counts)) / fetched.size) > self.threshold) or (
                    (counts[1:].sum() / fetched.size) < self.threshold_zero_label)



def get_cremi_loader(config):
    """
    Get Cremi loader given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    loader_config = config.get('loader_config')
    inference_mode = config.get('inference_mode', False)

    if inference_mode:
        raise NotImplementedError("Inference loader not yet implemented")
        datasets = CremiDatasetInference(
            config.get("master_config"),
            name=config.get('name'),
            **config.get('volume_config'))
        # Avoid to wrap arrays into tensors:
        loader_config["collate_fn"] = collate_indices
    else:
        datasets = CremiDatasets.from_config(config)
    # Don't wrap stuff in tensors:
    loader = DataLoader(datasets, **loader_config)
    return loader


def collate_indices(batch):
    tensor_list = [itm[0] for itm in batch]
    indices_list = [itm[1] for itm in batch]
    return default_collate(tensor_list), indices_list
