"""Compatibility path for the primary BreaKHis loader."""
if __package__ and "." in __package__:
    from ..DataPreprocessing import (
        get_class_names, get_dataloaders, get_datasets, image_size,
        set_seed, test_transform, train_transform,
    )
else:
    from DataPreprocessing import (
        get_class_names, get_dataloaders, get_datasets, image_size,
        set_seed, test_transform, train_transform,
    )
