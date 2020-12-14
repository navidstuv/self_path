from data.datasets import Histodata, Histodata_unlabel_domain_adopt, Histodata_main, Histodata_jigsaw,\
    Histodata_magnification, Histodata_hematoxylin, Histodata_auto, Histodata_flip, Histodata_rot

from torch.utils.data import DataLoader
from data.augmentations import get_medium_augmentations
from utils.utils import worker_init_fn

all_loader = {'main_task': Histodata_main, 'magnification': Histodata_magnification,
              'jigsaw': Histodata_jigsaw, 'hematoxylin': Histodata_hematoxylin,
              'rot': Histodata_rot, 'auto': Histodata_auto, 'flip':Histodata_flip}


def get_loaders(config):
    """

    :param config:
    :return: src_loader and tar_loader ad dictionary and val_loader an test_loader
    """
    if config.augmentation == True:
        augmentation = get_medium_augmentations
    else:
        augmentation = False

    # for labled data
    src_loader = {}
    tar_loader = {}
    for task in config.task_names:
        lab_train_generator = all_loader[task](config.base_data_path, config.pickle_path, config.budget,
                                        augment=augmentation)
        src_loader[task] = DataLoader(lab_train_generator, batch_size=config.src_batch_size, shuffle=True, num_workers=0,
                                pin_memory=True, worker_init_fn=worker_init_fn)

        unlab_train_generator = all_loader[task](config.base_data_path_unlabel,
                                                               config.pickle_path_unlabel,
                                                               config.budget_unlabel,
                                                               augment=augmentation)

        tar_loader[task] = DataLoader(unlab_train_generator, batch_size=config.tar_batch_size, shuffle=True,
                                num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)




    valid_generator = Histodata_main(config.val_data_path, config.pickle_path_valid, config.budget_valid)
    test_generator = Histodata_main(config.test_data_path, config.pickle_path_test, config.budget_test)
    val_loader = DataLoader(valid_generator, batch_size=config.eval_batch_size, shuffle=False, num_workers=0,
                            pin_memory=True)
    test_loader = DataLoader(test_generator, batch_size=config.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    return src_loader, tar_loader, val_loader, test_loader
