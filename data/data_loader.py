from data.datasets import Histodata, Histodata_unlabel_domain_adopt
from torch.utils.data import DataLoader
from data.augmentations import get_medium_augmentations


def get_loaders(config):

    lab_train_generator = Histodata(config.base_data_path, config.pickle_path, config.budget, unlabeled=False , augment = get_medium_augmentations)

    unlab_train_generator = Histodata_unlabel_domain_adopt(config.base_data_path_unlabel, config.pickle_path_unlabel,
                                                           config.budget_unlabel, unlabeled = True, augment= get_medium_augmentations)

    valid_generator = Histodata(config.base_data_path_unlabel , config.pickle_path_valid, config.budget_valid, unlabeled = False)
    test_generator = Histodata(config.base_data_path_unlabel , config.pickle_path_test, config.budget_test, unlabeled = False)

    src_loader = DataLoader(lab_train_generator, batch_size=config.src_batch_size, shuffle=True, num_workers=0,
                               pin_memory=False)
    tar_loader = DataLoader(unlab_train_generator, batch_size=config.tar_batch_size , shuffle=True,
                                 num_workers=0, pin_memory=False)
    val_loader = DataLoader(valid_generator, batch_size=config.eval_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_generator, batch_size=config.test_batch_size, shuffle=False, num_workers=0, pin_memory=False)


    return src_loader, tar_loader, val_loader, test_loader


