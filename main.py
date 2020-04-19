import random
from utils.dirs import create_dirs
from configs.configs import config
from utils.utils import  get_logger, set_seed

from models.all_models import AuxModel
from data.data_loader import get_loaders

def main():

    # create the experiments dirs
    create_dirs([config.cache_dir, config.model_dir,
                 config.log_dir , config.best_model_dir])

    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)

    logger.info('Random seed: {:d}'.format(config.random_seed))

    # model = get_model(config)
    model = AuxModel(config, logger)



    src_loader, tar_loader, val_loader, test_loader  = get_loaders(config)


    if config.mode == 'train':
        model.train(src_loader, tar_loader, val_loader, None)
    elif config.mode == 'test':
        model.test(test_loader)


if __name__ == '__main__':
    set_seed(config.random_seed)
    main()

