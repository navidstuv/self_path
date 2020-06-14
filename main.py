
from configs.configs import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus).strip('[]')
from utils.dirs import create_dirs
from configs.configs import config
from utils.utils import  get_logger
from utils.utils import set_seed
from models.model import get_model
from models.all_models import AuxModel
from data.data_loader import get_loaders
import warnings
warnings.filterwarnings("ignore")
def main():

    # create the experiments dirs
    create_dirs([config.cache_dir, config.model_dir,
                 config.log_dir , config.best_model_dir])

    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)

    # fix random seed to reproduce results
    logger.info('Random seed: {:d}'.format(config.random_seed))

    # model = get_model(config)
    model = AuxModel(config, logger)


    src_loader, _, val_loader, test_loader  = get_loaders(config)


    if config.mode == 'train':
        model.train(src_loader)
    elif config.mode == 'val':
        model.test(val_loader)
    elif config.mode == 'test':
        model.test(test_loader)
    elif config.mode == 'fine-tune':
        model.fine_tune(src_loader, val_loader, test_loader)


if __name__ == '__main__':
    set_seed(config.random_seed)
    main()

