import os
class DefaultConfigs(object):


    mode = 'train'
    encoder_name = 'resnet50'
    pretrained = False
    stain_normalized = True
    augmentation = True
    exp_name = ' cam-oscc'
    task_names = ['main_task']#['main_task', 'magnification', 'jigsaw', 'domain_classifier']
    aux_task_names =task_names[1:]
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3},
             'main_task': {'type': 'classification', 'n_classes': 2},
             'jigsaw': {'type': 'classification', 'n_classes': 12},
             'domain_classifier': {'type': 'classification', 'n_classes': 2},
             'hematoxylin': {'type': 'pixel', 'n_classes': 1}
             }
    loss_weight = {'magnification': 1, 'domain_classifier': 1, 'main_task': 1, 'jigsaw': 1, 'hematoxylin': 1}

    log_dir = './test_main_semi0.2/logs'
    cache_dir = './test_main_semi0.2/cache'
    model_dir = './test_main_semi0.2/model'
    best_model_dir = './test_main_semi0.2/best_model'

    training_resume = ''
    training_num_print_epoch = 20


    #source domain
    src_batch_size = 64
    # base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    base_data_path = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path = 'pickle_files/training_balanced.pickle'
    budget = 'training0.2'

    #target domain
    tar_batch_size = 64
    # base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_unlabel= 'pickle_files/training_balanced.pickle'
    budget_unlabel = 'training1'

    # validation
    pickle_path_valid = './pickle_files/validation_balanced.pickle'
    budget_valid = 'validation1'

    # test
    # base_data_path_unlabel_new = 'G://test_camelyon'
    pickle_path_test = './pickle_files/test_balanced.pickle'
    budget_test = 'test1'

    testing_model ='./test_magnification_semi0.05/best_model/model_best.pth'


    save_output = True
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 33
    num_epochs = 100
    gpus = [0]
    lr =  0.001



config = DefaultConfigs()