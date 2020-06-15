import os
class DefaultConfigs(object):


    mode = 'feature_extractor'# if mode fine-tune is used weights should be loaded
    encoder_name = 'resnet50'
    pretrained = False
    stain_normalized = True
    augmentation = True
    exp_name = ' cam-oscc'
    task_names = ['main']#['hematoxylin','magnification', 'jigsaw', , eosin]
    aux_task_names =task_names
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3},
             'jigsaw': {'type': 'classification', 'n_classes': 12},
             'main': {'type': 'classification', 'n_classes': 2},
             'hematoxylin': {'type': 'pixel', 'n_classes': 1}
             }
    loss_weight = {'magnification': 1, 'domain_classifier': 1, 'jigsaw': 1, 'hematoxylin': 1, 'main':1}

    log_dir = './hematoxylin_self/logs'
    cache_dir = './hematoxylin_self/cache'
    model_dir = './hematoxylin_self/model'
    best_model_dir = './hematoxylin_self/best_model'

    training_resume = ''
    training_num_print_epoch = 20


    #source domain
    src_batch_size = 64
    # base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    pickle_path = 'pickle_files/training_cam.pickle'
    budget = 'training_cam1'

    #target domain
    tar_batch_size = 64
    # base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_unlabel= './pickle_files/training.pickle'
    budget_unlabel = 'training1'

    # validation
    pickle_path_valid = './pickle_files/validation.pickle'
    budget_valid = 'validation1'

    # test
    # base_data_path_unlabel_new = 'G://test_camelyon'
    pickle_path_test = './pickle_files/test.pickle'
    budget_test = 'test1'

    testing_model ='/media/navid/HDD1/Back_up/experiments_log/domain_adoptation/main_normalized2/best_model/model_best.pth'


    save_output = True
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 33
    num_epochs = 100
    gpus = [1]
    lr =  0.001



config = DefaultConfigs()