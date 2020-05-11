import os
class DefaultConfigs(object):


    mode = 'train'
    encoder_name = 'resnet50'
    pretrained = False
    stain_normalized = True
    augmentation = False
    exp_name = ' cam-oscc'
    task_names = ['main_task', 'magnification']#['main_task', 'magnification', 'gamma', 'domain_classifier']
    aux_task_names =task_names[1:]
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3},
             'main_task': {'type': 'classification', 'n_classes': 2},
             'gamma': {'type': 'regression', 'n_classes': None},
             'domain_classifier': {'type': 'classification', 'n_classes': 2}}
    loss_weight = {'magnification': 0.2, 'domain_classifier':0.2, 'main_task': 1, 'stain':0.2}

    log_dir = './main_magnification_normalized1/logs'
    cache_dir = './main_magnification_normalized1/cache'
    model_dir = './main_magnification_normalized1/model'
    best_model_dir = './main_magnification_normalized1/best_model'
    training_resume = 'model_last.pth'
    training_num_print_epoch = 20


    #source domain
    src_batch_size = 128
    # base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    pickle_path = 'pickle_files/training_cam.pickle'
    budget = 'training_cam1'

    #target domain
    tar_batch_size = 128
    # base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_unlabel= './pickle_files/training.pickle'
    budget_unlabel = 'training1'

    # validation
    pickle_path_valid = './pickle_files/validation_cam.pickle'
    budget_valid = 'validation_cam1'

    # test
    # base_data_path_unlabel_new = 'G://test_camelyon'
    pickle_path_test = './pickle_files/test.pickle'
    budget_test = 'test1'
    testing_model ='/media/navid/HDD1/Back_up/experiments_log/domain_adoptation/main_dom3_1/best_model/model_best.pth'

    save_output = False
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 22
    num_epochs = 100
    gpus = [0,1]
    lr =  0.001



config = DefaultConfigs()