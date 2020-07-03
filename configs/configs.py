import os
class DefaultConfigs(object):


    mode = 'test'# if mode fine-tune is used weights should be loaded
    encoder_name = 'resnet18'
    pretrained = False
    task_pretrained = True
    stain_normalized = True
    augmentation = True
    exp_name = ' cam-oscc'
    task_names = ['magnification']#['hematoxylin','magnification', 'jigsaw', , eosin]
    aux_task_names =task_names
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3},
             'jigsaw': {'type': 'classification', 'n_classes': 12},
             'domain_classifier': {'type': 'classification', 'n_classes': 2},
             'main_task': {'type': 'classification', 'n_classes': 2},
             'hematoxylin': {'type': 'pixel', 'n_classes': 1}
             }
    loss_weight = {'magnification': 1, 'domain_classifier': 1, 'jigsaw': 1, 'hematoxylin': 1, 'main_task':1}

    log_dir = './magnification_f_all_self0.5/logs'
    cache_dir = './magnification_f_all_self0.5/cache'
    model_dir = './magnification_f_all_self0.5/model'
    best_model_dir = './magnification_f_all_self0.5/best_model'
    training_resume = './magnification_self/model/model_last.pth'
    training_num_print_epoch = 20


    #source domain
    src_batch_size = 64
    base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    # base_data_path = '/media/navid/SeagateBackupPlusDrive/test_camelyon'
    pickle_path = 'pickle_files/training_cam.pickle'
    budget = 'training_cam0.5'

    #target domain
    tar_batch_size = 64
    # base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_unlabel= './pickle_files/training.pickle'
    budget_unlabel = 'training1'

    # validation
    pickle_path_valid = './pickle_files/validation_cam.pickle'
    budget_valid = 'validation_cam1'

    # test
    # base_data_path_unlabel_new = 'G://test_camelyon'
    # base_data_path_test = '/media/navid/SeagateBackupPlusDrive/test_camelyon'
    base_data_path_test = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_test = './pickle_files/test.pickle'
    budget_test = 'test1'

    testing_model ='./magnification_f_all_self0.5/best_model/model_best_acc.pth'


    save_output = True
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 33
    num_epochs = 100
    gpus = [1]
    lr =  0.001



config = DefaultConfigs()