import os
class DefaultConfigs(object):


    mode = 'test'
    encoder_name = 'resnet50'
    pretrained = False
    stain_normalized = True
    augmentation = True
    exp_name = ' cam-oscc'
    task_names = ['main_task', 'jigsaw']#['main_task', 'magnification', 'jigsaw', 'domain_classifier']
    aux_task_names =task_names[1:]
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3},
             'main_task': {'type': 'classification', 'n_classes': 2},
             'jigsaw': {'type': 'classification', 'n_classes': 8},
             'domain_classifier': {'type': 'classification', 'n_classes': 2}}
    loss_weight = {'magnification': 1, 'domain_classifier': 1, 'main_task': 1, 'jigsaw': 1}

    log_dir = './testtt/logs'
    cache_dir = './testtt/cache'
    model_dir = './testtt/model'
    best_model_dir = './testtt/best_model'
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
    pickle_path_valid = './pickle_files/validation.pickle'
    budget_valid = 'validation1'

    # test
    # base_data_path_unlabel_new = 'G://test_camelyon'
    pickle_path_test = './pickle_files/test.pickle'
    budget_test = 'test1'
    testing_model ='testtt/best_model/model_best.pth'

    save_output = True
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 33
    num_epochs = 100
    gpus = [0,1]
    lr =  0.001



config = DefaultConfigs()