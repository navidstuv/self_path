import os
class DefaultConfigs(object):


    mode = 'train'
    encoder_name = 'resnet50'
    pretrained = False
    exp_name = ' cam-oscc'
    task_names = ['main_task','domain_classifier']#['main_task', 'magnification', 'stain', 'domain_classifier']
    aux_task_names =task_names[1:]
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3},
             'main_task': {'type': 'classification', 'n_classes': 2},
             'stain': {'type': 'classification', 'n_classes': 2},
             'domain_classifier': {'type': 'classification', 'n_classes': 2}}
    loss_weight = {'magnification': 0.2, 'domain_classifier':0.2, 'main_task': 1, 'stain':0.2}

    log_dir = './test/logs'
    cache_dir = './test/cache'
    model_dir = './test/model'
    best_model_dir = './test/best_model'
    training_resume = ''
    training_num_print_epoch = 1000


    #source domain
    src_batch_size = 32
    base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    pickle_path = 'pickle_files/training_cam.pickle'
    budget = 'training_cam1'

    #target domain
    tar_batch_size = 32
    base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_unlabel= './pickle_files/training.pickle'
    budget_unlabel = 'training1'



    #validation
    pickle_path_valid = './pickle_files/validation_cam.pickle'
    budget_valid = 'validation_cam1'


    #test
    pickle_path_test = './pickle_files/test.pickle'
    budget_test = 'test1'
    testing_model ='/media/navid/HDD1/Back_up/experiments_log/domain_adoptation/stain1/best_model/model_best.pth'

    save_output = True
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 22
    num_epochs = 100
    gpus = [0,1]
    lr =  0.001



config = DefaultConfigs()