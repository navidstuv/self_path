import os
class DefaultConfigs(object):


    mode = 'train'
    encoder_name = 'resnet50'
    pretrained = False
    exp_name = ' cam-oscc'
    task_names = ['main_task', 'magnification']
    aux_task_names = ['magnification']
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3}, 'main_task': {'type': 'classification', 'n_classes': 2} }
    loss_weight = {'magnification': 0.2, 'main_task': 1}

    log_dir = 'E:\Back_up\experiments_log\domain_adoptation\main_task_mag\logs'
    cache_dir = 'E:\Back_up\experiments_log\domain_adoptation\main_task_mag\cache'
    model_dir = 'E:\Back_up\experiments_log\domain_adoptation\main_task_mag\model'
    best_model_dir = 'E:\Back_up\experiments_log\domain_adoptation\main_task_mag\\best_model'
    training_resume = ''
    training_num_print_epoch = 20

    #source domain
    src_batch_size = 32
    base_data_path = 'G:\\512allcamelyon'
    pickle_path = 'E:\Back_up\git-files\Multi_task_domain_adapt\pickle_files\\training_cam.pickle'
    budget = 'training_cam1'

    #target domain
    tar_batch_size = 32
    base_data_path_unlabel = 'G:\\512all'
    pickle_path_unlabel= 'E:\Back_up\git-files\Multi_task_domain_adapt\pickle_files\\training.pickle'
    budget_unlabel = 'training1'



    #validation
    pickle_path_valid = 'E:\Back_up\git-files\Multi_task_domain_adapt\pickle_files\\validation_cam.pickle'
    budget_valid = 'validation_cam1'


    #test
    pickle_path_test = 'E:\Back_up\git-files\Multi_task_domain_adapt\pickle_files\\test.pickle'
    budget_test = 'test1'
    testing_model ='model_028623.pth'

    save_output = False
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 22
    num_epochs = 100

    optimizer = 'sgd'
    lr =  0.001
    weight_decay= 0.0005
    momentum= 0.9
    nesterov= True

lr_scheduler= {'name': 'step', 'step_size': 24}



validation_model=''

testing_model=''

config = DefaultConfigs()