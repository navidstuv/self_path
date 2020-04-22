import os
class DefaultConfigs(object):


    mode = 'train'
    encoder_name = 'Disc128'
    pretrained = False
    gan_latent_dim = 100
    exp_name = ' cam-oscc'
    task_names = ['main_task']# ['main_task', 'magnification','domain_classifier','stain']
    aux_task_names = []#['magnification', 'domain_classifier', 'stain']
    tasks = {'magnification': {'type': 'classification', 'n_classes': 3}, 'main_task': {'type': 'classification', 'n_classes': 2},
             'domain_classifier': {'type': 'classification', 'n_classes': 2},
             'stain': {'type': 'classification', 'n_classes': 2}}
    loss_weight = {'magnification': 0.2, 'main_task': 1, 'domain_classifier':0.2}

    log_dir = 'GAN_da_classifier/logs'
    cache_dir = 'GAN_da_classifier/cache'
    model_dir = 'GAN_da_classifier/model'
    best_model_dir = 'GAN_da_classifier/best_model'
    training_resume = ''
    training_num_print_epoch = 1


    #source domain
    src_batch_size = 16
    base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
    pickle_path = 'pickle_files/training_cam.pickle'
    budget = 'training_cam1'

    #target domain
    tar_batch_size = 16
    base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
    pickle_path_unlabel= 'pickle_files/training.pickle'
    budget_unlabel = 'training1'



    #validation
    pickle_path_valid = 'pickle_files/validation.pickle'
    budget_valid = 'validation1'
    validation_model = 'model_002120.pth'


    #test
    pickle_path_test = 'pickle_files/test.pickle'
    budget_test = 'test1'
    testing_model ='model_002120.pth'

    save_output = False
    eval_batch_size = 128
    test_batch_size = 128



    random_seed = 22
    num_epochs = 100

    optimizer = 'sgd'
    lr =  0.0003
    weight_decay= 0.0005
    momentum= 0.9
    nesterov= True
    gpus = '0'
config = DefaultConfigs()