import os
class DefaultConfigs(object):

    mode = 'train'
    encoder_name = 'wide_resnet'
    pretrained = False
    stain_normalized = False
    augmentation = True
    training_num_print_epoch = 20
    save_output = True
    eval_batch_size = 128
    test_batch_size = 128
    random_seed = 33
    num_epochs = 300
    gpus = [0]
    lr =  0.05
    weight_decay = 10e-3
    src_batch_size = 32
    tar_batch_size = 32
    dataset = 'kather'
    # for resumin training
    training_resume = ''

    task_names = ['main_task', 'jigsaw']#['main_task', 'magnification', 'jigsaw', 'domain_classifier', hematoxylin, 'rot']
    aux_task_names =task_names[1:]
    tasks = {'magnification': {'type': 'classification_self', 'n_classes': 3},
             'main_task': {'type': 'classification_main', 'n_classes': 9},
             'jigsaw': {'type': 'classification_self', 'n_classes': 12},
             'domain_classifier': {'type': 'classification_adapt', 'n_classes': 2},
             'hematoxylin': {'type': 'pixel_self', 'n_classes': 1},
             'flip': {'type': 'classification_self', 'n_classes': 2},
             'rot': {'type': 'classification_self', 'n_classes': 4},
             'auto': {'type': 'pixel_self', 'n_classes': 3}
             }
    loss_weight = {'magnification': 1, 'domain_classifier': 1,
                   'main_task': 1, 'jigsaw': 1, 'hematoxylin': 1,
                   'flip': 1, 'rot':1, 'auto': 1}

    annotation_budget = 0.00125

    log_dir = './exp/' + dataset + '/'
    cache_dir = './exp/' + dataset + '/'
    model_dir = './exp/' + dataset + '/'
    best_model_dir = './exp/' + dataset + '/'
    for task_name in task_names:
        log_dir = log_dir +'_' +task_name[:3]
        cache_dir = cache_dir +'_' +task_name[:3]
        model_dir = model_dir +'_' +task_name[:3]
        best_model_dir = best_model_dir +'_' +task_name[:3]
    log_dir = log_dir + str(annotation_budget)+'/logs'
    cache_dir = cache_dir + str(annotation_budget)+'/cache'
    model_dir = model_dir + str(annotation_budget)+'/model'
    best_model_dir = best_model_dir + str(annotation_budget)+'/best_model'




    if dataset == 'kather':

        base_data_path = '/media/navid/SeagateBackupPlusDrive/DATA/Kather/train'
        pickle_path = './pickle_files/training_kather2.pickle'
        budget = 'training_kather2' + str(annotation_budget)

        base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/DATA/Kather/train'
        pickle_path_unlabel = './pickle_files/training_kather.pickle'
        budget_unlabel = 'training_kather1'

        val_data_path = '/media/navid/SeagateBackupPlusDrive/DATA/Kather/validation'
        pickle_path_valid = './pickle_files/validation_kather.pickle'
        budget_valid = 'validation_kather1'

        test_data_path = '/media/navid/SeagateBackupPlusDrive/DATA/Kather/test'
        pickle_path_test = './pickle_files/test_kather.pickle'
        budget_test = 'test_kather1'
        class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    if dataset == 'oscc':
        #source domain path
        # base_data_path = '/media/navid/SeagateBackupPlusDrive/512allcamelyon'
        base_data_path = '/media/navid/SeagateBackupPlusDrive/512all'
        pickle_path = 'pickle_files/training_balanced.pickle'
        budget = 'training' + str(annotation_budget)

        #target domain path
        # base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
        base_data_path_unlabel = '/media/navid/SeagateBackupPlusDrive/512all'
        pickle_path_unlabel= 'pickle_files/training_balanced.pickle'
        budget_unlabel = 'training1'

        # validation path
        val_data_path = '/media/navid/SeagateBackupPlusDrive/512all'
        pickle_path_valid = './pickle_files/validation_balanced.pickle'
        budget_valid = 'validation1'

        # test path
        # base_data_path_unlabel_new = 'G://test_camelyon'
        test_data_path = '/media/navid/SeagateBackupPlusDrive/512all'
        pickle_path_test = './pickle_files/test_balanced.pickle'
        budget_test = 'test1'
        class_names = []
    testing_model ='./exp/kather/_mai0.00125/best_model/model_best_acc.pth'

config = DefaultConfigs()