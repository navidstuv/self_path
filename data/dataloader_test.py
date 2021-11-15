from torch.utils.data import Dataset
from config import config
from itertools import chain
from glob import glob
from tqdm import tqdm
# from .augmentations import get_train_transform,get_test_transform
import random
import numpy as np
import pandas as pd
import os
import cv2
import torch
import openslide as op
import pickle
from torch.utils.data import DataLoader

# 1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


def preprocess_input(x):
    x /= 127
    return x - 1

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label

def get_files(root, mode):
    # for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename": files})
        return files
    elif mode != "test":
        # for train and val
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: root + x, os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files
    else:
        print("check the mode please!")


def creat_dataset(wsi_path, patches_path, pickle_path, max_patches, class_names):
    '''
    Function for sorting patch names in to their corresponding classes:
    example:
    The code returns a dictionary like :{"Train1":{"WSI":[wsi names],"pathces":{"Normal":[patch names], "Tumour":[pathc_names]}},
    "Train0.5":....}

    :param wsi_path:
    :param patches_path:
    :param pickle_path: where pickle will be saved
    :param max_patches: maximium number of patches to include from each WSI
    :param class_names: directory name of classes
    :return:
    '''
    np.random.seed(111)  # 111 222
    wsi_path = wsi_path
    split = 'test'
    # to extract names of WSIs
    image_names = []
    items_wsi = os.listdir(wsi_path)
    # --------------------------------------------
    for names in items_wsi:
        if names.endswith(".svs"):
            image_names.append(names[:-4])
        elif names.endswith((".tiff")):
            image_names.append(names[:-5])
    # --------------------------------------------
    data_budget = {}
    cs = {}
    patches = {}
    percentages = [1, 0.8, 0.5, 0.2, 0.1, 0.05]
    # percentages = [1]
    indices = list(range(len(image_names)))
    np.random.shuffle(indices)
    for percentage in percentages:
        if percentage == 1:
            sample_indices = indices[:int(np.round(len(indices) * percentage))]
            cs['WSI'] = [image_names[i] for i in sample_indices]
            for class_name in class_names:
                patche_names = []
                all_image = os.listdir(os.path.join(patches_path, class_name))
                for wsi_name in cs['WSI']:
                    image_name_patches = [this_image for this_image in all_image if wsi_name in this_image]
                    np.random.shuffle(image_name_patches)
                    if len(image_name_patches) > max_patches:
                        image_name_patches = image_name_patches[:max_patches]
                        patche_names = np.append(patche_names, image_name_patches)
                        # patche_names.append(image_name_patches)
                    else:
                        patche_names = np.append(patche_names, image_name_patches)
                        # patche_names.append(image_name_patches)
                patches[class_name] = patche_names
            patches1 = patches.copy()
            cs['patches'] = patches1
            ad = cs.copy()
            data_budget[split + str(percentage)] = ad


        else:
            sample_indices = indices[:int(np.round(len(indices) * percentage))]
            cs['WSI'] = [image_names[i] for i in sample_indices]
            for class_name in class_names:
                patche_names = []
                for wsi_name in cs['WSI']:
                    image_name_patches = [this_image for this_image in data_budget[split + '1']['patches'][class_name]
                                          if wsi_name in this_image]
                    if len(image_name_patches) > max_patches:
                        image_name_patches = image_name_patches[:max_patches]
                        patche_names = np.append(patche_names, image_name_patches)
                        # patche_names.append(image_name_patches)
                    else:
                        patche_names = np.append(patche_names, image_name_patches)
                        # patche_names.append(image_name_patches)
                patches[class_name] = patche_names
            patches1 = patches.copy()
            cs['patches'] = patches1
            ad = cs.copy()
            data_budget[split + str(percentage)] = ad
    with open(os.path.join(pickle_path + 'test512-tiff.pickle'), 'wb') as f:
        pickle.dump(data_budget, f)


def show_images(images, iter, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(10, 10))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        a.set_title(str(title))
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    # plt.savefig('../patches/'+str(iter)+'.png')
    # plt.close()


if __name__ == '__main__':
    # # to creat pickle file for training validation or test (test)
    # wsi_path = 'E:\Back_up\git-files\OSCC_detection\Data\OSCC_LNM\sorted\Test'
    # patches_path = 'G:\Data512'
    # pickle_path = 'logs'
    # max_patches = 500
    # class_names = ['Normal', 'Tumour']
    # creat_dataset(wsi_path, patches_path, pickle_path, max_patches, class_names)
# ------------------------------------------------------------------------------------------------------
# to visualize output of dataloader
    import numpy as np
    import matplotlib.pyplot as plt

    wsi_path = 'E:\Back_up\git-files\OSCC_detection\Data\OSCC_LNM\sorted\Train\\09-3162B2.svs'
    polygon_path = 'E:\Back_up\git-files\OSCC_detection\Data\OSCC_LNM\sorted\Train\\09-3162B2.xml'
    val_dataloader = DataLoader(
        Histodata_wsi(wsi_path, polygon_path,augment=False),
        batch_size=100, shuffle=False, pin_memory=True, num_workers=4)
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 2
    for iter, (input, target) in enumerate(val_dataloader):
        # for i in range(1, columns * rows + 1):
        #     fig.add_subplot(rows, columns, i)
        #     img = input[i-1].permute(1,2,0)
        #     plt.imshow(img)
        # plt.show()
        img = list(input.permute(0, 2, 3, 1).numpy())
        show_images(img,iter, cols=10, titles=target.tolist())




