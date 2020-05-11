import numpy as np
import torch
from configs.configs import config
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import center_crop
from skimage.color import rgb2hed, gray2rgb
import matplotlib.pyplot as plt
from data import stainNorm_Reinhard
from albumentations.augmentations import functional as F_aug
from data.augmentations import get_medium_augmentations

from albumentations.augmentations.transforms import CenterCrop
import os
import pickle
import cv2

class Unlabeller():
    def __call__(self, x):
        return -1
def preprocess_input(x):
    x /= 255
    return x

def preprocess_input_stain(x, maxx = 1, minn = -1):
    if np.amax(x)==0 and np.amin(x)==0 or (np.amax(x)-np.amin(x))==0:
        std=np.zeros_like(x)
    else:
        if (np.amax(x) - np.amin(x)) ==0:
            print((np.amax(x) - np.amin(x)))
        std  = (x - np.amin(x))/(np.amax(x) - np.amin(x))
    x = std*(maxx - minn) + minn
    return x

class Histodata(Dataset):

    def __init__(self, data_path , pickle_path, budget, unlabeled = True, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        normal_label = []
        tumour_label = []
        normal_path = []
        tumour_path = []

        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for image_name in data_budget[budget]['patches']['Normal']:
            normal_path.append(os.path.join('Normal', image_name))
            normal_label.append(0)
        for image_name in data_budget[budget]['patches']['Tumour']:
            tumour_path.append(os.path.join('Tumour', image_name))
            tumour_label.append(1)
        self.imgs = np.append(normal_path,tumour_path)
        self.labels = np.append(normal_label,tumour_label)

    def __getitem__(self,index):
        filename = self.imgs[index]
        img = cv2.imread(os.path.join(self.path, filename))
        label = self.labels[index]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)

        if self.augment:
            img = self.augment(img.shape)(image=img)
            img = img['image']
        # resize image
        width = int(img.shape[1] * 25 / 100)
        height = int(img.shape[0] * 25 / 100)
        dim = (width, height)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if 'magnification' in config.task_names:
            mag = np.random.choice(['40x', '20x', '10x'], 1)
            if mag=='40x':
                aux_image_mag = center_crop(img, 128, 128)
                aux_label_mag = 0
            if mag=='20x':
                aux_image_mag = center_crop(img, 256, 256)
                aux_image_mag = cv2.resize(aux_image_mag, (128, 128), interpolation=cv2.INTER_AREA)
                aux_label_mag = 1
            if mag=='10x':
                aux_image_mag = main_img
                aux_label_mag = 2
            aux_image_mag = preprocess_input(aux_image_mag.astype(np.float32))
            aux_image_mag = torch.from_numpy(aux_image_mag).float()
            aux_image_mag = aux_image_mag.permute(2, 0, 1)
            aux_label_mag = torch.from_numpy(np.array(aux_label_mag)).long()

        if 'gamma' in config.task_names:
            gamma = np.random.randint(50, 150) / 100.0
            # print(gamma)
            aux_image_gamma = F_aug.gamma_transform(main_img, gamma=gamma)
            aux_label_gamma = gamma
            aux_image_gamma = preprocess_input(aux_image_gamma.astype(np.float32))
            aux_image_gamma = torch.from_numpy(aux_image_gamma).float()
            aux_image_gamma = aux_image_gamma.permute(2, 0, 1)
            aux_label_gamma = torch.from_numpy(np.array(aux_label_gamma)).float()
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)
        label = torch.from_numpy(np.array(label)).long()
        if 'magnification' in config.task_names and 'gamma' not in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, -1, -1
        elif 'gamma' in config.task_names and 'magnification' not in config.task_names:
            return main_img, label, -1, -1, aux_image_gamma, aux_label_gamma
        elif 'gamma' in config.task_names and 'magnification'  in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, aux_image_gamma, aux_label_gamma
        else:
            return main_img, label, -1, -1, -1, -1

    def __len__(self):
        return len(self.imgs)


class Histodata_unlabel_domain_adopt(Dataset):
    def __init__(self, data_path , pickle_path, budget, unlabeled = True, augment = False):
        self.unlabeled = unlabeled
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        normal_path_all = []
        tumour_path_all = []
        normal_label_all = []
        tumour_label_all = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)

        if self.unlabeled ==True:
            for image_name in data_budget[config.budget_unlabel]['patches']['Normal']:
                normal_path_all.append(os.path.join('Normal', image_name))
                normal_label_all.append(0)
            for image_name in data_budget[config.budget_unlabel]['patches']['Tumour']:
                tumour_path_all.append(os.path.join('Tumour', image_name))
                tumour_label_all.append(1)
            self.imgs_unlabel = np.append(normal_path_all, tumour_path_all)

    def __getitem__(self,index):
        filename = self.imgs_unlabel[index]
        img = cv2.imread(os.path.join(self.path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)
        if self.augment:
            img = self.augment(img.shape)(image=img)
            img = img['image']
        # resize image
        width = int(img.shape[1] * 25 / 100)
        height = int(img.shape[0] * 25 / 100)
        dim = (width, height)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if 'magnification' in config.task_names:
            mag = np.random.choice(['40x', '20x', '10x'], 1)
            if mag=='40x':
                aux_image_mag = center_crop(img, 128, 128)
                aux_label_mag = 0
            if mag=='20x':
                aux_image_mag = center_crop(img, 256, 256)
                aux_image_mag = cv2.resize(aux_image_mag, (128, 128), interpolation=cv2.INTER_AREA)
                aux_label_mag = 1
            if mag=='10x':
                aux_image_mag = main_img
                aux_label_mag = 2

            aux_image_mag = preprocess_input(aux_image_mag.astype(np.float32))
            aux_image_mag = torch.from_numpy(aux_image_mag).float()
            aux_image_mag = aux_image_mag.permute(2, 0, 1)

        if 'gamma' in config.task_names:
            gamma = np.random.randint(50, 150) / 100.0
            # print(gamma)
            aux_image_gamma = F_aug.gamma_transform(main_img, gamma=gamma)
            aux_label_gamma = gamma
            aux_image_gamma = preprocess_input(aux_image_gamma.astype(np.float32))
            aux_image_gamma = torch.from_numpy(aux_image_gamma).float()
            aux_image_gamma = aux_image_gamma.permute(2, 0, 1)
            aux_label_gamma = torch.from_numpy(np.array(aux_label_gamma)).float()
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)

        if 'magnification' in config.task_names and 'stain' not in config.task_names:
            return main_img, -1, aux_image_mag, aux_label_mag, -1, -1
        elif 'stain' in config.task_names and 'magnification' not in config.task_names:
            return main_img, -1, -1, -1, aux_image_gamma, aux_label_gamma
        elif 'stain' in config.task_names and 'magnification' in config.task_names:
            return main_img, -1, aux_image_mag, aux_label_mag, aux_image_gamma, aux_label_gamma
        else:
            return main_img, -1, -1, -1, -1, -1

    def __len__(self):
        if self.unlabeled==True:
            return len(self.imgs_unlabel)
        else:
            return len(self.imgs)

def show_images(images, cols=1, titles=None):
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

if __name__=='__main__':

    from torch.utils.data import DataLoader


    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    lab_train_generator = Histodata(config.base_data_path_unlabel, '../pickle_files/training.pickle', 'training1', unlabeled=False , augment = False)
    unlab_train_generator = Histodata_unlabel_domain_adopt(config.base_data_path_unlabel, '../pickle_files/training.pickle',
                                                           config.budget_unlabel, unlabeled = True, augment= False)
    src_loader = DataLoader(unlab_train_generator, batch_size=config.src_batch_size, shuffle=True, num_workers=20,
                            pin_memory=True, worker_init_fn=worker_init_fn)
    for it, src in enumerate(src_loader):
        img,lbl, aux_img,aux_lbs,_,_ = src
        img = img.permute(0, 2, 3, 1)
        aux_img = aux_img.permute(0, 2, 3, 1)
        show_images(aux_img.numpy(), cols=5, titles=aux_lbs.numpy())



