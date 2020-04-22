import numpy as np
import torch
from configs.configs import config
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import center_crop

from albumentations.augmentations.transforms import CenterCrop
import os
import pickle
import cv2

class Unlabeller():
    def __call__(self, x):
        return -1
def preprocess_input(x):
    x /= 127
    return x - 1

import numpy as np
import torch
from configs.configs import config
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import center_crop
from skimage.color import rgb2hed, gray2rgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from albumentations.augmentations.transforms import CenterCrop
import os
import pickle
import cv2

class Unlabeller():
    def __call__(self, x):
        return -1
def preprocess_input(x):
    x /= 127
    return x - 1

def preprocess_input_stain(x):
    x  = x - np.amin(x)
    x = x/np.amax(x)
    return x

class Histodata(Dataset):

    def __init__(self, data_path , pickle_path, budget, unlabeled = True, augment = False):
        self.unlabeled = unlabeled
        self.path = data_path
        self.augment = augment
        normal_label = []
        tumour_label = []
        normal_path = []
        tumour_path = []
        normal_path_all = []
        tumour_path_all = []
        normal_label_all = []
        tumour_label_all = []


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

        if self.unlabeled ==True:

            for image_name in data_budget['training_cam1']['patches']['Normal']:
                normal_path_all.append(os.path.join('Normal', image_name))
                normal_label_all.append(0)
            for image_name in data_budget['training_cam1']['patches']['Tumour']:
                tumour_path_all.append(os.path.join('Tumour', image_name))
                tumour_label_all.append(1)
            self.imgs_all = np.append(normal_path_all, tumour_path_all)
            self.imgs_unlabel = list(set(self.imgs_all) - set(self.imgs))



    def __getitem__(self,index):
        if self.unlabeled==True:
            filename = self.imgs_unlabel[index]
            img = cv2.imread(os.path.join(self.path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # resize image
            width = int(img.shape[1] * 25 / 100)
            height = int(img.shape[0] * 25 / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            if self.augment:
                img = self.augment(img.shape)(image=img)
                img = img['image']
            img = preprocess_input(img.astype(np.float32))
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)

            return img, -1
        else:
            filename = self.imgs[index]
            img = cv2.imread(os.path.join(self.path, filename))
            label = self.labels[index]
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # resize image
            width = int(img.shape[1] * 25 / 100)
            height = int(img.shape[0] * 25 / 100)
            dim = (width, height)
            main_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            if self.augment:
                main_img = self.augment(main_img.shape)(image=main_img)
                main_img = main_img['image']

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
                    aux_image_mag = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    aux_label_mag = 2

                aux_image_mag = preprocess_input(aux_image_mag.astype(np.float32))
                aux_image_mag = torch.from_numpy(aux_image_mag).float()
                aux_image_mag = aux_image_mag.permute(2, 0, 1)
                aux_label_mag = torch.from_numpy(np.array(aux_label_mag)).long()

            if 'stain' in config.task_names:
                stain = np.random.choice(['H', 'E'], 1)
                if stain=='H':
                    aux_image_stain = rgb2hed(main_img)[: ,: ,0]
                    aux_image_stain = preprocess_input_stain(aux_image_stain)
                    aux_image_stain = aux_image_stain[:,:,np.newaxis]
                    aux_image_stain = np.repeat(aux_image_stain,3, axis=2)
                    aux_label_stain = 0
                if stain=='E':
                    aux_image_stain = rgb2hed(main_img)[:, :, 1]
                    aux_image_stain = preprocess_input_stain(aux_image_stain)
                    aux_image_stain = aux_image_stain[:,:,np.newaxis]
                    aux_image_stain = np.repeat(aux_image_stain,3, axis=2)
                    aux_label_stain = 1

                aux_image_stain = torch.from_numpy(aux_image_stain).float()
                aux_image_stain = aux_image_stain.permute(2, 0, 1)
                aux_label_stain = torch.from_numpy(np.array(aux_label_stain)).long()


            main_img = preprocess_input (main_img.astype(np.float32))
            main_img = torch.from_numpy(main_img).float()
            main_img = main_img.permute(2, 0, 1)
            label = torch.from_numpy(np.array(label)).long()
            if 'magnification' in config.task_names and 'stain' not in config.task_names:
                return main_img, label, aux_image_mag, aux_label_mag, -1, -1
            elif 'stain' in config.task_names and 'magnification' not in config.task_names:
                return main_img, label, -1, -1, aux_image_stain, aux_label_stain
            elif 'stain' in config.task_names and 'magnification'  in config.task_names:
                return main_img, label, aux_image_mag, aux_label_mag, aux_image_stain, aux_label_stain
            else:
                return main_img, label, -1, -1, -1, -1

    def __len__(self):
        if self.unlabeled==True:
            return len(self.imgs_unlabel)
        else:
            return len(self.imgs)


class Histodata_unlabel_domain_adopt(Dataset):
    def __init__(self, data_path , pickle_path, budget, unlabeled = True, augment = False):
        self.unlabeled = unlabeled
        self.path = data_path
        self.augment = augment
        normal_label = []
        tumour_label = []
        normal_path = []
        tumour_path = []
        normal_path_all = []
        tumour_path_all = []
        normal_label_all = []
        tumour_label_all = []


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

        if self.unlabeled ==True:

            for image_name in data_budget[config.budget_unlabel]['patches']['Normal']:
                normal_path_all.append(os.path.join('Normal', image_name))
                normal_label_all.append(0)
            for image_name in data_budget[config.budget_unlabel]['patches']['Tumour']:
                tumour_path_all.append(os.path.join('Tumour', image_name))
                tumour_label_all.append(1)
            self.imgs_unlabel = np.append(normal_path_all, tumour_path_all)
            # self.imgs_unlabel = list(set(self.imgs_all) - set(self.imgs))

    def __getitem__(self,index):
        if self.unlabeled==True:
            filename = self.imgs_unlabel[index]
            img = cv2.imread(os.path.join(self.path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image
            width = int(img.shape[1] * 25 / 100)
            height = int(img.shape[0] * 25 / 100)
            dim = (width, height)
            main_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            if self.augment:
                main_img = self.augment(main_img.shape)(image=main_img)
                main_img = main_img['image']
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
                    aux_image_mag = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    aux_label_mag = 2
                if self.augment:
                    aux_image_mag = self.augment(aux_image_mag.shape)(image=aux_image_mag)
                    aux_image_mag = aux_image_mag['image']

                aux_image_mag = preprocess_input(aux_image_mag.astype(np.float32))
                aux_image_mag = torch.from_numpy(aux_image_mag).float()
                aux_image_mag = aux_image_mag.permute(2, 0, 1)

            if 'stain' in config.task_names:
                stain = np.random.choice(['H', 'E'], 1)
                main_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                if stain=='H':
                    aux_image_stain = rgb2hed(main_img)[: ,: ,0]
                    aux_image_stain = preprocess_input_stain(aux_image_stain)
                    aux_image_stain = aux_image_stain[:,:,np.newaxis]
                    aux_image_stain = np.repeat(aux_image_stain,3, axis=2)
                    aux_label_stain = 0
                if stain=='E':
                    aux_image_stain = rgb2hed(main_img)[:, :, 1]
                    aux_image_stain = preprocess_input_stain(aux_image_stain)
                    aux_image_stain = aux_image_stain[:,:,np.newaxis]
                    aux_image_stain = np.repeat(aux_image_stain,3, axis=2)
                    aux_label_stain = 1
                aux_image_stain = torch.from_numpy(aux_image_stain).float()
                aux_image_stain = aux_image_stain.permute(2, 0, 1)

            main_img = preprocess_input (main_img.astype(np.float32))
            main_img = torch.from_numpy(main_img).float()
            main_img = main_img.permute(2, 0, 1)

            if 'magnification' in config.task_names and 'stain' not in config.task_names:
                return main_img, -1, aux_image_mag, aux_label_mag, -1, -1
            elif 'stain' in config.task_names and 'magnification' not in config.task_names:
                return main_img, -1, -1, -1, aux_image_stain, aux_label_stain
            elif 'stain' in config.task_names and 'magnification' in config.task_names:
                return main_img, -1, aux_image_mag, aux_label_mag, aux_image_stain, aux_label_stain
            else:
                return main_img, -1, -1, -1, -1, -1
        else:
            filename = self.imgs[index]
            img = cv2.imread(os.path.join(self.path, filename))
            label = self.labels[index]
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # resize image
            width = int(img.shape[1] * 25 / 100)
            height = int(img.shape[0] * 25 / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            if self.augment:
                img = self.augment(img.shape)(image=img)
                img = img['image']
            img = preprocess_input (img.astype(np.float32))
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
            label = torch.from_numpy(np.array(label)).long()
            return img, label

    def __len__(self):
        if self.unlabeled==True:
            return len(self.imgs_unlabel)
        else:
            return len(self.imgs)