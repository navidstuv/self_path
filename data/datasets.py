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

            mag = np.random.choice(['40x', '20x', '10x'], 1)
            if mag=='40x':
                aux_image = center_crop(img, 128, 128)
                aux_label = 0
            if mag=='20x':
                aux_image = center_crop(img, 256, 256)
                aux_image = cv2.resize(aux_image, (128, 128), interpolation=cv2.INTER_AREA)
                aux_label = 1
            if mag=='10x':
                aux_image = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                aux_label = 2

            aux_image = preprocess_input (aux_image.astype(np.float32))
            aux_image = torch.from_numpy(aux_image).float()
            aux_image = aux_image.permute(2, 0, 1)
            aux_label = torch.from_numpy(np.array(aux_label)).long()

            main_img = preprocess_input (main_img.astype(np.float32))
            main_img = torch.from_numpy(main_img).float()
            main_img = main_img.permute(2, 0, 1)
            label = torch.from_numpy(np.array(label)).long()
            return main_img, label, aux_image, aux_label
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

            mag = np.random.choice(['40x', '20x', '10x'], 1)
            if mag=='40x':
                aux_image = center_crop(img, 128, 128)
                aux_label = 0
            if mag=='20x':
                aux_image = center_crop(img, 256, 256)
                aux_image = cv2.resize(aux_image, (128, 128), interpolation=cv2.INTER_AREA)
                aux_label = 1
            if mag=='10x':
                aux_image = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                aux_label = 2

            if self.augment:
                aux_image = self.augment(aux_image.shape)(image=aux_image)
                aux_image = aux_image['image']

            aux_image = preprocess_input (aux_image.astype(np.float32))
            aux_image = torch.from_numpy(aux_image).float()
            aux_image = aux_image.permute(2, 0, 1)


            return aux_image, aux_label
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