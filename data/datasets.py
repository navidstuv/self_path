import numpy as np
import torch
from configs.configs import config
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import center_crop, jigsaw_res
from skimage.color import rgb2hed, gray2rgb
import matplotlib.pyplot as plt
from data import stainNorm_Reinhard

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
        self.unlabeled = unlabeled
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

        # resize image
        width = int(img.shape[1] * 25 / 100)
        height = int(img.shape[0] * 25 / 100)
        dim = (width, height)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
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

        if 'jigsaw' in config.task_names:
            jig_img, jig_label = jigsaw_res(img)
            jig_img = preprocess_input(jig_img.astype(np.float32))
            jig_img = torch.from_numpy(jig_img).float()
            jig_img = jig_img.permute(2, 0, 1)
            jig_label = torch.from_numpy(np.array(jig_label)).long()
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)
        label = torch.from_numpy(np.array(label)).long()
        if 'magnification' in config.task_names and 'jigsaw' not in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, -1, -1
        elif 'jigsaw' in config.task_names and 'magnification' not in config.task_names:
            return main_img, label, -1, -1, jig_img, jig_label
        elif 'jigsaw' in config.task_names and 'magnification'  in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, jig_img, jig_label
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
                    aux_image_mag = cv2.resize(aux_image_mag, (128, 128), interpolation=cv2.INTER_CUBIC)
                    aux_label_mag = 1
                if mag=='10x':
                    aux_image_mag = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
                    aux_label_mag = 2
                if self.augment:
                    aux_image_mag = self.augment(aux_image_mag.shape)(image=aux_image_mag)
                    aux_image_mag = aux_image_mag['image']

                if config.stain_normalized:
                    aux_image_mag = self.n.transform(aux_image_mag)
                aux_image_mag = preprocess_input(aux_image_mag.astype(np.float32))
                aux_image_mag = torch.from_numpy(aux_image_mag).float()
                aux_image_mag = aux_image_mag.permute(2, 0, 1)

            if 'jigsaw' in config.task_names:
                jig_img, jig_label = jigsaw_res(img)
                jig_img = preprocess_input(jig_img.astype(np.float32))
                jig_img = torch.from_numpy(jig_img).float()
                jig_img = jig_img.permute(2, 0, 1)
                jig_label = torch.from_numpy(np.array(jig_label)).long()
            if config.stain_normalized:
                main_img = self.n.transform(main_img)
            main_img = preprocess_input (main_img.astype(np.float32))
            main_img = torch.from_numpy(main_img).float()
            main_img = main_img.permute(2, 0, 1)

            if 'magnification' in config.task_names and 'stain' not in config.task_names:
                return main_img, -1, aux_image_mag, aux_label_mag, -1, -1
            elif 'jigsaw' in config.task_names and 'magnification' not in config.task_names:
                return main_img, -1, -1, -1, jig_img, jig_label
            elif 'jigsaw' in config.task_names and 'magnification' in config.task_names:
                return main_img, -1, aux_image_mag, aux_label_mag, jig_img, jig_label
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