import numpy as np
import torch
from configs.configs import config
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import center_crop, jigsaw_res
from skimage.color import rgb2hed, gray2rgb
import matplotlib.pyplot as plt
from data import stainNorm_Reinhard
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import imutils
import random

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
# TO DO: Separate dataloader for each task
def preprocess_input_stain(x, maxx = 1, minn = -1):
    if np.amax(x)==0 and np.amin(x)==0 or (np.amax(x)-np.amin(xHistodata_hematoxylin))==0:
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
            i1 = cv2.imread('./data/source.png')
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
        print(f'number of normal classes: {len(normal_label)}')
        for image_name in data_budget[budget]['patches']['Tumour']:
            tumour_path.append(os.path.join('Tumour', image_name))
            tumour_label.append(1)
        print(f'number of tumour classes: {len(tumour_label)}')
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
        if 'hematoxylin' in config.task_names:
            ihc_hed = rgb2hed(main_img)
            hem_img = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
            hem_img = torch.from_numpy(np.array(hem_img)).long()
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)
        label = torch.from_numpy(np.array(label)).long()
        if 'magnification' in config.task_names and 'jigsaw' not in config.task_names and 'hematoxylin' not in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, -1, -1, -1
        elif 'jigsaw' in config.task_names and 'magnification' not in config.task_names and 'hematoxylin' not in config.task_names:
            return main_img, label, -1, -1, jig_img, jig_label, -1
        elif 'hematoxylin' in config.task_names and 'magnification' not in config.task_names and 'jigsaw' not in config.task_names:
            return main_img, label, -1, -1, -1, -1, hem_img
        elif 'hematoxylin' in config.task_names and 'magnification' in config.task_names and 'jigsaw' not in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, -1, -1, hem_img
        elif 'hematoxylin' in config.task_names and 'jigsaw' in config.task_names and 'magnification' not in config.task_names:
            return main_img, label, -1, -1, jig_img, jig_label, hem_img
        elif 'magnification' in config.task_names and 'jigsaw' in config.task_names and 'hematoxylin' not in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, jig_img, jig_label, -1
        elif 'jigsaw' in config.task_names and 'magnification'  in config.task_names and 'hematoxylin'  in config.task_names:
            return main_img, label, aux_image_mag, aux_label_mag, jig_img, jig_label, hem_img
        else:
            return main_img, label, -1, -1, -1, -1, -1

    def __len__(self):
            return len(self.imgs)

class Histodata_hematoxylin(Dataset):

    def __init__(self, data_path, pickle_path, budget, augment=False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        label = []
        img_path = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for i, class_name in enumerate(data_budget[budget]['patches'].keys()):
            img_path = img_path + [os.path.join(class_name, x) for x in data_budget[budget]['patches'][class_name]]
            num_imgs = len(data_budget[budget]['patches'][class_name])
            label = label + [i] * num_imgs
            print(f'number of images in class {class_name} are {num_imgs}')
        self.imgs = img_path
        self.labels = label

    def __getitem__(self,index):
        filename = self.imgs[index]
        img = cv2.imread(os.path.join(self.path, filename))
        label = self.labels[index]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)

        # resize image
        dim = (128, 128)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        ihc_hed = rgb2hed(main_img)
        hem_img = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
        hem_img = torch.from_numpy(np.array(hem_img)).long()
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)

        return main_img, hem_img

    def __len__(self):
            return len(self.imgs)

class Histodata_main(Dataset):

    def __init__(self, data_path , pickle_path, budget, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        label = []
        img_path = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for i, class_name in enumerate(data_budget[budget]['patches'].keys()):
            img_path = img_path + [os.path.join(class_name, x) for x in  data_budget[budget]['patches'][class_name]]
            num_imgs = len(data_budget[budget]['patches'][class_name])
            label = label + [i]*num_imgs
            print(f'number of images in class {class_name} are {num_imgs}')
        self.imgs = img_path
        self.labels = label

    def __getitem__(self,index):
        filename = self.imgs[index]
        img = cv2.imread(os.path.join(self.path, filename))
        label = self.labels[index]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)

        # resize image
        dim = (128, 128)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        if self.augment:
            main_img = self.augment(main_img.shape)(image=main_img)
            main_img = main_img['image']
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)
        label = torch.from_numpy(np.array(label)).long()
        return main_img, label

    def __len__(self):
            return len(self.imgs)

class Histodata_jigsaw(Dataset):
    def __init__(self, data_path, pickle_path, budget, augment=False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        label = []
        img_path = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for i, class_name in enumerate(data_budget[budget]['patches'].keys()):
            img_path = img_path + [os.path.join(class_name, x) for x in data_budget[budget]['patches'][class_name]]
            num_imgs = len(data_budget[budget]['patches'][class_name])
            label = label + [i] * num_imgs
            print(f'number of images in class {class_name} are {num_imgs}')
        self.imgs = img_path
        self.labels = label

    def __getitem__(self,index):

        filename = self.imgs[index]
        img = cv2.imread(os.path.join(self.path, filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)
        # resize image
        if config.dataset == 'kather':
            dim = (512, 512)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        jig_img, jig_label = jigsaw_res(img)
        jig_img = preprocess_input(jig_img.astype(np.float32))
        jig_img = torch.from_numpy(jig_img).float()
        jig_img = jig_img.permute(2, 0, 1)
        jig_label = torch.from_numpy(np.array(jig_label)).long()
        return jig_img, jig_label

    def __len__(self):
        return len(self.imgs)

class Histodata_magnification(Dataset):

    def __init__(self, data_path , pickle_path, budget, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        label = []
        img_path = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for i, class_name in enumerate(data_budget[budget]['patches'].keys()):
            img_path = img_path + [os.path.join(class_name, x) for x in  data_budget[budget]['patches'][class_name]]
            num_imgs = len(data_budget[budget]['patches'][class_name])
            label = label + [i]*num_imgs
            print(f'number of images in class {class_name} are {num_imgs}')
        self.imgs = img_path
        self.labels = label

    def __getitem__(self,index):
        filename = self.imgs[index]
        img = cv2.imread(os.path.join(self.path, filename))
        label = self.labels[index]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)

        # resize image
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
        return aux_image_mag, aux_label_mag

    def __len__(self):
            return len(self.imgs)

class Histodata_flip(Dataset):
    def __init__(self, data_path , pickle_path, budget, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        label = []
        img_path = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for i, class_name in enumerate(data_budget[budget]['patches'].keys()):
            img_path = img_path + [os.path.join(class_name, x) for x in  data_budget[budget]['patches'][class_name]]
            num_imgs = len(data_budget[budget]['patches'][class_name])
            label = label + [i]*num_imgs
            print(f'number of images in class {class_name} are {num_imgs}')
        self.imgs = img_path
        self.labels = label

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
        aux_label_flip = 1
        if random.random() > 0.5:
            main_img = cv2.flip(main_img, 1)
            aux_label_flip = 0
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)
        return main_img, aux_label_flip

    def __len__(self):
            return len(self.imgs)

class Histodata_rot(Dataset):
    def __init__(self, data_path , pickle_path, budget, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
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
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)
        cv2.flip(img, 1)
        # resize image
        width = int(img.shape[1] * 25 / 100)
        height = int(img.shape[0] * 25 / 100)
        dim = (width, height)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        angle = np.random.choice([0, 90, 180, 270], 1)
        if angle == 90:
            main_img = imutils.rotate(main_img, angle)
            aux_label_rot = 1
        elif angle == 180:
            main_img = imutils.rotate(main_img, angle)
            aux_label_rot = 2
        elif angle == 270:
            main_img = imutils.rotate(main_img, angle)
            aux_label_rot = 3
        else:
            aux_label_rot = 0
        main_img = preprocess_input(main_img.astype(np.float32))
        main_img = torch.from_numpy(main_img).float()
        main_img = main_img.permute(2, 0, 1)
        return main_img, aux_label_rot

    def __len__(self):
        return len(self.imgs)

class Histodata_auto(Dataset):
    def __init__(self, data_path , pickle_path, budget, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            self.n.fit(i1)
        label = []
        img_path = []
        with open(pickle_path, 'rb') as f:
            data_budget = pickle.load(f)
        for i, class_name in enumerate(data_budget[budget]['patches'].keys()):
            img_path = img_path + [os.path.join(class_name, x) for x in  data_budget[budget]['patches'][class_name]]
            num_imgs = len(data_budget[budget]['patches'][class_name])
            label = label + [i]*num_imgs
            print(f'number of images in class {class_name} are {num_imgs}')
        self.imgs = img_path
        self.labels = label

    def __getitem__(self,index):
        filename = self.imgs[index]
        img = cv2.imread(os.path.join(self.path, filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if config.stain_normalized:
            img = self.n.transform(img)
        cv2.flip(img, 1)
        # resize image
        width = int(img.shape[1] * 25 / 100)
        height = int(img.shape[0] * 25 / 100)
        dim = (width, height)
        main_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        main_img = preprocess_input (main_img.astype(np.float32))
        main_img_out = torch.from_numpy(main_img).float()
        main_label = torch.from_numpy(np.array(main_img)).long()
        main_img_out = main_img_out.permute(2, 0, 1)
        main_label = main_label.permute(2, 0, 1)
        return main_img_out, main_label

    def __len__(self):
        return len(self.imgs)

class Histodata_unlabel_domain_adopt(Dataset):
    def __init__(self, data_path , pickle_path, budget, augment = False):
        self.path = data_path
        self.augment = augment
        if config.stain_normalized:
            self.n = stainNorm_Reinhard.Normalizer()
            i1 = cv2.imread('./data/source.png')
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
        print(f'number of normal classes for unlabel : {len(normal_label)}')
        for image_name in data_budget[budget]['patches']['Tumour']:
            tumour_path.append(os.path.join('Tumour', image_name))
            tumour_label.append(1)
        print(f'number of tumour classes for unlabel : {len(tumour_label)}')
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
            if config.stain_normalized:
                img = self.n.transform(img)

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

            if 'hematoxylin' in config.task_names:
                ihc_hed = rgb2hed(main_img)
                hem_img = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
                hem_img = torch.from_numpy(np.array(hem_img)).long()
            main_img = preprocess_input (main_img.astype(np.float32))
            main_img = torch.from_numpy(main_img).float()
            main_img = main_img.permute(2, 0, 1)

            if 'magnification' in config.task_names and 'jigsaw' not in config.task_names and 'hematoxylin' not in config.task_names:
                return main_img,-1, aux_image_mag, aux_label_mag, -1, -1, -1
            elif 'jigsaw' in config.task_names and 'magnification' not in config.task_names and 'hematoxylin' not in config.task_names:
                return main_img, -1, -1, -1, jig_img, jig_label, -1
            elif 'hematoxylin' in config.task_names and 'magnification' not in config.task_names and 'jigsaw' not in config.task_names:
                return main_img, -1, -1, -1, -1, -1, hem_img
            elif 'hematoxylin' in config.task_names and 'magnification' in config.task_names and 'jigsaw' not in config.task_names:
                return main_img, -1, aux_image_mag, aux_label_mag, -1, -1, hem_img
            elif 'hematoxylin' in config.task_names and 'jigsaw' in config.task_names and 'magnification' not in config.task_names:
                return main_img, -1, -1, -1, jig_img, jig_label, hem_img
            elif 'magnification' in config.task_names and 'jigsaw' in config.task_names and 'hematoxylin' not in config.task_names:
                return main_img,-1, aux_image_mag, aux_label_mag, jig_img, jig_label, -1
            elif 'jigsaw' in config.task_names and 'magnification' in config.task_names and 'hematoxylin' in config.task_names:
                return main_img,-1, aux_image_mag, aux_label_mag, jig_img, jig_label, hem_img
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
    src_loader = DataLoader(lab_train_generator, batch_size=10, shuffle=True, num_workers=5,
                            pin_memory=True, worker_init_fn=worker_init_fn)
    for it, src in enumerate(src_loader):
        img,lbl,_,_, aux_img,aux_lbs = src
        img = img.permute(0, 2, 3, 1)
        aux_img = aux_img.permute(0, 2, 3, 1)
        show_images(aux_img.numpy(), cols=5, titles=aux_lbs.numpy())