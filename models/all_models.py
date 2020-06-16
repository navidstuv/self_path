import os
import time
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
from utils.utils import stats
from sklearn.metrics import roc_auc_score
# custom modules
from schedulers import get_scheduler
from models.encoder import Finetune
from optimizers import get_optimizer
from models.model import get_model
from utils.metrics import AverageMeter, accuracy
from utils.utils import to_device, make_inf_dl, save_output_img

# summary
from tensorboardX import SummaryWriter


class LinearRampdown(_LRScheduler):
    def __init__(self, opt, rampdown_from=1000, rampdown_till=1200, last_epoch=-1):
        self.rampdown_from = rampdown_from
        self.rampdown_till = rampdown_till
        super(LinearRampdown, self).__init__(opt, last_epoch)

    def ramp(self, e):
        if e > self.rampdown_from:
            f = (e - self.rampdown_from) / (self.rampdown_till - self.rampdown_from)
            return 1 - f
        else:
            return 1.0

    def get_lr(self):
        factor = self.ramp(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


class AuxModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = SummaryWriter(config.log_dir)
        cudnn.enabled = True

        # set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(config)
        if len(config.gpus) > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.best_acc = 0
        self.best_auc = 0

        if config.mode == 'train':
            # set up optimizer, lr scheduler and loss functions

            lr = config.lr
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.5, .999))
            self.scheduler = LinearRampdown(self.optimizer, rampdown_from=1000, rampdown_till=1200)

            self.class_loss_func = nn.CrossEntropyLoss()
            self.pixel_loss = nn.L1Loss()

            self.start_iter = 0

            # resume
            if config.training_resume:
                self.load(config.model_dir + '/' + config.training_resume)

            cudnn.benchmark = True

        elif config.mode == 'fine-tune':
            # set up optimizer, lr scheduler and loss functions

            self.lr = config.lr
            self.load(os.path.join(config.training_resume))
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.5, .999))
            # self.scheduler = LinearRampdown(self.optimizer, rampdown_from=1000, rampdown_till=1200)

            self.class_loss_func = nn.CrossEntropyLoss()
            # self.pixel_loss = nn.L1Loss()

            self.start_iter = 0

            cudnn.benchmark = True

        elif config.mode == 'val':
            self.load(os.path.join(self.config.testing_model))
        else:
            self.model = Finetune(self.model)
            self.model = self.model.to(self.device)
            self.load(os.path.join(self.config.testing_model))


    def entropy_loss(self, x):
        return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()

    def train_epoch_main_task(self, src_loader, tar_loader, epoch, print_freq):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        for it, src_batch in enumerate(src_loader):
            t = time.time()

            self.optimizer.zero_grad()
            src = src_batch
            src = to_device(src, self.device)
            src_imgs, src_cls_lbls, _, _, _, _ = src

            self.optimizer.zero_grad()

            src_main_logits = self.model(src_imgs, 'main_task')
            src_main_loss = self.class_loss_func(src_main_logits, src_cls_lbls)
            loss = src_main_loss * self.config.loss_weight['main_task']

            precision1_train, precision2_train = accuracy(src_main_logits, src_cls_lbls, topk=(1, 2))
            top1.update(precision1_train[0], src_imgs.size(0))

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), src_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - t)

            self.start_iter += 1

            if self.start_iter % print_freq == 0:
                print_string = 'Epoch {:>2} | iter {:>4} | loss:{:.3f}| acc:{:.3f}| src_main: {:.3f} |' + '|{:4.2f} s/it'
                self.logger.info(print_string.format(epoch, self.start_iter,
                                                     losses.avg,
                                                     top1.avg,
                                                     src_main_loss.item(),
                                                     batch_time.avg))
                self.writer.add_scalar('losses/all_loss', losses.avg, self.start_iter)
                self.writer.add_scalar('losses/src_main_loss', src_main_loss, self.start_iter)
        self.scheduler.step()

        # del loss, src_class_loss, src_aux_loss, tar_aux_loss, tar_entropy_loss
        # del src_aux_logits, src_class_logits
        # del tar_aux_logits, tar_class_logits
    def train_epoch_all_tasks(self, src_loader, epoch, print_freq):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        for it, src_batch in enumerate(src_loader):
            t = time.time()

            self.optimizer.zero_grad()
            src = src_batch
            src = to_device(src, self.device)
            src_imgs, src_cls_lbls, src_aux_mag_imgs, src_aux_mag_lbls, src_aux_jigsaw_imgs, src_aux_jigsaw_lbls, src_aux_hem_lbls = src


            loss = 0

            src_aux_loss = {}

            if 'magnification' in self.config.task_names:
                _,src_aux_mag_logits = self.model(src_aux_mag_imgs, 'magnification')
                src_aux_loss['magnification'] = self.class_loss_func(src_aux_mag_logits, src_aux_mag_lbls)
                loss += src_aux_loss['magnification'] * self.config.loss_weight[
                    'magnification']  # todo: magnification weight

            if 'jigsaw' in self.config.task_names:
                _,src_aux_jigsaw_logits = self.model(src_aux_jigsaw_imgs, 'jigsaw')
                src_aux_loss['jigsaw'] = self.class_loss_func(src_aux_jigsaw_logits, src_aux_jigsaw_lbls)
                loss += src_aux_loss['jigsaw'] * self.config.loss_weight['jigsaw']  # todo: main task weight

            if 'hematoxylin' in self.config.task_names:
                _, src_aux_hem_logits = self.model(src_imgs, 'hematoxylin')
                src_aux_loss['hematoxylin'] = self.pixel_loss(src_aux_hem_logits, src_aux_hem_lbls)
                loss += src_aux_loss['hematoxylin'] * self.config.loss_weight['hematoxylin']  # todo: main task weight


            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), src_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - t)

            self.start_iter += 1

            if self.start_iter % print_freq == 0:
                printt = ''
                for task_name in self.config.aux_task_names:
                    printt = printt + 'src_aux_' + task_name + ': {:.3f} |'
                print_string = 'Epoch {:>2} | iter {:>4} | loss:{:.3f} |' + printt + '{:4.2f} s/it'
                src_aux_loss_all = [loss.item() for loss in src_aux_loss.values()]
                self.logger.info(print_string.format(epoch, self.start_iter,
                                                     losses.avg,
                                                     *src_aux_loss_all,
                                                     batch_time.avg))
                self.writer.add_scalar('losses/all_loss', losses.avg, self.start_iter)
                for task_name in self.config.aux_task_names:
                    self.writer.add_scalar('losses/src_aux_loss_' + task_name, src_aux_loss[task_name],
                                               self.start_iter)

        self.scheduler.step()

        # del loss, src_class_loss, src_aux_loss, tar_aux_loss, tar_entropy_loss
        # del src_aux_logits, src_class_logits
        # del tar_aux_logits, tar_class_logits

    def train(self, src_loader):

        num_batches = len(src_loader)
        print_freq = max(num_batches // self.config.training_num_print_epoch, 1)
        start_epoch = self.start_iter // num_batches
        num_epochs = self.config.num_epochs
        for epoch in range(start_epoch, num_epochs):
            self.train_epoch_all_tasks(src_loader, epoch, print_freq)
            # validation
            self.save(self.config.model_dir, 'last')

    def save(self, path, ext):
        state = {"iter": self.start_iter + 1,
                 "model_state": self.model.state_dict(),
                 "optimizer_state": self.optimizer.state_dict(),
                 "scheduler_state": self.scheduler.state_dict(),
                 "best_acc": self.best_acc,
                 }
        save_path = os.path.join(path, f'model_{ext}.pth')
        self.logger.info('Saving model to %s' % save_path)
        torch.save(state, save_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.logger.info('Loaded model from: ' + path)

        if self.config.mode == 'train':
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_iter = checkpoint['iter']
            self.best_acc = checkpoint['best_acc']
            self.best_auc= checkpoint['best_auc']
            self.logger.info('Start iter: %d ' % self.start_iter)

    def fine_tune(self, src_loader, val_loader, test_loader):
        num_batches = len(src_loader)
        print_freq = max(num_batches // self.config.training_num_print_epoch, 1)
        start_epoch = self.start_iter // num_batches
        num_epochs = self.config.num_epochs
        self.model = Finetune(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(.5, .999))
        self.scheduler = LinearRampdown(self.optimizer, rampdown_from=1000, rampdown_till=1200)
        self.model.train()
        for epoch in range(start_epoch, num_epochs):
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top2 = AverageMeter()
            for it, src_batch in enumerate(src_loader):
                t = time.time()

                self.optimizer.zero_grad()
                src = src_batch
                src = to_device(src, self.device)
                src_imgs, src_cls_lbls, _, _, _, _,_ = src

                self.optimizer.zero_grad()

                src_main_logits = self.model(src_imgs)
                #for AUC
                smax = nn.Softmax(dim=1)
                smax_out = smax(src_main_logits)
                true_labels = src_cls_lbls.cpu().detach().numpy()
                pred_trh = smax_out.cpu().detach().numpy()[:, 1]


                src_main_loss = self.class_loss_func(src_main_logits, src_cls_lbls)
                loss = src_main_loss

                precision1_train, precision2_train = accuracy(src_main_logits, src_cls_lbls, topk=(1, 2))
                top1.update(precision1_train[0], src_imgs.size(0))

                AUC = roc_auc_score(true_labels, pred_trh)
                top2.update(AUC, src_imgs.size(0))

                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), src_imgs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - t)

                self.start_iter += 1

                if self.start_iter % print_freq == 0:
                    print_string = 'Epoch {:>2} | iter {:>4} | loss:{:.3f}| acc:{:.3f}| AUC:{:.3f}| src_main: {:.3f} |' + '|{:4.2f} s/it'
                    self.logger.info(print_string.format(epoch, self.start_iter,
                                                         losses.avg,
                                                         top1.avg,
                                                         top2.avg,
                                                         src_main_loss.item(),
                                                         batch_time.avg))
                    self.writer.add_scalar('losses/all_loss', losses.avg, self.start_iter)
                    self.writer.add_scalar('losses/src_main_loss', src_main_loss, self.start_iter)
            self.scheduler.step()
            self.save(self.config.model_dir, 'last')

            if val_loader is not None:
                self.logger.info('validating...')
                class_acc, auc = self.test(val_loader)
                # self.writer.add_scalar('val/aux_acc', class_acc, i_iter)
                self.writer.add_scalar('val/class_acc', class_acc, self.start_iter)
                self.writer.add_scalar('val/auc', auc, self.start_iter)
                if class_acc > self.best_acc:
                    self.best_acc = class_acc
                    self.save(self.config.best_model_dir, 'best_acc')
                if auc > self.best_auc:
                    self.best_auc = auc
                    self.save(self.config.best_model_dir, 'best_auc')
                    # todo copy current model to best model
                self.logger.info('Best validation accuracy: {:.2f} %'.format(self.best_acc))
                self.logger.info('Best validation auc: {:.2f} %'.format(self.best_auc))

            if test_loader is not None:
                self.logger.info('testing...')
                class_acc, auc = self.test(test_loader)
                # self.writer.add_scalar('test/aux_acc', class_acc, i_iter)
                self.writer.add_scalar('test/class_acc', class_acc, self.start_iter)
                self.writer.add_scalar('test/auc', auc, self.start_iter)
                # if class_acc > self.best_acc:
                #     self.best_acc = class_acc
                # todo copy current model to best model
                self.logger.info('testing accuracy: {:.2f} %'.format(class_acc))
                self.logger.info('testing auc: {:.2f} %'.format(auc))

        self.logger.info('Best validation accuracy: {:.2f} %'.format(self.best_acc))
        self.logger.info('Finished Training.')

    def feature_extractor(self, val_loader):
        val_loader_iterator = iter(val_loader)
        num_val_iters = len(val_loader)
        tt = tqdm(range(num_val_iters), total=num_val_iters, desc="Validating")
        kk = 1
        aux_correct = 0
        class_correct = 0
        total = 0
        soft_labels = np.zeros((1, 2))
        true_labels = []
        features = np.empty((1,2048))


        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:

                data = next(val_loader_iterator)
                data = to_device(data, self.device)
                imgs, cls_lbls, _, _, _, _,_ = data
                # Get the inputs

                feature, _ = self.model(imgs, self.config.task_names[0])

                if self.config.save_output == True:

                    maxpool = nn.AdaptiveMaxPool2d(1)
                    x = maxpool(feature)
                    x = x.reshape(x.size(0), -1)
                    true_labels = np.append(true_labels, cls_lbls.cpu().numpy())
                    features = np.append(features, x.cpu().numpy(), axis=0)
                    kk += 1
            tt.close()
        if self.config.save_output == True:
            np.save('true_train_main_' + self.config.mode + '.npy', true_labels)
            np.save('features_train_main_' + self.config.mode + '.npy', features[1:,:])


    def test(self, val_loader):
        val_loader_iterator = iter(val_loader)
        num_val_iters = len(val_loader)
        tt = tqdm(range(num_val_iters), total=num_val_iters, desc="Validating")
        kk = 1
        aux_correct = 0
        class_correct = 0
        total = 0
        soft_labels = np.zeros((1, 2))
        true_labels = []

        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:

                data = next(val_loader_iterator)
                data = to_device(data, self.device)
                imgs, cls_lbls, _, _, _, _,_ = data
                # Get the inputs

                logits = self.model(imgs)

                if self.config.save_output == True:
                    smax = nn.Softmax(dim=1)
                    smax_out = smax(logits)
                    soft_labels = np.concatenate((soft_labels, smax_out.cpu().numpy()), axis=0)
                    true_labels = np.append(true_labels, cls_lbls.cpu().numpy())
                    # pred_trh = smax_out.cpu().numpy()[:, 1]
                    # pred_trh[pred_trh >= 0.5] = 1
                    # pred_trh[pred_trh < 0.5] = 0
                    # compare = cls_lbls.cpu().numpy() - pred_trh
                    # FP_idx = np.where(compare == -1)
                    # FN_idx = np.where(compare == 1)
                    # FP_imgs = imgs.cpu().numpy()[FP_idx, ...]
                    # FN_imgs = imgs.cpu().numpy()[FN_idx, ...]
                    # save_output_img(FP_imgs[0, ...], 'FP_images', 'FP', kk * imgs.shape[0])
                    # save_output_img(FN_imgs[0, ...], 'FN_images', 'FN', kk * imgs.shape[0])
                    kk += 1

                _, cls_pred = logits.max(dim=1)
                # _, aux_pred = aux_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls)
                # aux_correct += torch.sum(aux_pred == aux_lbls.data)
                total += imgs.size(0)

            tt.close()
        if self.config.save_output == True:
            soft_labels = soft_labels[1:, :]
            np.save('pred_' + self.config.mode + '.npy', soft_labels)
            np.save('true_' + self.config.mode + '.npy', true_labels)
            auc = stats(soft_labels, true_labels, opt_thresh=0.5)

        # aux_acc = 100 * float(aux_correct) / total
        class_acc = 100 * float(class_correct) / total
        self.logger.info('class_acc: {:.2f} %'.format(class_acc))
        return class_acc,auc
